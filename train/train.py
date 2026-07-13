import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb
# 精准屏蔽 diffusers 导致的 _pytree 弃用警告
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch.utils._pytree._register_pytree_node.*")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

import sys
sys.path.append("/home/yyz/visualnav-transformer/train/vint_train/models/gnm")
"""
IMPORT YOUR MODEL HERE
"""
# from vint_train.models.gnm.gnm import GNM
# from vint_train.models.vint.vint import ViNT
# from vint_train.models.vint.vit import ViT
# from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
# from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
# from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
# 👇 直接导入da3.py（100%成功，无任何依赖）
from depth_anything_3.model.da3 import DepthAnything3Net
from omegaconf import OmegaConf
from safetensors.torch import load_file  # 新增：safetensors加载工具

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.data.vint_dataset_da3 import ViNT_Dataset_DA3
from vint_train.training.train_eval_loop import (
    train_eval_loop,
    # train_eval_loop_nomad,
    load_model,
)


def _set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def _unfreeze_da3_last_blocks(model, num_blocks, train_norm=True):
    if num_blocks <= 0:
        return []

    backbone = model.backbone
    pretrained = getattr(backbone, "pretrained", None)
    blocks = getattr(pretrained, "blocks", None)
    if blocks is None:
        print("⚠️  未找到 DA3 backbone blocks，无法按层解冻")
        return []

    total_blocks = len(blocks)
    num_blocks = min(num_blocks, total_blocks)
    start_idx = total_blocks - num_blocks
    unfrozen = []
    for block_idx in range(start_idx, total_blocks):
        _set_requires_grad(blocks[block_idx], True)
        unfrozen.append(f"backbone.pretrained.blocks.{block_idx}")

    if train_norm and hasattr(pretrained, "norm"):
        _set_requires_grad(pretrained.norm, True)
        unfrozen.append("backbone.pretrained.norm")

    return unfrozen


def _unfreeze_da3_backbone(model):
    _set_requires_grad(model.backbone, True)
    return ["backbone"]


def _build_optimizer(model, config):
    lr = float(config["lr"])
    optimizer_name = config["optimizer"].lower()

    if config["model_type"] == "da3":
        backbone_param_ids = {
            id(param)
            for param in model.backbone.parameters()
            if param.requires_grad
        }
        backbone_params = [
            param for param in model.backbone.parameters() if param.requires_grad
        ]
        nav_params = [
            param
            for param in model.parameters()
            if param.requires_grad and id(param) not in backbone_param_ids
        ]
        params = []
        if nav_params:
            params.append({"params": nav_params, "lr": lr})
        if backbone_params:
            params.append({
                "params": backbone_params,
                "lr": float(config.get("da3_backbone_lr", lr * 0.1)),
            })
    else:
        params = [param for param in model.parameters() if param.requires_grad]

    if optimizer_name == "adam":
        return Adam(params, lr=lr, betas=(0.9, 0.98))
    if optimizer_name == "adamw":
        return AdamW(params, lr=lr)
    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Optimizer {config['optimizer']} not supported")


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if "gpu_ids" not in config:
        config["gpu_ids"] = [0]
    elif type(config["gpu_ids"]) == int:
        config["gpu_ids"] = [config["gpu_ids"]]

    config["gpu_ids"] = [int(x) for x in config["gpu_ids"]]
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    if torch.cuda.is_available():
        print("Using cuda devices:", config["gpu_ids"])
    else:
        print("Using cpu")

    device = torch.device(f"cuda:{config['gpu_ids'][0]}" if torch.cuda.is_available() else "cpu")

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = False
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    # Check if we should use DA3 preprocessing
    use_da3_preprocessing = config.get("use_da3_preprocessing", False)

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                if use_da3_preprocessing:
                    # Use DA3 dataset class
                    dataset = ViNT_Dataset_DA3(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                        da3_process_res=config.get("da3_process_res", 504),
                        da3_process_res_method=config.get("da3_process_res_method", "upper_bound_resize"),
                        da3_num_workers=config.get("da3_num_workers", 1),
                        load_original_images=config.get("load_original_images", config.get("num_images_log", 8) > 0),
                    )
                else:
                    # Use original dataset class
                    dataset = ViNT_Dataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                    )
                if data_split_type == "train":
                    train_dataset.append(dataset)
                else:
                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_dataloaders:
                        test_dataloaders[dataset_type] = {}
                    test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    # elif config["model_type"] == "nomad":
    #     if config["vision_encoder"] == "nomad_vint":
    #         vision_encoder = NoMaD_ViNT(
    #             obs_encoding_size=config["encoding_size"],
    #             context_size=config["context_size"],
    #             mha_num_attention_heads=config["mha_num_attention_heads"],
    #             mha_num_attention_layers=config["mha_num_attention_layers"],
    #             mha_ff_dim_factor=config["mha_ff_dim_factor"],
    #         )
    #         vision_encoder = replace_bn_with_gn(vision_encoder)
    #     elif config["vision_encoder"] == "vib": 
    #         vision_encoder = ViB(
    #             obs_encoding_size=config["encoding_size"],
    #             context_size=config["context_size"],
    #             mha_num_attention_heads=config["mha_num_attention_heads"],
    #             mha_num_attention_layers=config["mha_num_attention_layers"],
    #             mha_ff_dim_factor=config["mha_ff_dim_factor"],
    #         )
    #         vision_encoder = replace_bn_with_gn(vision_encoder)
    #     elif config["vision_encoder"] == "vit": 
    #         vision_encoder = ViT(
    #             obs_encoding_size=config["encoding_size"],
    #             context_size=config["context_size"],
    #             image_size=config["image_size"],
    #             patch_size=config["patch_size"],
    #             mha_num_attention_heads=config["mha_num_attention_heads"],
    #             mha_num_attention_layers=config["mha_num_attention_layers"],
    #         )
    #         vision_encoder = replace_bn_with_gn(vision_encoder)
    #     else: 
    #         raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
    elif config["model_type"] == "da3":
        # 加载 DA3 官方配置
        da3_cfg = OmegaConf.load("/home/yyz/visualnav-transformer/train/vint_train/models/gnm/depth_anything_3/configs/da3-small.yaml")
        # 初始化你的 DA3 导航模型（参数完全对齐配置）
        model = DepthAnything3Net(
            net=da3_cfg.net,
            head=da3_cfg.head,
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            embed_dim=384,
            transformer_vint=False,  # DA3不使用ViNT结构，保持原有设计
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            nav_fusion_mode=config.get("nav_fusion_mode", "vint_like"),
        )
            
        # ===================== 【加载 DA3 预训练权重 + 冻结主干】 =====================
        try:
            # 👇 这里填你的本地预训练模型路径（官方权重文件夹/文件）
            model_dir= "/home/yyz/depth-anything-3/DA3-SMALL"
        
            # 加载safetensor权重（注意文件名：如果是model.safetensor就去掉s）
            safetensor_path = os.path.join(model_dir, "model.safetensors")
            pretrain_dict = load_file(safetensor_path, device="cpu")
            
            # 加载权重：strict=False 忽略你新增的导航层（decoder/MLP/预测头）
            model.load_state_dict(pretrain_dict, strict=False)
            print("✅ DA3 safetensors预训练权重加载成功！")

            # 冻结主干网络（只训练你新增的导航/融合层）
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = False
            if bool(config.get("da3_unfreeze_all_backbone", False)):
                unfrozen = _unfreeze_da3_backbone(model)
            else:
                unfrozen = _unfreeze_da3_last_blocks(
                    model,
                    int(config.get("da3_unfreeze_last_blocks", 0)),
                    train_norm=bool(config.get("da3_unfreeze_norm", True)),
                )
            if unfrozen:
                print(f"✅ 已解冻 DA3 backbone 参数: {', '.join(unfrozen)}")
            else:
                print("✅ 已冻结主干，仅训练导航模块！")

        except Exception as e:
            print(f"⚠️  预训练权重加载失败: {e}")
            print("⚠️  将从头训练整个模型")    
        # noise_pred_net = ConditionalUnet1D(
        #         input_dim=2,
        #         global_cond_dim=config["encoding_size"],
        #         down_dims=config["down_dims"],
        #         cond_predict_scale=config["cond_predict_scale"],
        #     )
        # dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        # model = NoMaD(
        #     vision_encoder=vision_encoder,
        #     noise_pred_net=noise_pred_net,
        #     dist_pred_net=dist_pred_network,
        # )

        # noise_scheduler = DDPMScheduler(
        #     num_train_timesteps=config["num_diffusion_iters"],
        #     beta_schedule='squaredcos_cap_v2',
        #     clip_sample=True,
        #     prediction_type='epsilon'
        # )
    else:
        raise ValueError(f"Model {config['model']} not supported")

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    config["optimizer"] = config["optimizer"].lower()
    optimizer = _build_optimizer(model, config)
    lr = float(config["lr"])

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path) #f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, config["model_type"], latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    # Multi-GPU
    model = model.to(device)
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())
# 🔥 把 da3 加入普通训练流程，彻底抛弃 NoMaD
    if config["model_type"] == "vint" or config["model_type"] == "gnm" or config["model_type"] == "da3": 
        train_eval_loop(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config.get("eval_freq", 1),
        )
    # DA3 永远不会走到这里，无需处理 NoMaD
    else:
        raise ValueError(f"不支持的模型类型: {config['model_type']}")


    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config["use_wandb"]:
        wandb.login()
        # 兼容配置格式：修复 OmegaConf / Dict 报错
        if isinstance(config, dict):
            wandb_config = config
        else:
            wandb_config = OmegaConf.to_container(config, resolve=True)

        # 初始化：自动创建项目，无403权限错误 + 无属性报错
        wandb.init(
            project="da3-visual-navigation",  # 自动新建项目，无权限问题
            config=wandb_config,              # 兼容配置
            name=f"da3_epoch_{config['epochs']}",  # 字典用 [] 取值，修复属性报错
        )
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)
