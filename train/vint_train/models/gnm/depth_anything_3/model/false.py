import sys
import os
from xml.parsers.expat import model
import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf

# ===================== 自动修复项目路径 =====================
file_path = Path(__file__).resolve()
current_dir = file_path.parent
root_dir = current_dir.parent  # depth_anything_3 根目录
if str(root_dir.parent) not in sys.path:
    sys.path.insert(0, str(root_dir.parent))
os.chdir(root_dir)

# ===================== 导入真实模块 =====================
from depth_anything_3.model.da3 import DepthAnything3Net

# ===================== 配置 =====================
CONFIG_PATH = os.path.join(root_dir, "configs", "da3-small.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 超详细数据流测试 =====================
def test_data_flow():
    print("=" * 80)
    print("📺 原始 DepthAnything3 模型 - [else 分支] 数据流原貌复原")
    print("=" * 80)

    # 1. 加载配置 + 初始化模型
    cfg = OmegaConf.load(CONFIG_PATH)
    model = DepthAnything3Net(
        net=cfg.net,
        head=cfg.head,
        cam_dec=None,
        cam_enc=None,
        gs_head=None,
        gs_adapter=None,
        context_size=5,
        len_traj_pred=5,
        learn_angle=True,
        embed_dim=384,
        transformer_vint=False,  # 🔥 走 else 分支
        mha_num_attention_heads=2,
        mha_num_attention_layers=2,
        mha_ff_dim_factor=4,
    ).to(device).eval()
    print("\n✅ 模型初始化完成 (transformer_vint=False)")

    # 2. 构造输入
    B, C, H, W = 1, 3, 224, 224  
    obs_img = torch.randn(B, C, H, W).to(device)
    goal_img = torch.randn(B, C, H, W).to(device)
    history_imgs = torch.randn(B, 5, C, H, W).to(device)

    # 3. 逐步骤执行 forward
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        print(f"\n🔹 【步骤1】拼接输入序列")
        x = torch.cat([history_imgs, obs_img.unsqueeze(1), goal_img.unsqueeze(1)], dim=1)
        print(f"   x 维度: {x.shape} (B=1, N=7, C=3, H=224, W=224)")

        print(f"\n🔹 【步骤2】Backbone 提取")
        feats, aux_feats = model.backbone(x, cam_token=None, export_feat_layers=[5,7,9,11])
        print(f"   aux_feats[0] 维度: {aux_feats[0].shape}")

        print(f"\n🔹 【步骤3】深度头预测")
        feats_obs = tuple([tuple([feat[:, 5:6] for feat in feat_group]) for feat_group in feats])
        output = model._process_depth_head(feats_obs, H, W)
        curr_obs_depth = output.depth 
        print(f"   深度图维度: {curr_obs_depth.shape}")

        print(f"\n🔹 【步骤4】深度 PatchEmbed")
        depth_embed = model.depth_patch_embed(curr_obs_depth)
        print(f"   depth_embed 维度: {depth_embed.shape} | (B, 256, 384)")

        print(f"\n🔹 【步骤5】复刻 else 分支逻辑开始 (提取特征)")
        # 你的原代码：obs_and_his = aux_feats[0][:, 0:6, :, :]
        obs_and_his = aux_feats[0][:, 0:6, :, :] 
        print(f"   obs_and_his 维度: {obs_and_his.shape} | (B, 6, 256, 384)")
        
        # 你的原代码：obs_and_his = obs_and_his.mean(dim=2)
        obs_and_his = obs_and_his.mean(dim=2)
        print(f"   执行 .mean(dim=2) 后维度: {obs_and_his.shape} | (B, 6, 384)")

        print(f"\n🔹 【步骤6】复刻 else 分支：过 Transformer")
        goal_visual = aux_feats[0][:, 6, :, :]  
        obs_and_his = model.decoder2(obs_and_his)
        print(f"   Transformer 输出维度: {obs_and_his.shape}")
                
        print(f"\n🔹 【步骤7】复刻 else 分支：与 depth_embed 拼接")
        # 你的原代码：obs_his_cat = torch.cat([obs_and_his, depth_embed], -1)
        obs_his_feature_expanded = obs_and_his.unsqueeze(1)
        
        # 🔥 关键修复 2：把中间的 1 扩展成 256 -> (B, 256, 32)
        obs_his_feature = obs_his_feature_expanded.expand(-1, 256, -1)
        obs_his_cat = torch.cat([obs_his_feature, depth_embed], -1) # (B, 6, 768)
        print(f"   拼接维度: {obs_his_feature.shape} | (B, 6, 768)")
        print(f"   准备拼接: obs_and_his {obs_his_feature.shape} 和 depth_embed {depth_embed.shape}")
        print(f"   拼接后维度: {obs_his_cat.shape}")
        obs_his_mlp = model.swin_MLP_linear_false(obs_his_cat) # (B, 6, 384)
        print(f"  MLP维度: {obs_his_mlp.shape} ")


        attn_out, _ = model.final_cross_attn(
            query=obs_his_mlp, 
            key=goal_visual, 
            value=goal_visual
        )
        print(f"   交叉注意力输出维度: {attn_out.shape} | (B, 1, 384)")

        print(f"\n🔹 【步骤9】最后的降维压缩 (384 -> 32)")
        # 去掉多余的序列维度 1 -> (B, 384)
        # attn_out = attn_out.squeeze(1)  
        
        print(f"   降维输出维度: {attn_out.shape} | (B, 384)")
        print(f"\n🔹 【步骤10】最后的融合 (384 -> 32)")
        # 过专属的压缩层 -> (B, 32)
        final_repr = model.final_compressor(attn_out)
        print(f"   降维输出维度: {final_repr.shape} | (B, 32)")
        final_repr = final_repr.mean(1)
        print(f"   最终融合全局特征: {final_repr.shape} | (B, 32) 完美！")

        dist_pred = model.dist_predictor(final_repr)
        action_pred = model.action_predictor(final_repr)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], model.len_traj_pred, model.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        if model.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        print(f"   距离预测 dist_pred: {dist_pred.shape}")
        print(f"   动作预测 action_pred: {action_pred.shape} | (B, 5, 4) 正确")

if __name__ == "__main__":
    test_data_flow()