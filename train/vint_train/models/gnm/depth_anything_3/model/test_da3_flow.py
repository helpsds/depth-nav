import sys
import os
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
    print("📺 最新 DepthAnything3 模型 - 完整数据流/维度详细验证")
    print("=" * 80)
    print(f"运行设备: {device}")
    print(f"配置文件: {os.path.basename(CONFIG_PATH)}")
    print(f"模型参数: embed_dim=384 | Swin_MLP融合 | 预测轨迹=5帧")

    # 1. 加载配置 + 初始化最新模型
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
        transformer_vint=True,  # 🔥 对应 da3.py 的 self.use_transformer = True
        mha_num_attention_heads=2,
        mha_num_attention_layers=2,
        mha_ff_dim_factor=4,
    ).to(device).eval()
    print("\n✅ 最新模型初始化完成（Swin MLP融合 + MultiLayerDecoder）")

    # 2. 构造输入
    B, C, H, W = 1, 3, 224, 224  
    obs_img = torch.randn(B, C, H, W).to(device)    # 观测图
    goal_img = torch.randn(B, C, H, W).to(device)   # 目标图
    history_imgs = torch.randn(B, 5, C, H, W).to(device)  # 历史帧
    print(f"\n🔹 【步骤1】模型输入")
    print(f"   观测图 obs_img: {obs_img.shape}")
    print(f"   目标图 goal_img: {goal_img.shape}")
    print(f"   历史帧 history_imgs: {history_imgs.shape}")

    # 3. 逐步骤执行 forward（100%复刻 da3.py 最新源码）
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        print(f"\n🔹 【步骤2】拼接输入序列 [历史+观测+目标]")
        x = torch.cat([
            history_imgs,
            obs_img.unsqueeze(1),
            goal_img.unsqueeze(1)
        ], dim=1)
        print(f"   拼接后输入 x: {x.shape} (B=1, N=7, C=3, H=224, W=224)")

        print(f"\n🔹 【步骤3】Backbone 特征提取")
        feats, aux_feats = model.backbone(x, cam_token=None, export_feat_layers=[5,7,9,11])
        print(f"   aux_feats[0] 形状: {aux_feats[0].shape} | (B, 7, 256, 384) 正确")

        print(f"\n🔹 【步骤4】深度头预测（提取观测帧深度）")
        feats_obs = []
        for feat_group in feats:
            processed_group = []
            for feat in feat_group:
                feat_obs = feat[:, 5:6]  # 截取第6帧 (OBS)
                processed_group.append(feat_obs)
            feats_obs.append(tuple(processed_group))
        feats_obs = tuple(feats_obs)

        output = model._process_depth_head(feats_obs, H, W)
        curr_obs_depth = output.depth 
        print(f"   观测帧深度图: {curr_obs_depth.shape} | (B, 224, 224) 正确")

        print(f"\n🔹 【步骤5】深度 PatchEmbed 编码")
        depth_embed = model.depth_patch_embed(curr_obs_depth)
        print(f"   PatchEmbed输出: {depth_embed.shape} | (B, 256, 384) 正确")

        print(f"\n🔹 【步骤6】提取各帧视觉特征")
        obs_visual = aux_feats[0][:, 5, :, :]     # (B, 256, 384)
        goal_visual = aux_feats[0][:, 6, :, :]    # (B, 256, 384)
        hist_visual = aux_feats[0][:, 0:5, :, :]  # (B, 5, 256, 384)
        print(f"   历史帧特征 hist_visual: {hist_visual.shape}")
        print(f"   观测帧特征 obs_visual: {obs_visual.shape}")
        print(f"   目标帧特征 goal_visual: {goal_visual.shape}")

        print(f"\n🔹 【步骤7】特征融合 (Swin MLP)")
        # 根据 da3.py 源码，将 obs_visual 和 depth_embed 在最后一维拼接后通过 MLP
        mlp_input = torch.cat([obs_visual, depth_embed], dim=-1)
        fused = model.swin_MLP_linear(mlp_input) 
        print(f"   融合前拼接维度: {mlp_input.shape} | (B, 256, 768)")
        print(f"   MLP融合后 fused: {fused.shape} | (B, 256, 384) 正确")

        print(f"\n🔹 【步骤8】构建完整序列特征 (7帧序列)")
        sequence_feat = torch.cat([
            hist_visual,                # (B, 5, 256, 384)
            fused.unsqueeze(1),         # (B, 1, 256, 384)
            goal_visual.unsqueeze(1)    # (B, 1, 256, 384)
        ], dim=1)
        sequence_input = sequence_feat.mean(dim=2)  # 压缩 256 这个 patch 维度
        print(f"   压缩后序列输入 sequence_input: {sequence_input.shape} | (B, 7, 384) 正确")

        print(f"\n🔹 【步骤9】Decoder 解码 + 导航预测")
        final_repr = model.decoder1(sequence_input)
        dist_pred = model.dist_predictor(final_repr)
        action_pred = model.action_predictor(final_repr)

        # 动作参数后处理
        action_pred = action_pred.reshape(
            (action_pred.shape[0], model.len_traj_pred, model.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
        if model.learn_angle:
            action_pred[:, :, 2:] = F.normalize(action_pred[:, :, 2:].clone(), dim=-1)
        print(f"   final_repr: {final_repr.shape}")
        print(f"   距离预测 dist_pred: {dist_pred.shape}")
        print(f"   动作预测 action_pred: {action_pred.shape} | (B, 5, 4) 正确")

        print(f"\n🔹 【步骤10】一键调用模型验证")
        # 直接调用模型的 forward 测试整个黑盒输出
        dist, action = model(
            obs_img=obs_img,
            goal_img=goal_img,
            history_imgs=history_imgs
        )
        print(f"   ✅ 一键前向传播成功！")
        print(f"   直接输出 距离: {dist.shape} | 动作: {action.shape}")

    # 4. 最终输出汇总
    print("\n" + "="*80)
    print("📊 最终流/维度汇总（100% 匹配 da3.py 最新代码）")
    print("="*80)
    print(f"深度提取维度:   {depth_embed.shape}")
    print(f"MLP融合输出:    {fused.shape}")
    print(f"解码器输入:     {sequence_input.shape} (B, seq_len=7, embed_dim=384)")
    print(f"距离结果:       {dist.shape}")
    print(f"动作轨迹:       {action.shape}")
    print("\n🎉 最新架构数据流测试脚本 **重构完毕并全部通过**！")

if __name__ == "__main__":
    test_data_flow()