import sys
import os
import torch
import torch.nn.functional as F
import traceback
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

def test_mode(transformer_flag):
    print("\n" + "=" * 80)
    print(f"🚀 正在测试模式: use_transformer = {transformer_flag}")
    print("=" * 80)

    # 1. 初始化模型并动态设置 transformer_vint 开关
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
        transformer_vint=transformer_flag, # <--- 动态切换 True 或 False
        mha_num_attention_heads=2,
        mha_num_attention_layers=2,
        mha_ff_dim_factor=4,
    ).to(device).eval()

    # 2. 构造虚拟数据
    B, C, H, W = 1, 3, 224, 224  
    obs_img = torch.randn(B, C, H, W).to(device)
    goal_img = torch.randn(B, C, H, W).to(device)
    history_imgs = torch.randn(B, 5, C, H, W).to(device)

    # 3. 运行前向传播
    try:
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            dist, action = model(
                obs_img=obs_img,
                goal_img=goal_img,
                history_imgs=history_imgs
            )
        print(f"✅ 【测试通过】模式 (use_transformer={transformer_flag}) 完美运行！")
        print(f"   距离预测输出: {dist.shape}")
        print(f"   动作预测输出: {action.shape}")
        
    except Exception as e:
        print(f"❌ 【测试失败】模式 (use_transformer={transformer_flag}) 发生崩溃！")
        print("-" * 50)
        traceback.print_exc()  # 打印详细错误栈，方便你修 Bug
        print("-" * 50)

if __name__ == "__main__":
    print("📺 DepthAnything3 双模式架构黑盒测试启动")
    
    # 连续跑两遍测试
    test_mode(transformer_flag=True)
    test_mode(transformer_flag=False)