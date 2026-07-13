import sys
import os
# Add the models directory to the path so we can import depth_anything_3
models_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(models_dir))

import torch
import torch.nn as nn
from depth_anything_3.model.da3 import DepthAnything3  # Use direct import from the local structure

def test_depth_anything3_flow():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型实例
    model = DepthAnything3().to(device)
    
    # 生成随机输入数据
    B, H, W = 2, 224, 224  # 批次大小、高度、宽度
    
    # 随机生成当前观测图像 (B, 3, H, W)
    obs_img = torch.randn(B, 3, H, W).to(device)
    
    # 随机生成目标图像 (B, 3, H, W)
    goal_img = torch.randn(B, 3, H, W).to(device)
    
    # 随机生成5张历史图像 (B, 5, 3, H, W)
    history_imgs = torch.randn(B, 5, 3, H, W).to(device)
    
    print("=== 输入数据形状 ===")
    print(f"obs_img shape: {obs_img.shape}")
    print(f"goal_img shape: {goal_img.shape}")
    print(f"history_imgs shape: {history_imgs.shape}")
    
    # 执行前向传播
    with torch.no_grad():
        dist, action = model(obs_img, goal_img, history_imgs)
    
    # 打印中间变量和最终输出的形状
    print("\n=== 中间变量形状 ===")
    print(f"dist shape: {dist.shape}")
    print(f"action shape: {action.shape}")
    
    # 验证输出是否符合预期
    expected_dist_shape = (B, 1)
    expected_action_shape = (B, 5, 4)
    
    assert dist.shape == expected_dist_shape, f"Expected dist shape {expected_dist_shape}, got {dist.shape}"
    assert action.shape == expected_action_shape, f"Expected action shape {expected_action_shape}, got {action.shape}"
    
    print("\n=== 测试通过！所有形状均符合预期 ===")

if __name__ == "__main__":
    test_depth_anything3_flow()