#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNM DA3 架构数据流维度验证脚本

此脚本详细验证 DepthAnything3Net 的每个组件的数据流维度，
确保从输入到输出的每个步骤都符合预期。
"""

import sys
import os
import torch
import torch.nn as nn

# 添加项目路径
models_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(models_dir))

from depth_anything_3.model.da3 import DepthAnything3Net


def create_mock_backbone():
    """创建模拟的 backbone 用于测试"""
    class MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, x, cam_token=None, export_feat_layers=None, ref_view_strategy="saddle_balanced"):
            B, N, C, H, W = x.shape
            # 模拟 DINOv2 输出
            # feats: 多层特征
            feats = [
                [torch.randn(B, N, 256, 768), torch.randn(B, N, 1, 768)],  # layer 0
                [torch.randn(B, N, 256, 768), torch.randn(B, N, 1, 768)],  # layer 1  
                [torch.randn(B, N, 256, 768), torch.randn(B, N, 1, 768)],  # layer 2
                [torch.randn(B, N, 256, 768), torch.randn(B, N, 1, 768)],  # layer 3
            ]
            # aux_feats: 辅助特征，只包含 layer 0
            aux_feats = [torch.randn(B, N, 256, 768)]
            return feats, aux_feats
    
    return MockBackbone()


def create_mock_head():
    """创建模拟的 depth head 用于测试"""
    class MockHead(nn.Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, feats, H, W, patch_start_idx=0):
            from addict import Dict
            B, N = feats[0][0].shape[:2]
            output = Dict()
            # 模拟深度输出 (B, N, H, W)
            output.depth = torch.randn(B, N, H, W)
            output.depth_conf = torch.randn(B, N, H, W)
            output.sky = torch.randn(B, N, H, W)
            return output
    
    return MockHead()


def test_da3_dataflow_detailed():
    """详细测试 DA3 数据流维度"""
    print("=" * 60)
    print("GNM DA3 架构数据流维度验证")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模拟组件
    mock_backbone = create_mock_backbone()
    mock_head = create_mock_head()
    
    # 创建模型实例
    model = DepthAnything3Net(
        net=mock_backbone,
        head=mock_head,
        context_size=5,
        len_traj_pred=5,
        learn_angle=True,
        embed_dim=384,
        use_transformer=False  # 先测试模式 3.1
    ).to(device)
    
    # 测试参数
    B, H, W = 2, 224, 224
    
    # 创建输入数据
    print("\n1. 输入数据准备:")
    history_imgs = torch.randn(B, 5, 3, H, W).to(device)  # 5张历史图像
    obs_img = torch.randn(B, 3, H, W).to(device)          # 1张当前观测
    goal_img = torch.randn(B, 3, H, W).to(device)         # 1张目标图像
    
    print(f"   history_imgs: {history_imgs.shape} (B, 5, 3, H, W)")
    print(f"   obs_img:      {obs_img.shape} (B, 3, H, W)")
    print(f"   goal_img:     {goal_img.shape} (B, 3, H, W)")
    
    # 组合输入
    x = torch.cat([history_imgs, obs_img.unsqueeze(1), goal_img.unsqueeze(1)], dim=1)
    print(f"   组合输入 x:   {x.shape} (B, 7, 3, H, W)")
    
    print("\n2. Backbone 特征提取:")
    with torch.no_grad():
        feats, aux_feats = model.backbone(x)
        print(f"   feats 长度: {len(feats)}")
        print(f"   feats[0][0] (patch_feat): {feats[0][0].shape} (B, 7, 256, 768)")
        print(f"   feats[0][1] (cls_feat):   {feats[0][1].shape} (B, 7, 1, 768)")
        print(f"   aux_feats 长度: {len(aux_feats)}")
        print(f"   aux_feats[0]:             {aux_feats[0].shape} (B, 7, 256, 768)")
    
    print("\n3. 深度估计:")
    with torch.no_grad():
        depth_output = model.head(feats, H, W)
        print(f"   depth_output.depth:       {depth_output.depth.shape} (B, 7, H, W)")
        curr_obs_depth = depth_output.depth[:, 5, :, :].unsqueeze(1)  # 当前观测深度
        print(f"   curr_obs_depth:           {curr_obs_depth.shape} (B, 1, H, W)")
        
        # 深度 patch embedding
        depth_embed = model.depth_patch_embed(curr_obs_depth)
        print(f"   depth_embed:              {depth_embed.shape} (B, embed_dim, H//14, W//14)")
        depth_global = depth_embed.mean(dim=(2, 3))
        print(f"   depth_global:             {depth_global.shape} (B, embed_dim)")
    
    print("\n4. 视觉特征提取:")
    with torch.no_grad():
        # 观测视觉特征
        obs_visual_patch = aux_feats[0][:, 5, :, :]  # 索引5是当前观测
        obs_visual = obs_visual_patch.mean(dim=1)
        print(f"   obs_visual_patch:         {obs_visual_patch.shape} (B, 256, 768)")
        print(f"   obs_visual:               {obs_visual.shape} (B, 768)")
        
        # 目标视觉特征  
        goal_visual_patch = aux_feats[0][:, 6, :, :]  # 索引6是目标
        goal_visual = goal_visual_patch.mean(dim=1)
        print(f"   goal_visual_patch:        {goal_visual_patch.shape} (B, 256, 768)")
        print(f"   goal_visual:              {goal_visual.shape} (B, 768)")
        
        # 历史视觉特征
        hist_visual_patch = aux_feats[0][:, 0:5, :, :]  # 索引0-4是历史
        hist_visual = hist_visual_patch.mean(dim=2)
        print(f"   hist_visual_patch:        {hist_visual_patch.shape} (B, 5, 256, 768)")
        print(f"   hist_visual:              {hist_visual.shape} (B, 5, 768)")
    
    print("\n5. 特征融合 (模式 3.1 - 无Transformer):")
    with torch.no_grad():
        # 融合观测视觉 + 深度全局特征
        fused = model.swin_MLP_linear(torch.cat([obs_visual, depth_global], 1))
        print(f"   fused (obs+depth):        {fused.shape} (B, 768)")
        
        # 聚合: fused_obs + 历史特征
        obs_hist = torch.cat([fused.unsqueeze(1), hist_visual], dim=1).mean(dim=1)
        print(f"   obs_hist (fused+hist):    {obs_hist.shape} (B, 768)")
        
        # 最终特征: obs_hist + goal_visual
        final = torch.cat([obs_hist, goal_visual], 1)
        print(f"   final (obs_hist+goal):    {final.shape} (B, 1536)")
        
        # 线性层处理
        z = model.linear_layers(final)
        print(f"   z (after linear layers):  {z.shape} (B, 32)")
    
    print("\n6. 导航预测输出:")
    with torch.no_grad():
        distance = model.dist_predictor(z)
        action = model.action_predictor(z).reshape(B, model.len_traj_pred, model.num_action_params)
        print(f"   distance:                 {distance.shape} (B, 1)")
        print(f"   action (before postproc): {action.shape} (B, 5, 4)")
        
        # 后处理
        action[:, :, :2] = torch.cumsum(action[:, :, :2], dim=1)
        action[:, :, 2:] = torch.nn.functional.normalize(action[:, :, 2:].clone(), dim=-1)
        print(f"   action (after postproc):  {action.shape} (B, 5, 4)")
    
    print("\n7. 完整前向传播测试:")
    with torch.no_grad():
        output = model(obs_img=obs_img, goal_img=goal_img, history_imgs=history_imgs)
        print(f"   output.distance:          {output.distance.shape}")
        print(f"   output.action:            {output.action.shape}")
    
    # 验证维度正确性
    assert output.distance.shape == (B, 1), f"距离维度错误: {output.distance.shape}"
    assert output.action.shape == (B, 5, 4), f"动作维度错误: {output.action.shape}"
    
    print("\n" + "=" * 60)
    print("✅ 所有数据流维度验证通过！")
    print("GNM DA3 架构数据流工作正常。")
    print("=" * 60)


def test_transformer_mode():
    """测试 Transformer 模式 (模式 3.2)"""
    print("\n" + "=" * 60)
    print("测试 Transformer 模式 (模式 3.2)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模拟组件
    mock_backbone = create_mock_backbone()
    mock_head = create_mock_head()
    
    # 创建使用 Transformer 的模型
    model = DepthAnything3Net(
        net=mock_backbone,
        head=mock_head,
        context_size=5,
        len_traj_pred=5,
        learn_angle=True,
        embed_dim=384,
        use_transformer=True  # 启用 Transformer 模式
    ).to(device)
    
    B, H, W = 2, 224, 224
    history_imgs = torch.randn(B, 5, 3, H, W).to(device)
    obs_img = torch.randn(B, 3, H, W).to(device)
    goal_img = torch.randn(B, 3, H, W).to(device)
    
    with torch.no_grad():
        # 提取特征
        _, aux_feats = model.backbone(torch.cat([
            history_imgs, obs_img.unsqueeze(1), goal_img.unsqueeze(1)
        ], dim=1))
        
        # Transformer 处理前6张图像 (历史+观测)
        six_visual_patch = aux_feats[0][:, 0:6, :, :]
        six_visual = six_visual_patch.mean(dim=2)
        print(f"   six_visual:               {six_visual.shape} (B, 6, 768)")
        
        trans_out = model.transformer(six_visual).mean(dim=1)
        print(f"   trans_out:                {trans_out.shape} (B, 768)")
        
        # 深度特征
        depth_output = model.head([[[torch.randn(B, 7, 256, 768), torch.randn(B, 7, 1, 768)]]*4, 
                                  [torch.randn(B, 7, 256, 768)], H, W)
        curr_obs_depth = depth_output.depth[:, 5, :, :].unsqueeze(1)
        depth_embed = model.depth_patch_embed(curr_obs_depth)
        depth_global = depth_embed.mean(dim=(2, 3))
        
        # 融合
        obs_hist = model.swin_MLP_linear(torch.cat([trans_out, depth_global], 1))
        print(f"   obs_hist (transformer):   {obs_hist.shape} (B, 768)")
    
    print("✅ Transformer 模式维度验证通过！")


if __name__ == "__main__":
    test_da3_dataflow_detailed()
    test_transformer_mode()