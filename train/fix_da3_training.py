#!/usr/bin/env python3
"""
DA3训练修复脚本 - 解决显存溢出和action_loss收敛问题
"""

import torch
import torch.nn as nn
import os
import yaml
from pathlib import Path

def fix_da3_config():
    """修复DA3配置文件中的问题"""
    config_path = "/home/yyz/visualnav-transformer/train/config/da3.yaml"
    
    # 读取现有配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("🔧 修复DA3配置...")
    
    # 1. 调整损失权重 - 给action_loss更多重视
    config['alpha'] = 0.3  # 从0.8改为0.3，更平衡
    print(f"   - alpha: {config.get('alpha', 'not set')} -> 0.3")
    
    # 2. 减小测试batch size以避免显存溢出
    if 'eval_batch_size' in config:
        config['eval_batch_size'] = min(config['eval_batch_size'], 64)  # 限制测试batch size
    else:
        config['eval_batch_size'] = 64
    print(f"   - eval_batch_size: {config['eval_batch_size']}")
    
    # 3. 启用梯度裁剪
    config['clipping'] = True
    config['max_norm'] = 1.0
    print(f"   - clipping: enabled (max_norm=1.0)")
    
    # 4. 调整学习率
    config['lr'] = 2e-4  # 稍高的学习率帮助action_loss收敛
    print(f"   - lr: {config.get('lr', 'not set')} -> 2e-4")
    
    # 保存修复后的配置
    backup_path = config_path + ".backup"
    os.rename(config_path, backup_path)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ 配置已修复并保存到 {config_path}")
    print(f"   备份文件: {backup_path}")

def create_memory_efficient_eval_config():
    """创建内存高效的评估配置"""
    eval_config = {
        "project_name": "da3_nav_eval_efficient",
        "run_name": "da3_efficient_eval",
        "use_wandb": False,  # 评估时禁用wandb减少内存占用
        "model_type": "da3",
        "batch_size": 32,  # 训练batch size
        "eval_batch_size": 16,  # 评估batch size - 更小以节省显存
        "num_workers": 4,  # 减少数据加载线程
        "gpu_ids": [7],
        "context_size": 5,
        "len_traj_pred": 5,
        "learn_angle": True,
        "mha_num_attention_heads": 4,
        "mha_num_attention_layers": 4,
        "mha_ff_dim_factor": 4,
        "normalize": True,
        "alpha": 0.3,
        "use_da3_preprocessing": True,
        "da3_process_res": 224,
        "da3_process_res_method": "upper_bound_resize",
        "da3_num_workers": 1,
        "distance": {"min_dist_cat": 0, "max_dist_cat": 20},
        "action": {"min_dist_cat": 0, "max_dist_cat": 10},
        "datasets": {
            "go_stanford": {
                "data_folder": "/home/abc/Datasets/GNM-Dataset/process_data/go_stanford",
                "train": "/home/abc/Datasets/GNM-Dataset/process_data/data_split/go_stanford_data/train",
                "test": "/home/abc/Datasets/GNM-Dataset/process_data/data_split/go_stanford_data/test",
                "end_slack": 0,
                "goals_per_obs": 2,
                "negative_mining": True
            }
        }
    }
    
    eval_config_path = "/home/yyz/visualnav-transformer/train/config/da3_eval_efficient.yaml"
    with open(eval_config_path, 'w') as f:
        yaml.dump(eval_config, f, default_flow_style=False)
    
    print(f"✅ 内存高效评估配置已创建: {eval_config_path}")

def add_gradient_monitoring_to_model():
    """向DA3模型添加梯度监控"""
    model_path = "/home/yyz/visualnav-transformer/train/vint_train/models/gnm/depth_anything_3/model/da3.py"
    
    # 读取现有模型代码
    with open(model_path, 'r') as f:
        lines = f.readlines()
    
    # 查找forward方法的开始位置
    forward_start = -1
    for i, line in enumerate(lines):
        if 'def forward(' in line:
            forward_start = i
            break
    
    if forward_start == -1:
        print("⚠️  未找到forward方法，跳过梯度监控添加")
        return
    
    # 在forward方法开始处添加梯度检查
    gradient_check_code = [
        "        # Gradient and NaN monitoring\n",
        "        if torch.isnan(obs_img).any() or torch.isnan(goal_img).any():\n",
        "            print('⚠️  Warning: NaN detected in input images')\n",
        "            obs_img = torch.nan_to_num(obs_img, nan=0.0)\n",
        "            goal_img = torch.nan_to_num(goal_img, nan=0.0)\n",
        "        \n"
    ]
    
    # 在forward方法结束前添加输出检查
    output_check_code = [
        "        # Check for NaN in outputs\n",
        "        if torch.isnan(output.distance).any():\n",
        "            print('⚠️  Warning: NaN in distance prediction')\n",
        "            output.distance = torch.nan_to_num(output.distance, nan=0.0)\n",
        "        if torch.isnan(output.action).any():\n",
        "            print('⚠️  Warning: NaN in action prediction')\n",
        "            output.action = torch.nan_to_num(output.action, nan=0.0)\n",
        "        \n"
    ]
    
    # 插入梯度检查代码
    lines.insert(forward_start + 1, ''.join(gradient_check_code))
    
    # 查找return语句
    return_line = -1
    for i in range(forward_start, len(lines)):
        if 'return output' in lines[i]:
            return_line = i
            break
    
    if return_line != -1:
        lines.insert(return_line, ''.join(output_check_code))
    
    # 创建备份
    backup_path = model_path + ".backup"
    os.rename(model_path, backup_path)
    
    # 写入修改后的代码
    with open(model_path, 'w') as f:
        f.writelines(lines)
    
    print(f"✅ 梯度监控已添加到模型: {model_path}")
    print(f"   备份文件: {backup_path}")

def main():
    """主修复函数"""
    print("🚀 开始修复DA3训练问题...")
    print("=" * 50)
    
    # 1. 修复配置文件
    fix_da3_config()
    print()
    
    # 2. 创建内存高效评估配置
    create_memory_efficient_eval_config()
    print()
    
    # 3. 添加梯度监控
    add_gradient_monitoring_to_model()
    print()
    
    print("✅ 所有修复已完成！")
    print("\n📋 使用建议:")
    print("1. 使用修复后的配置重新训练: python train.py --config config/da3.yaml")
    print("2. 使用高效评估配置进行测试: python evaluate.py --config config/da3_eval_efficient.yaml")
    print("3. 监控训练日志中的NaN警告信息")
    print("4. 如果仍有显存问题，可进一步减小eval_batch_size到8或4")

if __name__ == "__main__":
    main()