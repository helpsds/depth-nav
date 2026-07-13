import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def process_camera_params(zattrs_path, target_width=224, target_height=224):
    # 1. 加载文件
    with open(zattrs_path, 'r') as f:
        data = json.load(f)
        
    # --- 处理内参 (Intrinsics) ---
    # 2. 提取 1D 的 K 并转为 3x3 矩阵
    K_flat = data['camera_info']['K']
    K_matrix = np.array(K_flat).reshape(3, 3)
    
    # 3. 缩放内参 (非常重要！)
    orig_width = data['camera_info']['width']
    orig_height = data['camera_info']['height']
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    
    K_matrix[0, :] *= scale_x  # 缩放第一行 (fx, cx)
    K_matrix[1, :] *= scale_y  # 缩放第二行 (fy, cy)
    
    # --- 处理外参 (Extrinsics) ---
    # 4. 提取旋转与平移
    rot_data = data['transform']['rotation']
    trans_data = data['transform']['translation']
    
    # 注意：scipy 接收的四元数顺序是 [x, y, z, w]
    quat = [rot_data['x'], rot_data['y'], rot_data['z'], rot_data['w']]
    rot_matrix = R.from_quat(quat).as_matrix() # 转为 3x3 旋转矩阵
    trans_vector = np.array([trans_data['x'], trans_data['y'], trans_data['z']])
    
    # 5. 组装 4x4 外参矩阵
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rot_matrix
    extrinsic_matrix[:3, 3] = trans_vector
    
    # --- 转换为 Tensor ---
    # 6. 转为 Tensor 并增加 Batch 维度 (变成 [1, 3, 3] 和 [1, 4, 4])
    intrinsics_tensor = torch.tensor(K_matrix, dtype=torch.float32).unsqueeze(0)
    extrinsics_tensor = torch.tensor(extrinsic_matrix, dtype=torch.float32).unsqueeze(0)
    
    return intrinsics_tensor, extrinsics_tensor

if __name__ == "__main__":
    # 替换为你 .zattrs 文件的实际路径
    file_path = '.zattrs' 
    
    # 假设你模型输入的图片大小是 224x224
    intrinsics, extrinsics = process_camera_params(file_path, target_width=224, target_height=224)
    
    print("✨ 内参 Tensor 形状:", intrinsics.shape)
    print(intrinsics)
    print("\n✨ 外参 Tensor 形状:", extrinsics.shape)
    print(extrinsics)