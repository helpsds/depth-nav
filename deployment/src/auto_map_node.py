#!/usr/bin/env python
import rospy
import torch
import cv2
import numpy as np
import pickle
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from PIL import Image as PILImage

import os
import argparse
import yaml

from utils import load_model, transform_images # 从仓库的 utils 导入关键函数

# ... 其他原有的导入 ...

# 导入我们的核心算法
from semantic_graph_mapper import SemanticGraphMapper

# 导入 GNM 仓库里的模型加载和图像预处理方法 (注意根据你的实际路径调整)
# 从 deployment 文件夹的相关工具库中导入
# 例如: from utils import load_model, transform_images 

class AutoMappingNode:
    # def __init__(self):
    #     rospy.init_node('auto_topological_mapper', anonymous=True)
    #     self.bridge = CvBridge()
        
    #     # 1. 加载 GNM/ViNT 模型
    #     rospy.loginfo("Loading Navigation Model...")
    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #     # 【注意】这里需要替换为你实际加载模型的代码
    #     # self.gnm_model = load_model("vint", "path/to/weights.pth").to(self.device)
    #     self.gnm_model = None # 占位符，请替换
        
    #     # 2. 初始化智能建图器
    #     self.mapper = SemanticGraphMapper(
    #         gnm_model=self.gnm_model, 
    #         device=self.device,
    #         dist_thresh=0.5,        # 每走0.5米考虑存一张
    #         yaw_thresh=0.5,         # 或每转约30度考虑存一张
    #         new_node_sim_thresh=0.85, 
    #         loop_closure_sim_thresh=0.95
    #     )
        
    #     # 3. 状态变量
    #     self.current_pose = None
    #     self.latest_image_msg = None
        
    #     # 4. 订阅 ROS 话题 (请确保话题名与你的真实机器人一致)
    #     rospy.Subscriber("/odom", Odometry, self.odom_callback)
    #     rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        
    #     # 5. 注册关闭事件（按 Ctrl+C 时保存地图）
    #     rospy.on_shutdown(self.save_map_on_exit)
    #     rospy.loginfo("Auto Mapper is ready! Start teleoperating your robot.")

    def __init__(self, args):
        rospy.init_node('auto_topological_mapper', anonymous=True)
        # ... cvBridge 等初始化 ...

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Using device: {self.device}")
        
        # ==========================================
        # 完美移植原仓库的模型加载逻辑
        # ==========================================
        MODEL_CONFIG_PATH = "../config/models.yaml"
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)

        # 获取配置路径和权重路径
        model_config_path = model_paths[args.model]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        ckpth_path = model_paths[args.model]["ckpt_path"]
        if not os.path.exists(ckpth_path):
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
            
        rospy.loginfo(f"Loading {args.model} model from {ckpth_path}...")
        
        # 使用 utils.py 中的 load_model 真正把模型加载到显存里
        self.gnm_model = load_model(
            ckpth_path,
            self.model_params,
            self.device,
        )
        self.gnm_model = self.gnm_model.to(self.device)
        self.gnm_model.eval() # 开启评估模式
        # ==========================================

        # 初始化你的建图算法，把加载好的模型传进去
        self.mapper = SemanticGraphMapper(
            gnm_model=self.gnm_model, 
            device=self.device,
            # ... 阈值设置保持不变 ...
        )
        
        # ... 后续的订阅 Topic 和变量初始化保持不变 ...
    def odom_callback(self, msg):
        """处理里程计数据，提取 x, y, yaw"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # 将四元数转换为欧拉角以获取偏航角(yaw)
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        
        self.current_pose = (x, y, yaw)

    def image_callback(self, msg):
        """处理图像数据并实时建图"""
        if self.current_pose is None:
            return # 等待里程计数据就绪

        # 将 ROS Image 转换为 OpenCV 格式，再转为 PIL 和 Tensor
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        image_pil = PILImage.fromarray(cv_image)
        
        # [NEW] 使用仓库自带的 transform 处理图像，适配模型输入
        # model_params["image_size"] 通常在 yaml 里配好了，比如 [85, 112]
        image_tensor = transform_images(image_pil, self.model_params["image_size"], center_crop=False)
        image_tensor = image_tensor.to(self.device)

        # 喂给建图算法
        is_added, status_msg = self.mapper.process_frame(image_pil, image_tensor, self.current_pose)
        
        if is_added:
            rospy.loginfo(status_msg)

    def save_map_on_exit(self):
        """遥控结束，保存地图"""
        save_path = "topological_map.pkl"
        rospy.loginfo(f"Stopping teleoperation. Saving map to {save_path}...")
        self.mapper.save_map(save_path)
        rospy.loginfo("Map saved successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Auto Topological Mapping Node")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (hint: check ../config/models.yaml) (default: nomad)",
    )
    args = parser.parse_args()
    
    try:
        node = AutoMappingNode(args)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass