import argparse
import os
import shutil
import time
import numpy as np
from PIL import Image as PILImage

# ROS相关
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

# 复用原仓库工具（确保utils.py在同目录下）
from utils import msg_to_pil

# 全局配置
IMAGE_TOPIC = "/usb_cam/image_raw"
ODOM_TOPIC = "/odom"  # 可选：用于运动触发采集
TOPOMAP_IMAGES_DIR = "../topomaps/images"

# 全局变量
obs_img = None
last_collect_pose = None
need_collect = False
history_feats = []  # 可选：用于智能去重

# ---------------------- 工具函数（复用原逻辑） ----------------------
def remove_files_in_dir(dir_path: str):
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# ---------------------- ROS回调函数 ----------------------
def callback_obs(msg: Image):
    """实时接收摄像头图像"""
    global obs_img
    obs_img = msg_to_pil(msg)

def callback_odom(msg: Odometry):
    """可选：接收里程计，用于运动触发采集（替代固定时间）"""
    global last_collect_pose, need_collect
    # 解析当前位姿
    curr_x = msg.pose.pose.position.x
    curr_y = msg.pose.pose.position.y
    curr_q = msg.pose.pose.orientation
    curr_yaw = euler_from_quaternion([curr_q.x, curr_q.y, curr_q.z, curr_q.w])[2]
    curr_pose = (curr_x, curr_y, curr_yaw)
    
    if last_collect_pose is None:
        last_collect_pose = curr_pose
        return
    
    # 计算位姿差（位移≥0.5m 或 角度≥15° 触发采集）
    dis = np.linalg.norm([curr_x - last_collect_pose[0], curr_y - last_collect_pose[1]])
    yaw_diff = abs(curr_yaw - last_collect_pose[2])
    yaw_diff = min(yaw_diff, 2*np.pi - yaw_diff)
    
    if dis >= 0.5 or yaw_diff >= np.deg2rad(15):
        need_collect = True

def callback_joy(msg: Joy):
    """手柄控制：按钮0结束建图"""
    if msg.buttons[0]:
        rospy.signal_shutdown("建图完成，手动结束")

# ---------------------- 主逻辑 ----------------------
def main(args: argparse.Namespace):
    global obs_img, last_collect_pose, need_collect, history_feats
    
    # 1. 初始化ROS节点
    rospy.init_node("ONLINE_CREATE_TOPOMAP", anonymous=False)
    rospy.Subscriber(IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    rospy.Subscriber(ODOM_TOPIC, Odometry, callback_odom, queue_size=1)  # 可选：运动触发
    rospy.Subscriber("joy", Joy, callback_joy, queue_size=1)
    
    # 2. 创建拓扑图目录
    topomap_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
    if not os.path.isdir(topomap_dir):
        os.makedirs(topomap_dir)
    else:
        print(f"{topomap_dir} 已存在，清空旧图像...")
        remove_files_in_dir(topomap_dir)
    
    # 3. 初始化采集状态
    i = 0
    rate = rospy.Rate(10)  # 循环频率10Hz
    last_collect_time = time.time()
    print("在线建图已启动！遥控小车走，图会自动生成...")
    print("按手柄按钮0结束建图")
    
    # 4. 实时采集循环
    while not rospy.is_shutdown():
        if obs_img is None:
            rate.sleep()
            continue
        
        # ---------------------- 采集触发逻辑（二选一） ----------------------
        # 选项A：固定时间采集（原仓库逻辑，简单稳定）
        # if time.time() - last_collect_time >= args.dt:
        #     trigger = True
        
        # 选项B：运动触发采集（推荐，更智能，避免冗余）
        trigger = need_collect
        
        # ---------------------- 执行采集 ----------------------
        if trigger:
            # 保存图像
            obs_img.save(os.path.join(topomap_dir, f"{i}.png"))
            print(f"已保存拓扑节点 {i}")
            
            # 更新状态
            i += 1
            last_collect_time = time.time()
            need_collect = False  # 重置运动触发标记
            # 可选：更新里程计位姿
            if last_collect_pose is not None:
                # 这里可以补充保存位姿到元数据的逻辑
                pass
        
        obs_img = None
        rate.sleep()
    
    print(f"建图完成！共生成 {i} 个拓扑节点，保存在 {topomap_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在线实时建图：遥控小车走的同时生成拓扑图")
    parser.add_argument("--dir", "-d", default="online_topomap", type=str, help="拓扑图保存目录名")
    parser.add_argument("--dt", "-t", default=1.0, type=float, help="固定时间采集间隔（仅选项A用）")
    args = parser.parse_args()
    main(args)