import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class SemanticGraphMapper:
    def __init__(self, gnm_model, device='cuda', 
                 dist_thresh=0.5, yaw_thresh=0.5, 
                 new_node_sim_thresh=0.85, 
                 loop_closure_sim_thresh=0.95):
        """
        :param gnm_model: 预训练的 GNM/ViNT 模型
        :param dist_thresh: 触发新节点的最短几何距离 (米)
        :param yaw_thresh: 触发新节点的最小转向角 (弧度)
        :param new_node_sim_thresh: 差异大于此阈值且位移满足时，创建新节点
        :param loop_closure_sim_thresh: 相似度大于此极高阈值时，认为是回到了岔路口/旧地点
        """
        self.device = device
        self.gnm_model = gnm_model.to(device)
        self.gnm_model.eval()
        
        # 加载 CLIP 模型用于语义特征
        print("Loading CLIP...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        
        # 核心：使用 networkx 构建有向或无向图来替代原版的 List
        self.topo_graph = nx.Graph() 
        
        # 状态追踪
        self.current_node_id = None  # 机器人当前所在的节点 ID
        self.node_counter = 0        # 用于分配新的节点 ID

        self.dist_thresh = dist_thresh
        self.yaw_thresh = yaw_thresh
        self.new_node_sim_thresh = new_node_sim_thresh
        self.loop_closure_sim_thresh = loop_closure_sim_thresh

    @torch.no_grad()
    def extract_semantic_vector(self, image_pil, image_tensor):
        """提取高维融合向量 (GNM特征 + CLIP语义)"""
        # 1. 提取视觉/导航特征 (视你的 GNM 代码接口而定，如 efficientnet 提取)
        # gnm_feat = self.gnm_model.encoder(image_tensor.to(self.device))
        # gnm_feat = F.normalize(gnm_feat, p=2, dim=-1)

        # 2. 提取语义特征
        clip_inputs = self.clip_processor(images=image_pil, return_tensors="pt").to(self.device)
        clip_feat = self.clip_model.get_image_features(**clip_inputs)
        fused_vector = F.normalize(clip_feat, p=2, dim=-1)

        # 融合向量
        # fused_vector = torch.cat([gnm_feat, clip_feat], dim=-1) # [1, Dim_GNM + Dim_CLIP]
        return fused_vector

    def calculate_distance(self, pose1, pose2):
        """计算两个位姿之间的欧式距离和偏航角差"""
        dx = pose1[0] - pose2[0]
        dy = pose1[1] - pose2[1]
        dyaw = abs(pose1[2] - pose2[2])
        return np.hypot(dx, dy), dyaw

    def process_frame(self, image_pil, image_tensor, current_pose):
        """
        在线遥控时，每一帧都会进入这个函数
        返回值: (bool 是否图发生了改变, str 状态信息)
        """
        # 获取当前画面的高维特征
        curr_vec = self.extract_semantic_vector(image_pil, image_tensor)

        # 1. 初始化（图为空时）
        if self.topo_graph.number_of_nodes() == 0:
            self._add_new_node(curr_vec, current_pose, image_pil)
            return True, "Initialized first node (0)."

        # 2. 岔路口检测 / 闭环检测 (Loop Closure)
        # 将当前特征与图中所有已有节点进行比对
        all_nodes = list(self.topo_graph.nodes(data=True))
        best_sim = -1.0
        best_node_id = None

        for n_id, data in all_nodes:
            sim = F.cosine_similarity(curr_vec, data['feature'], dim=-1).item()
            if sim > best_sim:
                best_sim = sim
                best_node_id = n_id

        # 如果发现极其相似的已知节点（说明退回到了岔路口或曾经走过的地方）
        if best_sim > self.loop_closure_sim_thresh and best_node_id != self.current_node_id:
            # 机器人回到了旧节点！
            old_node_id = self.current_node_id
            self.current_node_id = best_node_id
            
            # 可选：如果物理距离也很近，可以在旧节点和找回的节点间连一条边
            # self.topo_graph.add_edge(old_node_id, best_node_id)
            
            return False, f"Loop closure detected! Relocalized to Node {best_node_id} (Sim: {best_sim:.2f})."

        # 3. 决定是否触发采集新节点
        curr_node_data = self.topo_graph.nodes[self.current_node_id]
        dist, dyaw = self.calculate_distance(current_pose, curr_node_data['pose'])
        sim_to_current = F.cosine_similarity(curr_vec, curr_node_data['feature'], dim=-1).item()

        # 触发条件：距离够远，或者转弯够大，且画面确实发生了一定变化（不过分相似）
        if (dist > self.dist_thresh or dyaw > self.yaw_thresh) and sim_to_current < self.new_node_sim_thresh:
            new_id = self._add_new_node(curr_vec, current_pose, image_pil)
            return True, f"Added Node {new_id}. Connected {self.current_node_id-1} -> {new_id}."
            
        return False, "Skipped. Staying at current node."

    def _add_new_node(self, feature_vec, pose, image_pil):
        """内部方法：向图中添加新节点和边"""
        new_id = self.node_counter
        self.topo_graph.add_node(
            new_id, 
            feature=feature_vec, 
            pose=pose, 
            # image=image_pil # 如果内存充裕可以存下来，但最好存路径或丢弃，因为我们已有高维向量
        )
        
        # 如果不是第一个节点，则与上一个节点建立连边（代表路径可达）
        if self.current_node_id is not None:
            # 这里的权重可以是物理距离，或者后续直接用 GNM 预测的可达时间
            dist, _ = self.calculate_distance(pose, self.topo_graph.nodes[self.current_node_id]['pose'])
            self.topo_graph.add_edge(self.current_node_id, new_id, weight=dist)

        self.current_node_id = new_id
        self.node_counter += 1
        return new_id

    def save_map(self, filepath):
        """建图完成后，保存带有高维向量和拓扑结构的图"""
        # nx.write_gpickle() 在新版 networkx 已弃用，可以使用 pickle 直接保存整个图对象
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.topo_graph, f)
        print(f"Topological Graph saved to {filepath} with {self.topo_graph.number_of_nodes()} nodes.")