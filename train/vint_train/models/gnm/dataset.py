import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional

# Import the InputProcessor from the local DepthAnything3 copy
from .depth_anything_3.utils.io.input_processor import InputProcessor


class VisualNavDataset(Dataset):
    def __init__(
        self, 
        root_dir: str = "./data",
        context_size: int = 5,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        num_workers: int = 4,
    ):
        """
        Visual navigation dataset that uses DepthAnything3's InputProcessor for image preprocessing.
        
        Args:
            root_dir (str): Root directory containing your navigation data (e.g., ROS bag files)
            context_size (int): Number of historical observations to include
            process_res (int): Target resolution for image preprocessing
            process_res_method (str): Preprocessing method ('upper_bound_resize', 'upper_bound_crop', etc.)
            num_workers (int): Number of workers for parallel preprocessing
        """
        self.root_dir = root_dir
        self.context_size = context_size
        self.process_res = process_res
        self.process_res_method = process_res_method
        self.num_workers = num_workers
        
        # Initialize the InputProcessor
        self.input_processor = InputProcessor()
        
        # TODO: Implement proper data loading logic here
        # For ROS bag integration, you would:
        # 1. Scan root_dir for .bag files
        # 2. Extract image sequences, poses, and metadata
        # 3. Create a list of sample indices or paths
        # For now, we'll keep the dummy implementation but with proper preprocessing
        
        # Placeholder: In a real implementation, load your actual data here
        self.samples = []  # List of (current_img_path, goal_img_path, history_img_paths, dist_label, action_label)
        
        # For demonstration, create dummy file paths that mimic real data structure
        # In practice, you'd populate this from your actual dataset
        dummy_data_dir = os.path.join(root_dir, "dummy_samples")
        os.makedirs(dummy_data_dir, exist_ok=True)
        
        # Create a few dummy samples
        for i in range(100):  # Adjust based on your needs
            current_img = f"{dummy_data_dir}/sample_{i}_current.jpg"
            goal_img = f"{dummy_data_dir}/sample_{i}_goal.jpg"
            history_imgs = [f"{dummy_data_dir}/sample_{i}_history_{j}.jpg" for j in range(context_size)]
            
            # Create dummy images if they don't exist (for testing)
            if not os.path.exists(current_img):
                dummy_pil = Image.new('RGB', (640, 480), color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
                dummy_pil.save(current_img)
            if not os.path.exists(goal_img):
                dummy_pil = Image.new('RGB', (640, 480), color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
                dummy_pil.save(goal_img)
            for hist_img in history_imgs:
                if not os.path.exists(hist_img):
                    dummy_pil = Image.new('RGB', (640, 480), color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
                    dummy_pil.save(hist_img)
            
            # Generate dummy labels
            dist_label = np.random.uniform(0.0, 10.0)
            action_label = np.random.randint(0, 20)
            
            self.samples.append((current_img, goal_img, history_imgs, dist_label, action_label))
        
        self.length = len(self.samples)
        
    def __len__(self):
        return self.length
        
    def _load_and_preprocess_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Load and preprocess a list of images using InputProcessor.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Preprocessed tensor of shape (N, 3, H, W) where N = len(image_paths)
        """
        # Use InputProcessor to handle all preprocessing
        batch_tensor, _, _ = self.input_processor(
            image=image_paths,
            process_res=self.process_res,
            process_res_method=self.process_res_method,
            num_workers=self.num_workers,
            sequential=(self.num_workers <= 1)
        )
        
        # InputProcessor returns (1, N, 3, H, W), we need (N, 3, H, W)
        return batch_tensor.squeeze(0)
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (obs_img, goal_img, history_imgs, dist_label, action_label)
        - obs_img: current observation image, shape (3, H, W)
        - goal_img: goal image, shape (3, H, W)  
        - history_imgs: historical observation images, shape (context_size, 3, H, W)
        - dist_label: distance label, scalar
        - action_label: action label, scalar
        """
        current_img_path, goal_img_path, history_img_paths, dist_label, action_label = self.samples[idx]
        
        # Preprocess current observation
        obs_img = self._load_and_preprocess_images([current_img_path])[0]  # Shape: (3, H, W)
        
        # Preprocess goal image
        goal_img = self._load_and_preprocess_images([goal_img_path])[0]  # Shape: (3, H, W)
        
        # Preprocess historical images
        history_imgs = self._load_and_preprocess_images(history_img_paths)  # Shape: (context_size, 3, H, W)
        
        # Convert labels to tensors
        dist_label = torch.tensor(dist_label, dtype=torch.float32)
        action_label = torch.tensor(action_label, dtype=torch.long)
        
        return obs_img, goal_img, history_imgs, dist_label, action_label