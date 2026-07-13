import numpy as np
from PIL import Image
import torch

VIZ_IMAGE_SIZE = (640, 480)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])

# ImageNet normalization parameters
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize_image(arr: np.ndarray) -> np.ndarray:
    """
    Denormalize an image that was normalized with ImageNet mean and std.
    Assumes input is in format (C, H, W) with values in approximately [-2, 2] range.
    Returns array in [0, 1] range.
    """
    # Transpose to (H, W, C) for easier processing
    if arr.shape[0] == 3:  # (C, H, W) format
        arr_hwc = np.transpose(arr, (1, 2, 0))
    else:
        arr_hwc = arr  # Assume already (H, W, C)
    
    # Apply denormalization: x = x * std + mean
    denorm_arr = arr_hwc * IMAGENET_STD + IMAGENET_MEAN
    
    # Clip to [0, 1] range to handle any numerical errors
    denorm_arr = np.clip(denorm_arr, 0, 1)
    
    # Transpose back to (C, H, W) if needed
    if arr.shape[0] == 3:
        return np.transpose(denorm_arr, (2, 0, 1))
    else:
        return denorm_arr


def numpy_to_img(arr: np.ndarray) -> Image:
    """
    Convert numpy array to PIL Image for visualization.
    Handles both normalized and unnormalized images by detecting the value range.
    """
    # Make a copy to avoid modifying the original
    arr_copy = arr.copy()
    
    # Check if the image appears to be normalized (values outside [0, 1] range)
    # This is a heuristic - if min < 0 or max > 1.5, assume it's normalized
    if arr_copy.min() < 0 or arr_copy.max() > 1.5:
        # Apply denormalization for ImageNet normalized images
        arr_copy = denormalize_image(arr_copy)
    
    # Ensure values are in [0, 1] range
    arr_copy = np.clip(arr_copy, 0, 1)
    
    img = Image.fromarray(np.transpose(np.uint8(255 * arr_copy), (1, 2, 0)))
    img = img.resize(VIZ_IMAGE_SIZE)
    return img


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()