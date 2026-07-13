#!/usr/bin/env python3
"""
Test script to verify GNM data flow with DINOv2 backbone using random input images.
This script tests the complete pipeline: 7 images input -> DINOv2 features -> depth estimation -> fusion -> output
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add the project path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'train'))

from vint_train.models.gnm.gnm import GNM


def create_test_input(batch_size=1, img_height=224, img_width=224):
    """
    Create test input with random images.
    Returns:
        obs_img: (B, (1+context_size)*3, H, W) - current observation + history
        goal_img: (B, 3, H, W) - goal image
    """
    context_size = 5
    
    # Current observation image (3 channels)
    current_obs = torch.randn(batch_size, 3, img_height, img_width)
    
    # History images (5 images * 3 channels = 15 channels)
    history_imgs = torch.randn(batch_size, context_size * 3, img_height, img_width)
    
    # Combine current observation and history for obs_img input
    obs_img = torch.cat([current_obs, history_imgs], dim=1)  # (B, 18, H, W) = (B, (1+5)*3, H, W)
    
    # Goal image (3 channels)
    goal_img = torch.randn(batch_size, 3, img_height, img_width)
    
    return obs_img, goal_img


def print_tensor_info(name, tensor, indent=0):
    """Helper function to print tensor information with proper formatting"""
    spaces = "  " * indent
    print(f"{spaces}{name}: shape={tensor.shape}, dtype={tensor.dtype}")
    if tensor.numel() <= 16:  # Only print values for small tensors
        print(f"{spaces}  Values: {tensor.flatten()[:min(8, tensor.numel())].tolist()}")


def test_gnm_with_dinov2(debug_mode=True):
    """Test GNM model with DINOv2 backbone and depth estimation."""
    print("=== Testing GNM Data Flow with DINOv2 Backbone ===")
    
    # Create model with DINOv2 backbone and depth estimation
    model = GNM(
        context_size=5,
        len_traj_pred=5,
        learn_angle=True,
        obs_encoding_size=1024,
        goal_encoding_size=1024,
        backbone="dinov2",
        dinov2_model_size="vitl",  # Using ViT-Large (1024 dim)
        use_depth=True,
        dpt_model="dpt_hybrid"
    )
    
    print(f"Model created successfully!")
    print(f"Backbone: {model.backbone}")
    if hasattr(model, 'dinov2_model_size'):
        print(f"DINOv2 model size: {model.dinov2_model_size}")
    print(f"Use depth estimation: {model.use_depth}")
    
    # Create test inputs
    obs_img, goal_img = create_test_input(batch_size=1, img_height=224, img_width=224)
    
    print(f"\nInput shapes:")
    print(f"obs_img: {obs_img.shape}  # (B, (1+5)*3, H, W) = (B, 18, H, W)")
    print(f"goal_img: {goal_img.shape}  # (B, 3, H, W)")
    
    # Set model to evaluation mode
    model.eval()
    
    try:
        with torch.no_grad():
            if debug_mode:
                print(f"\n{'='*60}")
                print(f"DETAILED DEBUG OUTPUT - DINOv2 WITH DEPTH")
                print(f"{'='*60}")
                
                # Print input details
                print(f"\n1. INPUT STAGE:")
                print_tensor_info("obs_img", obs_img, 1)
                print_tensor_info("goal_img", goal_img, 1)
                
                # Extract current observation for depth estimation
                current_obs = obs_img[:, :3, :, :]
                print_tensor_info("current_obs (for depth)", current_obs, 1)
                
                # Get visual features from encoder
                obs_encoding_raw = model.obs_encoder(obs_img)
                print(f"\n2. OBSERVATION ENCODING STAGE:")
                print_tensor_info("obs_encoder output (raw)", obs_encoding_raw, 1)
                obs_encoding = model.compress_observation(obs_encoding_raw)
                print_tensor_info("obs_encoding (compressed)", obs_encoding, 1)
                
                # Goal encoding
                goal_encoding_raw = model.goal_encoder(goal_img)
                print(f"\n3. GOAL ENCODING STAGE:")
                print_tensor_info("goal_encoder output (raw)", goal_encoding_raw, 1)
                goal_encoding = model.compress_goal(goal_encoding_raw)
                print_tensor_info("goal_encoding (compressed)", goal_encoding, 1)
                
                # Depth estimation
                depth_features_raw = model.depth_estimator(current_obs)['out']
                print(f"\n4. DEPTH ESTIMATION STAGE:")
                print_tensor_info("depth_estimator output (raw)", depth_features_raw, 1)
                depth_features = depth_features_raw.squeeze(1)
                depth_features = depth_features.mean(dim=[1, 2])
                print_tensor_info("depth_features (averaged)", depth_features, 1)
                
                # Feature fusion
                combined_features = torch.cat([obs_encoding, depth_features.unsqueeze(1)], dim=1)
                print(f"\n5. FEATURE FUSION STAGE:")
                print_tensor_info("combined_features (obs + depth)", combined_features, 1)
                projected_features = model.mlp_projector(combined_features)
                print_tensor_info("projected_features (MLP output)", projected_features, 1)
                
                # Final combination and prediction
                z = torch.cat([goal_encoding, projected_features], dim=1)
                print(f"\n6. FINAL COMBINATION STAGE:")
                print_tensor_info("z (goal + projected)", z, 1)
                z_processed = model.linear_layers(z)
                print_tensor_info("z_processed (linear layers)", z_processed, 1)
                
                # Outputs
                dist_pred = model.dist_predictor(z_processed)
                action_pred_raw = model.action_predictor(z_processed)
                action_pred = action_pred_raw.reshape((action_pred_raw.shape[0], model.len_trajectory_pred, model.num_action_params))
                action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
                if model.health_angle:
                    action_pred[:, :, 2:] = F.normalize(action_pred[:, :, 2:].clone(), dim=-1)
                
                print(f"\n7. FINAL OUTPUT STAGE:")
                print_tensor_info("dist_pred", dist_pred, 1)
                print_tensor_info("action_pred", action_pred, 1)
                
                print(f"\n✅ Detailed debug output completed!")
            
            # Standard forward pass
            dist_pred, action_pred = model(obs_img, goal_img)
            
            print(f"\nOutput shapes:")
            print(f"Distance prediction: {dist_pred.shape}  # (B, 1)")
            print(f"Action prediction: {action_pred.shape}  # (B, len_traj_pred, num_action_params)")
            
            print(f"\nSample outputs:")
            print(f"Distance: {dist_pred[0].item():.4f}")
            print(f"First action waypoint: [{action_pred[0, 0, 0].item():.4f}, {action_pred[0, 0, 1].item():.4f}]")
            if action_pred.shape[-1] > 2:
                print(f"First action angle: [{action_pred[0, 0, 2].item():.4f}, {action_pred[0, 0, 3].item():.4f}]")
            
            print(f"\n✅ Data flow test PASSED!")
            return True
            
    except Exception as e:
        print(f"\n❌ Data flow test FAILED with error: {e}")
        return False


def test_gnm_without_depth(debug_mode=True):
    """Test GNM model with DINOv2 backbone without depth estimation."""
    print("\n=== Testing GNM Data Flow WITHOUT Depth Estimation ===")
    
    # Create model with DINOv2 backbone but without depth estimation
    model = GNM(
        context_size=5,
        len_traj_pred=5,
        learn_angle=True,
        obs_encoding_size=1024,
        goal_encoding_size=1024,
        backbone="dinov2",
        dinov2_model_size="vitl",
        use_depth=False  # Disable depth estimation
    )
    
    print(f"Model created successfully (without depth estimation)!")
    print(f"Backbone: {model.backbone}")
    print(f"Use depth estimation: {model.use_depth}")
    
    # Create test inputs
    obs_img, goal_img = create_test_input(batch_size=1, img_height=224, img_width=224)
    
    print(f"Input shapes:")
    print(f"obs_img: {obs_img.shape}")
    print(f"goal_img: {goal_img.shape}")
    
    # Set model to evaluation mode
    model.eval()
    
    try:
        with torch.no_grad():
            if debug_mode:
                print(f"\n{'='*60}")
                print(f"DETAILED DEBUG OUTPUT - DINOv2 WITHOUT DEPTH")
                print(f"{'='*60}")
                
                # Print input details
                print(f"\n1. INPUT STAGE:")
                print_tensor_info("obs_img", obs_img, 1)
                print_tensor_info("goal_img", goal_img, 1)
                
                # Get visual features from encoder
                obs_encoding_raw = model.obs_encoder(obs_img)
                print(f"\n2. OBSERVATION ENCODING STAGE:")
                print_tensor_info("obs_encoder output (raw)", obs_encoding_raw, 1)
                obs_encoding = model.compress_observation(obs_encoding_raw)
                print_tensor_info("obs_encoding (compressed)", obs_encoding, 1)
                
                # Goal encoding
                goal_encoding_raw = model.goal_encoder(goal_img)
                print(f"\n3. GOAL ENCODING STAGE:")
                print_tensor_info("goal_encoder output (raw)", goal_encoding_raw, 1)
                goal_encoding = model.compress_goal(goal_encoding_raw)
                print_tensor_info("goal_encoding (compressed)", goal_encoding, 1)
                
                # Feature projection (no depth)
                projected_features = model.mlp_projector(obs_encoding)
                print(f"\n4. FEATURE PROJECTION STAGE:")
                print_tensor_info("projected_features (MLP output)", projected_features, 1)
                
                # Final combination and prediction
                z = torch.cat([goal_encoding, projected_features], dim=1)
                print(f"\n5. FINAL COMBINATION STAGE:")
                print_tensor_info("z (goal + projected)", z, 1)
                z_processed = model.linear_layers(z)
                print_tensor_info("z_processed (linear layers)", z_processed, 1)
                
                # Outputs
                dist_pred = model.dist_predictor(z_processed)
                action_pred_raw = model.action_predictor(z_processed)
                action_pred = action_pred_raw.reshape((action_pred_raw.shape[0], model.len_trajectory_pred, model.num_action_params))
                action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
                if model.health_angle:
                    action_pred[:, :, 2:] = F.normalize(action_pred[:, :, 2:].clone(), dim=-1)
                
                print(f"\n6. FINAL OUTPUT STAGE:")
                print_tensor_info("dist_pred", dist_pred, 1)
                print_tensor_info("action_pred", action_pred, 1)
                
                print(f"\n✅ Detailed debug output completed!")
            
            # Standard forward pass
            dist_pred, action_pred = model(obs_img, goal_img)
            
            print(f"Output shapes:")
            print(f"Distance prediction: {dist_pred.shape}")
            print(f"Action prediction: {action_pred.shape}")
            
            print(f"✅ Data flow test without depth PASSED!")
            return True
            
    except Exception as e:
        print(f"❌ Data flow test without depth FAILED with error: {e}")
        return False


def test_original_mobilenet(debug_mode=True):
    """Test original MobileNet backbone for comparison."""
    print("\n=== Testing Original MobileNet Backbone ===")
    
    # Create model with original MobileNet backbone
    model = GNM(
        context_size=5,
        len_traj_pred=5,
        learn_angle=True,
        obs_encoding_size=1024,
        goal_encoding_size=1024,
        backbone="mobilenet"  # Use original MobileNet
    )
    
    print(f"Model created successfully (MobileNet backbone)!")
    print(f"Backbone: {model.backbone}")
    
    # Create test inputs
    obs_img, goal_img = create_test_input(batch_size=1, img_height=224, img_width=224)
    
    print(f"Input shapes:")
    print(f"obs_img: {obs_img.shape}")
    print(f"goal_img: {goal_img.shape}")
    
    # Set model to evaluation mode
    model.eval()
    
    try:
        with torch.no_grad():
            if debug_mode:
                print(f"\n{'='*60}")
                print(f"DETAILED DEBUG OUTPUT - MOBILENET BACKBONE")
                print(f"{'='*60}")
                
                # Print input details
                print(f"\n1. INPUT STAGE:")
                print_tensor_info("obs_img", obs_img, 1)
                print_tensor_info("goal_img", goal_img, 1)
                
                # Get visual features from encoder
                obs_encoding_raw = model.obs_encoder(obs_img)
                print(f"\n2. OBSERVATION ENCODING STAGE:")
                print_tensor_info("obs_encoder output (raw)", obs_encoding_raw, 1)
                obs_encoding_flattened = model.flatten(obs_encoding_raw)
                print_tensor_info("obs_encoding (flattened)", obs_encoding_flattened, 1)
                obs_encoding = model.compress_observation(obs_encoding_flattened)
                print_tensor_info("obs_encoding (compressed)", obs_encoding, 1)
                
                # Goal encoding (with concatenated input)
                obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
                print_tensor_info("obs_goal_input (concatenated)", obs_goal_input, 1)
                goal_encoding_raw = model.goal_encoder(obs_goal_input)
                print(f"\n3. GOAL ENCODING STAGE:")
                print_tensor_info("goal_encoder output (raw)", goal_encoding_raw, 1)
                goal_encoding_flattened = model.flatten(goal_encoding_raw)
                print_tensor_info("goal_encoding (flattened)", goal_encoding_flattened, 1)
                goal_encoding = model.compress_goal(goal_encoding_flattened)
                print_tensor_info("goal_encoding (compressed)", goal_encoding, 1)
                
                # Feature projection
                projected_features = model.mlp_projector(obs_encoding)
                print(f"\n4. FEATURE PROJECTION STAGE:")
                print_tensor_info("projected_features (MLP output)", projected_features, 1)
                
                # Final combination and prediction
                z = torch.cat([goal_encoding, projected_features], dim=1)
                print(f"\n5. FINAL COMBINATION STAGE:")
                print_tensor_info("z (goal + projected)", z, 1)
                z_processed = model.linear_layers(z)
                print_tensor_info("z_processed (linear layers)", z_processed, 1)
                
                # Outputs
                dist_pred = model.dist_predictor(z_processed)
                action_pred_raw = model.action_predictor(z_processed)
                action_pred = action_pred_raw.reshape((action_pred_raw.shape[0], model.len_trajectory_pred, model.num_action_params))
                action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
                if model.health_angle:
                    action_pred[:, :, 2:] = F.normalize(action_pred[:, :, 2:].clone(), dim=-1)
                
                print(f"\n6. FINAL OUTPUT STAGE:")
                print_tensor_info("dist_pred", dist_pred, 1)
                print_tensor_info("action_pred", action_pred, 1)
                
                print(f"\n✅ Detailed debug output completed!")
            
            # Standard forward pass
            dist_pred, action_pred = model(obs_img, goal_img)
            
            print(f"Output shapes:")
            print(f"Distance prediction: {dist_pred.shape}")
            print(f"Action prediction: {action_pred.shape}")
            
            print(f"✅ MobileNet data flow test PASSED!")
            return True
            
    except Exception as e:
        print(f"❌ MobileNet data flow test FAILED with error: {e}")
        return False


def main():
    """Main function with option to run specific tests"""
    print("GNM Data Flow Testing Script")
    print("=" * 50)
    
    # Test all configurations
    success_count = 0
    total_tests = 3
    
    print("Running DINOv2 with depth estimation (detailed debug output)...")
    if test_gnm_with_dinov2(debug_mode=True):
        success_count += 1
        
    print("\n" + "="*80 + "\n")
    
    print("Running DINOv2 without depth estimation (detailed debug output)...")
    if test_gnm_without_depth(debug_mode=True):
        success_count += 1
        
    print("\n" + "="*80 + "\n")
    
    print("Running original MobileNet backbone (detailed debug output)...")
    if test_original_mobilenet(debug_mode=True):
        success_count += 1
    
    print(f"\n{'='*50}")
    print(f"Test Summary: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Data flow is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main()