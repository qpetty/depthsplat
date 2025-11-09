"""
Run encoder using direct config loading (no Hydra) and export to PLY.

Image resolution: 512x960
3 input views: left 90°, head-on, right 90°
"""

# ============================================================================
# Configuration - Modify these paths as needed
# ============================================================================

CHECKPOINT_PATH = "pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"  # Set to None for random init
CONFIG_ROOT = "config"  # Path to config directory
OUTPUT_DIR = "run-output"

# Input image paths (set to None to use random images)
IMAGE_LEFT_PATH = "3_input/underwater-left.png"   # View 0: 90° left
IMAGE_CENTER_PATH = "3_input/Underwater.png"  # View 1: Head-on
IMAGE_RIGHT_PATH = "3_input/underwater-right.png"  # View 2: 90° right

# Encoder config overrides (set to None to use YAML defaults)
ENCODER_OVERRIDES = {
    "num_scales": 2,
    "upsample_factor": 4,
    "lowest_feature_resolution": 8,
    "monodepth_vit_type": "vitb",
    "gaussian_adapter": {
        "gaussian_scale_max": 0.1
    }
}

# ============================================================================

import torch
from pathlib import Path
from omegaconf import OmegaConf
from src.config import load_typed_config
from src.model.encoder import EncoderDepthSplatCfg, get_encoder
from src.model.ply_export import export_ply
from src.misc.image_io import load_image
from einops import rearrange
import numpy as np
import math
import torchvision.transforms as tf


def create_rotation_matrix_y(angle_degrees: float) -> torch.Tensor:
    """
    Create a rotation matrix around Y-axis.

    Args:
        angle_degrees: Rotation angle in degrees (positive = counterclockwise when viewed from above)

    Returns:
        3x3 rotation matrix
    """
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Rotation around Y-axis (in OpenCV convention, Y points down)
    # This rotates the camera frame
    rotation = torch.tensor(
        [[cos_a, 0, sin_a],
         [0, 1, 0],
         [-sin_a, 0, cos_a]]
    , dtype=torch.float32)

    return rotation


def create_camera_pose(rotation: torch.Tensor, translation: torch.Tensor = None) -> torch.Tensor:
    """
    Create a 4x4 camera-to-world (C2W) transformation matrix.

    Args:
        rotation: 3x3 rotation matrix
        translation: 3D translation vector (default: [0, 0, 0])

    Returns:
        4x4 C2W matrix
    """
    if translation is None:
        translation = torch.zeros(3, dtype=torch.float32)

    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def load_and_resize_image(image_path: str, target_size: tuple[int, int]) -> torch.Tensor:
    """
    Load an image from disk and resize it to target size.

    Args:
        image_path: Path to image file
        target_size: (height, width) target size

    Returns:
        Float tensor [3, height, width] in range [0, 1]
    """
    if image_path is None or not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    image = load_image(image_path)  # [3, H, W] in range [0, 1]

    # Resize to target size
    target_height, target_width = target_size
    resize_transform = tf.Resize((target_height, target_width), antialias=True)
    image = resize_transform(image)

    return image


def load_encoder_config(config_root: str, overrides: dict = None) -> EncoderDepthSplatCfg:
    """
    Load encoder config directly from YAML file without Hydra.

    Args:
        config_root: Path to config directory
        overrides: Dictionary of config values to override

    Returns:
        EncoderDepthSplatCfg instance
    """
    config_root_path = Path(config_root)
    encoder_cfg_path = config_root_path / "model/encoder/depthsplat.yaml"

    if not encoder_cfg_path.exists():
        raise FileNotFoundError(
            f"Encoder config not found at {encoder_cfg_path}. "
            f"Please check CONFIG_ROOT path: {config_root}"
        )

    print(f"Loading encoder config from: {encoder_cfg_path}")
    encoder_cfg_dict = OmegaConf.load(encoder_cfg_path)

    # Apply overrides if provided
    if overrides:
        print("Applying config overrides:")
        for key, value in overrides.items():
            if isinstance(value, dict) and key in encoder_cfg_dict:
                # Merge nested dicts
                for nested_key, nested_value in value.items():
                    print(f"  {key}.{nested_key}: {nested_value}")
                    encoder_cfg_dict[key][nested_key] = nested_value
            else:
                print(f"  {key}: {value}")
                encoder_cfg_dict[key] = value

    # Convert to typed config
    encoder_cfg = load_typed_config(encoder_cfg_dict, EncoderDepthSplatCfg)
    
    print(f"Encoder config loaded successfully!")
    print(f"  name: {encoder_cfg.name}")
    print(f"  num_scales: {encoder_cfg.num_scales}")
    print(f"  monodepth_vit_type: {encoder_cfg.monodepth_vit_type}")
    print(f"  gaussian_scale_max: {encoder_cfg.gaussian_adapter.gaussian_scale_max}")
    
    return encoder_cfg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load encoder config
    print("\n" + "="*70)
    print("Loading Encoder Config")
    print("="*70)
    encoder_cfg = load_encoder_config(CONFIG_ROOT, ENCODER_OVERRIDES)

    # Initialize encoder
    print("\n" + "="*70)
    print("Initializing Encoder")
    print("="*70)
    encoder, encoder_visualizer = get_encoder(encoder_cfg)
    encoder = encoder.to(device)
    encoder.eval()
    print("Encoder initialized successfully!")

    # Load checkpoint if provided
    if CHECKPOINT_PATH and Path(CHECKPOINT_PATH).exists():
        print(f"\nLoading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            encoder_state_dict = {
                k.replace('encoder.', ''): v
                for k, v in state_dict.items()
                if k.startswith('encoder.')
            }
            encoder.load_state_dict(encoder_state_dict, strict=False)
        else:
            encoder.load_state_dict(checkpoint, strict=False)
        print("Checkpoint loaded successfully!")
    elif CHECKPOINT_PATH:
        print(f"\nWarning: Checkpoint path '{CHECKPOINT_PATH}' does not exist. Using randomly initialized weights.")
    else:
        print("\nNo checkpoint path provided. Using randomly initialized weights.")

    # Create input with resolution 512x960
    batch_size = 1
    num_views = 3  # Left, center, right
    height, width = 512, 960

    # Load images from paths or generate random ones
    print("\n" + "="*70)
    print("Loading Images")
    print("="*70)
    if IMAGE_LEFT_PATH and IMAGE_CENTER_PATH and IMAGE_RIGHT_PATH:
        try:
            image_left = load_and_resize_image(IMAGE_LEFT_PATH, (height, width))
            image_center = load_and_resize_image(IMAGE_CENTER_PATH, (height, width))
            image_right = load_and_resize_image(IMAGE_RIGHT_PATH, (height, width))

            # Stack images: [3, height, width] -> [1, 3, 3, height, width]
            images = torch.stack([image_left, image_center, image_right], dim=0).unsqueeze(0)
            print(f"  Loaded images from:")
            print(f"    Left: {IMAGE_LEFT_PATH}")
            print(f"    Center: {IMAGE_CENTER_PATH}")
            print(f"    Right: {IMAGE_RIGHT_PATH}")
            print(f"  Image shape: {images.shape}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            print("  Falling back to random images.")
            images = torch.rand(batch_size, num_views, 3, height, width)
    else:
        print("  Using random images (image paths not set or incomplete).")
        images = torch.rand(batch_size, num_views, 3, height, width)

    # Create camera poses for 3 views:
    # View 0: 90 degrees to the left
    # View 1: Head-on (center)
    # View 2: 90 degrees to the right
    print("\n" + "="*70)
    print("Setting Up Camera Poses")
    print("="*70)

    camera_distance = 0.1  # Distance from origin (0 = at origin)

    # View 0: Left 90° (camera looks down -X axis)
    rotation_left = create_rotation_matrix_y(90.0)
    translation_left = torch.tensor([camera_distance, 0.0, 0.0], dtype=torch.float32)
    pose_left = create_camera_pose(rotation_left, translation_left)

    # View 1: Head-on (camera looks down +Z axis) - identity rotation
    rotation_center = torch.eye(3, dtype=torch.float32)
    translation_center = torch.tensor([0.0, 0.0, camera_distance], dtype=torch.float32)
    pose_center = create_camera_pose(rotation_center, translation_center)

    # View 2: Right 90° (camera looks down +X axis)
    rotation_right = create_rotation_matrix_y(-90.0)
    translation_right = torch.tensor([-camera_distance, 0.0, 0.0], dtype=torch.float32)
    pose_right = create_camera_pose(rotation_right, translation_right)

    # Stack all poses
    extrinsics = torch.stack([pose_left, pose_center, pose_right], dim=0).unsqueeze(0)  # [1, 3, 4, 4]

    print(f"  View 0 (Left 90°):\n{extrinsics[0, 0]}")
    print(f"  View 1 (Head-on):\n{extrinsics[0, 1]}")
    print(f"  View 2 (Right 90°):\n{extrinsics[0, 2]}")

    # Normalized intrinsics (same for all views)
    # Example: fx=fy=800, cx=480, cy=256 for 960x512 image
    intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
    # from colmap: 1152 focal length
    fx, fy = 1152.0, 1152.0  # Focal length in pixels
    # fx, fy = 800.0, 800.0  # Focal length in pixels
    cx, cy = width / 2.0, height / 2.0  # Principal point at center (480, 256)
    intrinsics[:, :, 0, 0] = fx / width   # fx normalized
    intrinsics[:, :, 1, 1] = fy / height  # fy normalized
    intrinsics[:, :, 0, 2] = cx / width   # cx normalized
    intrinsics[:, :, 1, 2] = cy / height  # cy normalized

    print(f"\n  Intrinsics (normalized):")
    print(f"    fx: {fx/width:.4f}, fy: {fy/height:.4f}")
    print(f"    cx: {cx/width:.4f}, cy: {cy/height:.4f}")

    # Near and far planes
    near = torch.ones(batch_size, num_views, dtype=torch.float32) * 0.1
    far = torch.ones(batch_size, num_views, dtype=torch.float32) * 100.0

    # Prepare context
    context = {
        "image": images.to(device),
        "extrinsics": extrinsics.to(device),
        "intrinsics": intrinsics.to(device),
        "near": near.to(device),
        "far": far.to(device),
    }

    # Prepare visualization dump to capture scales and rotations for PLY export
    visualization_dump = {}

    # Run encoder
    print("\n" + "="*70)
    print("Running Encoder Inference")
    print("="*70)
    with torch.no_grad():
        result = encoder(
            context=context,
            global_step=0,
            deterministic=False,
            visualization_dump=visualization_dump,
            scene_names=None,
        )

    # Handle both dict and direct gaussians return
    if isinstance(result, dict):
        gaussians = result["gaussians"]
        depths = result.get("depths", None)
        if depths is not None:
            print(f"  Depths: {depths.shape}")
        if gaussians is None:
            raise ValueError("Encoder returned None for gaussians. Check config (train_depth_only should be False).")
    else:
        gaussians = result

    print(f"\nGaussian Splat Output:")
    print(f"  Means: {gaussians.means.shape}")
    print(f"  Covariances: {gaussians.covariances.shape}")
    print(f"  Harmonics: {gaussians.harmonics.shape}")
    print(f"  Opacities: {gaussians.opacities.shape}")

    # Export to PLY
    print("\n" + "="*70)
    print("Exporting to PLY")
    print("="*70)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "gaussians.ply"

    # Check if visualization dump contains required data
    if "scales" in visualization_dump and "rotations" in visualization_dump:
        scales = visualization_dump["scales"][0]  # [num_gaussians, 3]
        rotations = visualization_dump["rotations"][0]  # [num_gaussians, 4] (xyzw format)

        # Use the center view's extrinsics as reference for the PLY export
        reference_extrinsics = context["extrinsics"][0, 1].detach().cpu()  # Use head-on view

        # Export to PLY
        export_ply(
            extrinsics=reference_extrinsics,
            means=gaussians.means[0].detach().cpu(),  # [num_gaussians, 3]
            scales=scales.detach().cpu(),  # [num_gaussians, 3]
            rotations=rotations.detach().cpu(),  # [num_gaussians, 4] (xyzw)
            harmonics=gaussians.harmonics[0].detach().cpu(),  # [num_gaussians, 3, d_sh]
            opacities=gaussians.opacities[0].detach().cpu(),  # [num_gaussians]
            path=ply_path,
        )
        print(f"✓ Successfully exported {gaussians.means.shape[1]} Gaussians to {ply_path}")
        print(f"  File size: {ply_path.stat().st_size / (1024*1024):.2f} MB")
    else:
        print("✗ Warning: visualization_dump does not contain scales/rotations.")
        print("  Cannot export to PLY without this information.")
        print("  This may happen if the encoder config has certain settings.")
        print(f"  Available keys in visualization_dump: {list(visualization_dump.keys())}")
        
        # Try to export with scales/rotations from gaussians if available
        # Note: The gaussians object from the adapter should have scales and rotations
        # But they're not in world space, so this is a fallback
        print("\n  Note: The visualization_dump should be populated by the encoder.")
        print("  If this is missing, check that the encoder is configured correctly.")

    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
