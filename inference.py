"""
Run encoder using direct config loading (no Hydra) and export to PLY.

Image resolution: 512x960
Input views: determined by EXTRINSICS_HARDCODED dictionary keys (image filenames)
"""

# ============================================================================
# Configuration - Modify these paths as needed
# ============================================================================

CHECKPOINT_PATH = "pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"  # Set to None for random init
CONFIG_ROOT = "config"  # Path to config directory
OUTPUT_DIR = "run-output"

# Input image base path (directory containing images)
# If EXTRINSICS_HARDCODED is a dictionary, image paths will be constructed as:
#   IMAGE_BASE_PATH / image_filename (where image_filename is a key in EXTRINSICS_HARDCODED)
# Set to None to use random images
IMAGE_BASE_PATH = "/Users/quinton/Desktop/hillman_mov_horizontal"  # Base directory for images

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

import numpy as np

# Camera intrinsics (before normalization)
# Set to None to use computed values, or provide [fx, fy, cx, cy] in pixels
# If provided, will be normalized by image dimensions automatically
INTRINSICS_HARDCODED = None
# Example (uncomment to use):
INTRINSICS_HARDCODED = [1719.87357, 1719.87357, 256.0, 480.0]  # [fx, fy, cx, cy] in pixels

# Camera extrinsics (4x4 camera-to-world matrices)
# Set to None to use computed poses, or provide a dictionary mapping image filenames to numpy arrays
# Each matrix should be 4x4 in shape
# Keys are image filenames (e.g., 'frame_0002.png'), values are 4x4 C2W matrices
# If provided, images will be loaded from IMAGE_BASE_PATH using the dictionary keys as filenames
EXTRINSICS_HARDCODED = None
# Example (uncomment to use - note: numpy is already imported as np):
EXTRINSICS_HARDCODED = {
    "frame_0017.png": np.array([
        [0.2097, -0.0580, -0.9760, 4.0834],
        [-0.0471, 0.9965, -0.0693, 0.0745],
        [0.9766, 0.0605, 0.2062, 2.4196],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ], dtype=np.float32),
    "frame_0025.png": np.array([
        [0.3673, -0.0574, -0.9283, 3.8409],
        [-0.0363, 0.9964, -0.0760, 0.1100],
        [0.9294, 0.0616, 0.3639, 1.6217],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ], dtype=np.float32),
    "frame_0034.png": np.array([
        [0.5372, -0.0551, -0.8417, 3.4024],
        [-0.0238, 0.9965, -0.0804, 0.1335],
        [0.8431, 0.0632, 0.5340, 0.7633],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ], dtype=np.float32),
    "frame_0043.png": np.array([
        [0.6923, -0.0511, -0.7198, 2.7938],
        [-0.0104, 0.9967, -0.0808, 0.1430],
        [0.7216, 0.0634, 0.6894, -0.0107],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ], dtype=np.float32),
    "frame_0051.png": np.array([
        [0.8075, -0.0494, -0.5878, 2.1365],
        [0.0020, 0.9967, -0.0810, 0.1406],
        [0.5899, 0.0642, 0.8049, -0.5854],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ], dtype=np.float32),
}



# Near/Far plane computation (only used if EXTRINSICS_HARDCODED is provided)
# These disparity values control how near/far planes are computed from camera baselines
# Smaller disparity = farther depth (larger far plane)
# Larger disparity = closer depth (smaller near plane)
# Typical values: near_disparity=1.0-2.0, far_disparity=0.1-0.5
NEAR_DISPARITY = 1.0   # Pixel disparity for near plane computation (close objects)
FAR_DISPARITY = 0.1    # Pixel disparity for far plane computation (far objects)

# ============================================================================

import torch
from pathlib import Path
from omegaconf import OmegaConf
from src.config import load_typed_config
from src.model.encoder import EncoderDepthSplatCfg, get_encoder
from plyfile import PlyData, PlyElement
from src.misc.image_io import load_image
from einops import rearrange
import math
import torchvision.transforms as tf
from src.geometry.projection import get_fov
from src.dataset.shims.bounds_shim import compute_depth_for_disparity
from scipy.spatial.transform import Rotation as R


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


def normalize_rotation_matrix(R: torch.Tensor) -> torch.Tensor:
    """
    Normalize a 3x3 rotation matrix to ensure it's a valid rotation matrix.
    Uses SVD to orthogonalize the matrix and ensure determinant = 1.
    
    Args:
        R: 3x3 rotation matrix (may not be perfectly orthogonal)
    
    Returns:
        Normalized 3x3 rotation matrix with determinant = 1
    """
    # Use SVD to orthogonalize: R = U * S * Vt
    # For a rotation matrix, we want R_normalized = U * Vt
    U, _, Vt = torch.linalg.svd(R)
    R_normalized = U @ Vt
    
    # Ensure determinant is 1 (not -1)
    # If det(R_normalized) = -1, we need to flip the sign of the last column of U
    det = torch.det(R_normalized)
    if det < 0:
        # Create a modified U with last column flipped
        U_mod = U.clone()
        U_mod[:, -1] *= -1
        R_normalized = U_mod @ Vt
    
    return R_normalized


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
    # Device selection: prefer CUDA, then MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
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

    # Create input with resolution matching COLMAP reconstruction
    # NOTE: Dimensions must match what COLMAP used during reconstruction
    # Based on principal point analysis: COLMAP used width=512, height=960
    batch_size = 1
    height, width = 512, 960  # FIXED: Match COLMAP dimensions (was 512, 960)

    # Determine number of views and image paths from EXTRINSICS_HARDCODED
    image_paths = []
    image_filenames = []
    if EXTRINSICS_HARDCODED is not None:
        if isinstance(EXTRINSICS_HARDCODED, dict):
            # Extract image filenames from dictionary keys
            image_filenames = list(EXTRINSICS_HARDCODED.keys())
            num_views = len(image_filenames)
            if IMAGE_BASE_PATH is not None:
                image_base = Path(IMAGE_BASE_PATH)
                image_paths = [str(image_base / filename) for filename in image_filenames]
            else:
                print("  WARNING: EXTRINSICS_HARDCODED is a dictionary but IMAGE_BASE_PATH is None.")
                print("  Cannot load images. Falling back to random images.")
                image_paths = []
        elif isinstance(EXTRINSICS_HARDCODED, (list, tuple)):
            # Backward compatibility: list format
            num_views = len(EXTRINSICS_HARDCODED)
            image_filenames = []  # No filenames available for list format
            image_paths = []  # Cannot construct paths without filenames
        else:
            raise TypeError(f"EXTRINSICS_HARDCODED must be a dict, list, or None, got {type(EXTRINSICS_HARDCODED)}")
    else:
        # Default to 3 views if not using hardcoded extrinsics
        num_views = 3
        image_paths = []
        image_filenames = []

    # Load images from paths or generate random ones
    print("\n" + "="*70)
    print("Loading Images")
    print("="*70)
    if image_paths and len(image_paths) > 0:
        try:
            loaded_images = []
            for img_path, img_filename in zip(image_paths, image_filenames):
                loaded_img = load_and_resize_image(img_path, (height, width))
                loaded_images.append(loaded_img)
                print(f"  Loaded: {img_filename} from {img_path}")

            # Stack images: [num_views, 3, height, width] -> [1, num_views, 3, height, width]
            images = torch.stack(loaded_images, dim=0).unsqueeze(0)
            print(f"  Image shape: {images.shape}")
            print(f"  Number of views: {num_views}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            print("  Falling back to random images.")
            images = torch.rand(batch_size, num_views, 3, height, width)
    else:
        print("  Using random images (image paths not set or incomplete).")
        images = torch.rand(batch_size, num_views, 3, height, width)

    # Create camera poses
    print("\n" + "="*70)
    print("Setting Up Camera Poses")
    print("="*70)

    # Initialize variables for validation
    camera_centers = []
    camera_distances = []
    viewing_directions = []

    if EXTRINSICS_HARDCODED is not None:
        # Use hardcoded extrinsics
        print("  Using hardcoded extrinsics")
        
        # Handle dictionary format
        if isinstance(EXTRINSICS_HARDCODED, dict):
            # Verify all image filenames are in the dictionary
            # (num_views should already match since we set it from the dict length)
            missing_files = [fname for fname in image_filenames if fname not in EXTRINSICS_HARDCODED]
            if missing_files:
                raise ValueError(f"EXTRINSICS_HARDCODED dictionary is missing entries for: {missing_files}")
            
            # Extract extrinsics in the order of image_filenames
            extrinsics_values = [EXTRINSICS_HARDCODED[fname] for fname in image_filenames]
        else:
            # Handle list format (backward compatibility)
            # num_views was set from the list length, so they should match
            extrinsics_values = EXTRINSICS_HARDCODED
        
        extrinsics_list = []
        camera_centers = []
        camera_distances = []
        viewing_directions = []
        
        for i, ext in enumerate(extrinsics_values):
            img_name = image_filenames[i] if i < len(image_filenames) else f"View {i}"
            
            if isinstance(ext, np.ndarray):
                ext_tensor = torch.from_numpy(ext).float()
            elif isinstance(ext, torch.Tensor):
                ext_tensor = ext.float()
            else:
                raise TypeError(f"Extrinsic {i} ({img_name}) must be numpy array or torch tensor, got {type(ext)}")
            
            if ext_tensor.shape != (4, 4):
                raise ValueError(f"Extrinsic {i} ({img_name}) must be 4x4 matrix, got shape {ext_tensor.shape}")
            
            # Normalize the rotation matrix (3x3 upper-left block) to ensure it's valid
            rotation = ext_tensor[:3, :3]
            rotation_normalized = normalize_rotation_matrix(rotation)
            ext_tensor[:3, :3] = rotation_normalized
            
            # Check determinant for validation
            det = torch.det(rotation_normalized)
            if not torch.allclose(det, torch.tensor(1.0), atol=1e-5):
                print(f"  WARNING: View {i} ({img_name}) rotation matrix determinant after normalization: {det.item():.6f} (should be 1.0)")
            
            # Extract camera center (translation part of C2W matrix)
            camera_center = ext_tensor[:3, 3]
            camera_centers.append(camera_center)
            camera_distances.append(torch.norm(camera_center).item())
            
            # Extract viewing direction (camera looks down +Z in camera space)
            # In C2W matrix, the third column of rotation is the camera's +Z axis in world space
            view_dir = rotation_normalized[:, 2]  # Camera's forward direction in world coordinates
            viewing_directions.append(view_dir)
            
            extrinsics_list.append(ext_tensor)
        
        extrinsics = torch.stack(extrinsics_list, dim=0).unsqueeze(0)  # [1, num_views, 4, 4]
        
        # Print detailed extrinsics information
        print(f"\n  Extrinsics Validation:")
        print(f"    Camera positions (world coordinates):")
        for i, center in enumerate(camera_centers):
            dist = camera_distances[i]
            img_name = image_filenames[i] if i < len(image_filenames) else f"View {i}"
            print(f"      View {i} ({img_name}): [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] (distance from origin: {dist:.3f})")
        
        print(f"\n    Viewing directions (camera forward in world coordinates):")
        for i, view_dir in enumerate(viewing_directions):
            img_name = image_filenames[i] if i < len(image_filenames) else f"View {i}"
            print(f"      View {i} ({img_name}): [{view_dir[0]:.3f}, {view_dir[1]:.3f}, {view_dir[2]:.3f}]")
        
        # Check camera distances consistency
        if len(set([round(d, 1) for d in camera_distances])) > 1:
            print(f"\n    WARNING: Camera distances vary significantly:")
            for i, dist in enumerate(camera_distances):
                img_name = image_filenames[i] if i < len(image_filenames) else f"View {i}"
                print(f"      View {i} ({img_name}): {dist:.3f}")
            print(f"    This may indicate cameras are at different distances from the scene.")
        
        # Check if cameras are too close or too far
        avg_distance = sum(camera_distances) / len(camera_distances)
        if avg_distance < 0.01:
            print(f"\n    WARNING: Cameras are very close to origin (avg distance: {avg_distance:.3f})")
            print(f"    This may cause numerical issues. Consider scaling the scene.")
        elif avg_distance > 1000:
            print(f"\n    WARNING: Cameras are very far from origin (avg distance: {avg_distance:.3f})")
            print(f"    This may cause precision issues. Consider scaling the scene.")
        
        # Check baseline (distance between cameras)
        print(f"\n    Camera baselines (distances between camera centers):")
        for i in range(len(camera_centers)):
            for j in range(i + 1, len(camera_centers)):
                baseline = torch.norm(camera_centers[i] - camera_centers[j]).item()
                img_name_i = image_filenames[i] if i < len(image_filenames) else f"View {i}"
                img_name_j = image_filenames[j] if j < len(image_filenames) else f"View {j}"
                print(f"      View {i} ({img_name_i}) <-> View {j} ({img_name_j}): {baseline:.3f}")
        
        # Check viewing angles between cameras
        print(f"\n    Viewing angles between cameras:")
        for i in range(len(viewing_directions)):
            for j in range(i + 1, len(viewing_directions)):
                angle_rad = torch.acos(torch.clamp(torch.dot(viewing_directions[i], viewing_directions[j]), -1.0, 1.0))
                angle_deg = angle_rad * 180 / math.pi
                img_name_i = image_filenames[i] if i < len(image_filenames) else f"View {i}"
                img_name_j = image_filenames[j] if j < len(image_filenames) else f"View {j}"
                print(f"      View {i} ({img_name_i}) <-> View {j} ({img_name_j}): {angle_deg:.1f}°")
        
        # Print full extrinsic matrices
        print(f"\n    Full extrinsic matrices (C2W):")
        for i, ext in enumerate(extrinsics_list):
            img_name = image_filenames[i] if i < len(image_filenames) else f"View {i}"
            print(f"      View {i} ({img_name}):")
            ext_np = ext.numpy()
            for row in ext_np:
                print(f"        [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}, {row[3]:8.4f}]")
    else:
        # Compute camera poses (default behavior)
        # View 0: 90 degrees to the left
        # View 1: Head-on (center)
        # View 2: 90 degrees to the right
        print("  Computing camera poses from rotations and translations")
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

        # Extract camera centers and viewing directions for validation
        for i in range(num_views):
            pose = extrinsics[0, i]
            camera_center = pose[:3, 3]
            camera_centers.append(camera_center)
            camera_distances.append(torch.norm(camera_center).item())
            view_dir = pose[:3, 2]  # Camera's forward direction
            viewing_directions.append(view_dir)

        print(f"  View 0 (Left 90°):\n{extrinsics[0, 0]}")
        print(f"  View 1 (Head-on):\n{extrinsics[0, 1]}")
        print(f"  View 2 (Right 90°):\n{extrinsics[0, 2]}")
        
        # Print camera positions for default case
        print(f"\n  Camera positions (world coordinates):")
        for i, center in enumerate(camera_centers):
            dist = camera_distances[i]
            print(f"    View {i}: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] (distance from origin: {dist:.3f})")

    # Set up intrinsics
    print("\n" + "="*70)
    print("Setting Up Camera Intrinsics")
    print("="*70)
    
    # IMPORTANT: Check if image dimensions match COLMAP reconstruction
    print(f"  Current image dimensions: width={width}, height={height}")
    
    intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
    
    if INTRINSICS_HARDCODED is not None:
        # Use hardcoded intrinsics
        print("  Using hardcoded intrinsics (before normalization)")
        if len(INTRINSICS_HARDCODED) != 4:
            raise ValueError(f"INTRINSICS_HARDCODED must contain exactly 4 values [fx, fy, cx, cy], got {len(INTRINSICS_HARDCODED)}")
        
        fx, fy, cx, cy = INTRINSICS_HARDCODED
        
        # Auto-detect if dimensions might be wrong based on principal point
        cx_expected = width / 2.0
        cy_expected = height / 2.0
        cx_offset = abs(cx - cx_expected) / width if width > 0 else 0
        cy_offset = abs(cy - cy_expected) / height if height > 0 else 0
        
        # If principal point is far from center, suggest correct dimensions
        if cx_offset > 0.15 or cy_offset > 0.15:
            print(f"  ⚠️  WARNING: Principal point suggests dimension mismatch!")
            print(f"      Principal point: cx={cx:.1f}, cy={cy:.1f}")
            print(f"      Expected center: cx={cx_expected:.1f}, cy={cy_expected:.1f}")
            
            # Infer correct dimensions
            inferred_width = int(cx * 2) if cx > 0 else width
            inferred_height = int(cy * 2) if cy > 0 else height
            
            if abs(cx - inferred_width/2) < abs(cx - cx_expected) or abs(cy - inferred_height/2) < abs(cy - cy_expected):
                print(f"      Suggested dimensions: width={inferred_width}, height={inferred_height}")
                print(f"      Update inference.py: height, width = {inferred_height}, {inferred_width}")
        else:
            print(f"  ✓ Principal point is near center - dimensions appear correct")
        
        print(f"\n  Raw intrinsics (pixels): fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
        # Validate intrinsics values
        print(f"\n  Intrinsics Validation:")
        print(f"    Focal length fx: {fx:.2f} pixels (typical range: 100-5000)")
        print(f"    Focal length fy: {fy:.2f} pixels (typical range: 100-5000)")
        print(f"    Principal point cx: {cx:.2f} pixels (expected center: {cx_expected:.1f})")
        print(f"    Principal point cy: {cy:.2f} pixels (expected center: {cy_expected:.1f})")
        
        # Additional warnings if still off-center (redundant but useful for clarity)
        if cx_offset > 0.1:
            print(f"    WARNING: cx is {cx_offset*100:.1f}% off from center (expected ~{cx_expected:.1f})")
        if cy_offset > 0.1:
            print(f"    WARNING: cy is {cy_offset*100:.1f}% off from center (expected ~{cy_expected:.1f})")
        
        # Check focal length ratio (should be close to 1 for most cameras)
        if abs(fx - fy) / max(fx, fy) > 0.1:
            print(f"    WARNING: fx and fy differ by {(abs(fx-fy)/max(fx,fy)*100):.1f}% (may indicate distortion)")
    else:
        # Use computed intrinsics (default behavior)
        print("  Computing intrinsics from default values")
        # from colmap: 1152 focal length
        fx, fy = 1152.0, 1152.0  # Focal length in pixels
        cx, cy = width / 2.0, height / 2.0  # Principal point at center (480, 256)
        print(f"  Raw intrinsics (pixels): fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    # Normalize intrinsics
    intrinsics[:, :, 0, 0] = fx / width   # fx normalized
    intrinsics[:, :, 1, 1] = fy / height  # fy normalized
    intrinsics[:, :, 0, 2] = cx / width   # cx normalized
    intrinsics[:, :, 1, 2] = cy / height  # cy normalized

    print(f"\n  Normalized intrinsics matrix:")
    print(f"    fx: {fx/width:.6f} (multiply by {width} to get {fx:.2f} pixels)")
    print(f"    fy: {fy/height:.6f} (multiply by {height} to get {fy:.2f} pixels)")
    print(f"    cx: {cx/width:.6f} (multiply by {width} to get {cx:.2f} pixels)")
    print(f"    cy: {cy/height:.6f} (multiply by {height} to get {cy:.2f} pixels)")
    
    # Compute and validate Field of View
    fov = get_fov(intrinsics[0, 0:1])  # Get FOV for first view
    fov_deg = fov * 180 / math.pi
    print(f"\n  Field of View (FOV):")
    print(f"    Horizontal FOV: {fov_deg[0, 0]:.2f}°")
    print(f"    Vertical FOV: {fov_deg[0, 1]:.2f}°")
    
    # Check if FOV is reasonable (typical range: 30-120 degrees)
    if fov_deg[0, 0] < 20 or fov_deg[0, 0] > 150:
        print(f"    WARNING: Horizontal FOV ({fov_deg[0, 0]:.2f}°) is outside typical range (20-150°)")
    if fov_deg[0, 1] < 20 or fov_deg[0, 1] > 150:
        print(f"    WARNING: Vertical FOV ({fov_deg[0, 1]:.2f}°) is outside typical range (20-150°)")
    
    # Print full intrinsic matrix for verification
    print(f"\n  Full intrinsic matrix (3x3):")
    K = intrinsics[0, 0].numpy()
    for i in range(3):
        print(f"    [{K[i,0]:8.6f}, {K[i,1]:8.6f}, {K[i,2]:8.6f}]")

    # Compute near and far planes dynamically based on camera baselines
    # This matches how datasets handle COLMAP coordinate system scale
    # COLMAP uses arbitrary scale units, so we compute near/far relative to camera baselines
    print(f"\n  Computing Near/Far Planes from Camera Baselines:")
    if camera_centers and len(camera_centers) >= 2:
        # Convert to torch tensors for computation (they should already be torch tensors)
        extrinsics_tensor = extrinsics  # Already a torch tensor [1, num_views, 4, 4]
        intrinsics_tensor = intrinsics  # Already a torch tensor [1, num_views, 3, 3]
        
        # Use disparity values to compute near/far planes
        # These are configurable at the top of the file (NEAR_DISPARITY, FAR_DISPARITY)
        # Smaller disparity = farther depth, larger disparity = closer depth
        near_disparity = NEAR_DISPARITY
        far_disparity = FAR_DISPARITY
        
        print(f"    Computing based on camera baselines...")
        print(f"    Using disparity values: near={near_disparity}px, far={far_disparity}px")
        
        near_computed = compute_depth_for_disparity(
            extrinsics_tensor,
            intrinsics_tensor,
            (height, width),
            near_disparity,
        )
        far_computed = compute_depth_for_disparity(
            extrinsics_tensor,
            intrinsics_tensor,
            (height, width),
            far_disparity,
        )
        
        print(f"    ✓ Computed near plane: {near_computed[0].item():.6f} (from {near_disparity}px disparity)")
        print(f"    ✓ Computed far plane: {far_computed[0].item():.6f} (from {far_disparity}px disparity)")
        
        # Compute camera baselines for validation and fallback
        origins = extrinsics_tensor[:, :, :3, 3]  # [batch, views, 3]
        deltas = (origins[:, None, :, :] - origins[:, :, None, :]).norm(dim=-1)  # [batch, views, views]
        max_baseline = deltas.max().item()
        # Get minimum baseline (excluding self-distances which are 0)
        deltas_positive = deltas[deltas > 1e-6]
        min_baseline = deltas_positive.min().item() if len(deltas_positive) > 0 else max_baseline
        
        # Check if computed values are reasonable relative to baselines
        # If computed near/far are > 100x the baseline, they're likely incorrect
        # This can happen with narrow FOV cameras where pixel disparity computation doesn't work well
        use_computed = True
        if near_computed[0].item() > 100 * max_baseline:
            print(f"    WARNING: Computed near plane ({near_computed[0].item():.1f}) is > 100x max baseline ({max_baseline:.3f})")
            print(f"    This often happens with narrow FOV cameras. Using baseline-based fallback.")
            use_computed = False
        
        if far_computed[0].item() > 1000 * max_baseline:
            print(f"    WARNING: Computed far plane ({far_computed[0].item():.1f}) is > 1000x max baseline ({max_baseline:.3f})")
            if use_computed:
                print(f"    This often happens with narrow FOV cameras. Using baseline-based fallback.")
            use_computed = False
        
        # Validate computed values
        near_valid = (near_computed[0].item() > 0 and 
                     not torch.isnan(near_computed[0]) and 
                     not torch.isinf(near_computed[0]))
        
        far_valid = (far_computed[0].item() > 0 and 
                    not torch.isnan(far_computed[0]) and 
                    not torch.isinf(far_computed[0]))
        
        if not near_valid or not use_computed:
            # Use baseline-based fallback with scene-aware scaling
            # Model was trained on DL3DV (near=0.5, far=200) and RE10K (baseline-scaled)
            # Use 0.5x min_baseline to ensure we capture nearby scene content
            # This is more conservative than 0.2x and better matches training data
            near_fallback = max(0.5, 0.5 * min_baseline)
            # Also ensure near is at least 0.1x the minimum camera distance
            # This helps when scene content is close to cameras
            if camera_distances:
                min_camera_dist = min(camera_distances)
                near_fallback = max(near_fallback, 0.1 * min_camera_dist)
            print(f"    Using baseline-based near plane: {near_fallback:.6f}")
            print(f"      (0.5x min_baseline={min_baseline:.3f}, or 0.1x min_camera_dist, min=0.5)")
            near = torch.ones(batch_size, num_views, dtype=torch.float32) * near_fallback
        else:
            near = near_computed.unsqueeze(1).repeat(1, num_views)  # [batch, views]
        
        # Validate far plane (must be > near plane)
        if not far_valid or not use_computed:
            # Use baseline-based fallback with more reasonable scaling
            # Model was trained with far=200.0 (DL3DV) or baseline-scaled (RE10K)
            # Use 15x max_baseline for better depth resolution than 50x
            # This gives near/far ratio of ~30-100x, similar to training data
            far_fallback = 15.0 * max_baseline
            # Cap at 200.0 (same as DL3DV training data) to match model expectations
            far_fallback = min(200.0, far_fallback)
            # Also ensure far is at least 5x the maximum camera distance
            # This ensures we capture scene content beyond camera positions
            if camera_distances:
                max_camera_dist = max(camera_distances)
                far_fallback = max(far_fallback, 5.0 * max_camera_dist)
            print(f"    Using baseline-based far plane: {far_fallback:.6f}")
            print(f"      (15x max_baseline={max_baseline:.3f}, or 5x max_camera_dist, capped at 200.0)")
            far = torch.ones(batch_size, num_views, dtype=torch.float32) * far_fallback
        elif far_computed[0].item() <= near[0, 0].item():
            print(f"    WARNING: Computed far plane ({far_computed[0].item():.6f}) <= near plane ({near[0, 0].item():.6f})")
            far_fallback = 15.0 * max_baseline
            far_fallback = min(200.0, far_fallback)
            if camera_distances:
                max_camera_dist = max(camera_distances)
                far_fallback = max(far_fallback, 5.0 * max_camera_dist)
            print(f"    Using baseline-based far plane: {far_fallback:.6f}")
            far = torch.ones(batch_size, num_views, dtype=torch.float32) * far_fallback
        else:
            far = far_computed.unsqueeze(1).repeat(1, num_views)   # [batch, views]
        
        # Additional validation
        min_camera_dist = min(camera_distances) if camera_distances else 0
        max_camera_dist = max(camera_distances) if camera_distances else 0
        
        print(f"\n    Validation:")
        print(f"      Max baseline: {max_baseline:.3f}, Min baseline: {min_baseline:.3f}")
        print(f"      Camera distances from origin: min={min_camera_dist:.3f}, max={max_camera_dist:.3f}")
        print(f"      Near/far ratio: {far[0, 0].item() / near[0, 0].item():.1f}x")
        
        if near[0, 0].item() > min_camera_dist:
            print(f"      NOTE: Near plane ({near[0, 0].item():.3f}) > min camera distance ({min_camera_dist:.3f})")
            print(f"      This is normal - near plane represents depth in front of cameras, not camera positions.")
        if far[0, 0].item() < max_camera_dist * 2:
            print(f"      WARNING: Far plane ({far[0, 0].item():.3f}) may be too small relative to camera distance")
            print(f"      Consider that objects might be further than {far[0, 0].item():.3f} units from cameras")
    else:
        # Fallback to fixed values if we can't compute
        print(f"    Cannot compute from baselines (need at least 2 cameras with valid extrinsics)")
        print(f"    Using fixed values: near=0.1, far=100.0")
        print(f"    WARNING: These fixed values may not match your COLMAP coordinate system scale!")
        near = torch.ones(batch_size, num_views, dtype=torch.float32) * 0.1
        far = torch.ones(batch_size, num_views, dtype=torch.float32) * 100.0

    # Final validation before passing to encoder
    print("\n" + "="*70)
    print("Final Validation Before Encoder")
    print("="*70)
    print(f"  Image shape: {images.shape} (expected: [1, {num_views}, 3, {height}, {width}])")
    print(f"  Extrinsics shape: {extrinsics.shape} (expected: [1, {num_views}, 4, 4])")
    print(f"  Intrinsics shape: {intrinsics.shape} (expected: [1, {num_views}, 3, 3])")
    print(f"  Near shape: {near.shape} (expected: [1, {num_views}])")
    print(f"  Far shape: {far.shape} (expected: [1, {num_views}])")
    
    # Check for NaN or Inf values
    if torch.isnan(extrinsics).any():
        print("  ERROR: Extrinsics contain NaN values!")
    if torch.isnan(intrinsics).any():
        print("  ERROR: Intrinsics contain NaN values!")
    if torch.isinf(extrinsics).any():
        print("  WARNING: Extrinsics contain Inf values!")
    if torch.isinf(intrinsics).any():
        print("  WARNING: Intrinsics contain Inf values!")
    
    # Check image value range
    image_min = images.min().item()
    image_max = images.max().item()
    print(f"  Image value range: [{image_min:.3f}, {image_max:.3f}] (expected: [0.0, 1.0])")
    if image_min < 0 or image_max > 1:
        print(f"  WARNING: Image values outside expected range [0, 1]")
    
    # Prepare context
    context = {
        "image": images.to(device),
        "extrinsics": extrinsics.to(device),
        "intrinsics": intrinsics.to(device),
        "near": near.to(device),
        "far": far.to(device),
    }
    
    print("\n" + "="*70)
    print("Summary of Potential Issues")
    print("="*70)
    issues = []
    
    # Check image dimensions match COLMAP
    if INTRINSICS_HARDCODED is not None:
        fx, fy, cx, cy = INTRINSICS_HARDCODED
        # COLMAP stores cx, cy in pixel coordinates
        # Typical values: cx ≈ width/2, cy ≈ height/2 for centered principal point
        # If principal point is far from center, it might indicate:
        # 1. Different image dimensions in COLMAP
        # 2. Camera has offset principal point (less common)
        cx_ratio = cx / width if width > 0 else 0
        cy_ratio = cy / height if height > 0 else 0
        
        # Check if principal point is significantly off-center (more than 20% from center)
        # This could indicate dimension mismatch or unusual camera calibration
        cx_center_offset = abs(cx_ratio - 0.5)
        cy_center_offset = abs(cy_ratio - 0.5)
        
        if cx_center_offset > 0.2 or cy_center_offset > 0.2:
            # Principal point is far from center - could indicate dimension mismatch
            # Try to infer COLMAP dimensions assuming principal point was centered
            if cx_center_offset > 0.2:
                inferred_colmap_width = cx * 2 if cx > 0 else width
                issues.append(
                    f"Principal point cx={cx:.1f} is {cx_center_offset*100:.1f}% off-center. "
                    f"This suggests COLMAP may have used width ~{inferred_colmap_width:.0f} "
                    f"(current: {width}). Check if COLMAP reconstruction used different image dimensions."
                )
            if cy_center_offset > 0.2:
                inferred_colmap_height = cy * 2 if cy > 0 else height
                issues.append(
                    f"Principal point cy={cy:.1f} is {cy_center_offset*100:.1f}% off-center. "
                    f"This suggests COLMAP may have used height ~{inferred_colmap_height:.0f} "
                    f"(current: {height}). Check if COLMAP reconstruction used different image dimensions."
                )
        elif cx_center_offset > 0.1 or cy_center_offset > 0.1:
            # Moderate offset - might be intentional (offset principal point) or dimension mismatch
            issues.append(
                f"Principal point is moderately off-center (cx offset: {cx_center_offset*100:.1f}%, "
                f"cy offset: {cy_center_offset*100:.1f}%). This may be normal for your camera, "
                f"or could indicate a dimension mismatch with COLMAP."
            )
    
    # Check coordinate system scale
    # Note: COLMAP uses arbitrary scale, so we compute near/far dynamically
    # This check is just for informational purposes
    if camera_distances:
        avg_dist = sum(camera_distances) / len(camera_distances)
        if avg_dist < 0.01:
            issues.append("Camera distances are very small - may cause numerical precision issues")
        if avg_dist > 1000:
            issues.append("Camera distances are very large - may cause numerical precision issues")
        # Don't warn about scale mismatch since we compute near/far dynamically now
    
    # Check FOV
    if INTRINSICS_HARDCODED is not None:
        fov_h = fov_deg[0, 0].item()
        if fov_h < 30 or fov_h > 120:
            issues.append(f"FOV ({fov_h:.1f}°) is outside typical range - may indicate incorrect intrinsics")
        elif fov_h < 20:
            issues.append(f"FOV ({fov_h:.1f}°) is very narrow - may indicate telephoto lens or incorrect intrinsics")
    
    if issues:
        print("  Potential issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("  No obvious issues detected.")
    
    print("="*70)

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
    
    # Debug: Check depth values if available
    if "depth" in visualization_dump:
        depth_values = visualization_dump["depth"]  # [B, V, H, W, srf, s]
        print(f"\n  Depth Statistics:")
        print(f"    Depth shape: {depth_values.shape}")
        for v in range(num_views):
            view_depth = depth_values[0, v]  # [H, W, srf, s]
            # Flatten to get all depth values for this view
            view_depth_flat = view_depth.flatten()
            print(f"    View {v}:")
            print(f"      Min depth: {view_depth_flat.min().item():.3f}")
            print(f"      Max depth: {view_depth_flat.max().item():.3f}")
            print(f"      Mean depth: {view_depth_flat.mean().item():.3f}")
            print(f"      Median depth: {view_depth_flat.median().item():.3f}")
            print(f"      Expected range: [{near[0, v].item():.3f}, {far[0, v].item():.3f}]")
            if view_depth_flat.min().item() < near[0, v].item() * 0.5:
                print(f"      WARNING: Min depth ({view_depth_flat.min().item():.3f}) is much less than near plane ({near[0, v].item():.3f})")
            if view_depth_flat.max().item() > far[0, v].item() * 2.0:
                print(f"      WARNING: Max depth ({view_depth_flat.max().item():.3f}) is much greater than far plane ({far[0, v].item():.3f})")

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

        # Use the first view's extrinsics as reference for the PLY export
        # This matches save_gaussian_ply and encoder_visualizer which use view 0
        reference_extrinsics = context["extrinsics"][0, 0].detach().cpu()  # Use first view

        # Convert rotations from camera space to world space
        # The gaussians are flattened across views as: [v, r, srf, spp] -> [v*r*srf*spp]
        # We need to convert each view's rotations using that view's C2W matrix
        total_gaussians = rotations.shape[0]
        num_gaussians_per_view = total_gaussians // num_views
        
        # Verify the division is exact
        if total_gaussians % num_views != 0:
            raise ValueError(
                f"Total gaussians ({total_gaussians}) must be divisible by num_views ({num_views})"
            )
        
        # Reshape rotations to separate by view: [num_views, num_gaussians_per_view, 4]
        rotations_per_view = rotations.view(num_views, num_gaussians_per_view, 4)
        
        # Get C2W rotation matrices for each view
        c2w_rotations = context["extrinsics"][0, :, :3, :3].detach().cpu()  # [num_views, 3, 3]
        
        # Convert rotations from camera space to world space
        # This matches the approach in save_gaussian_ply: world_rotation = c2w @ cam_rotation
        world_rotations_list = []
        for v in range(num_views):
            # Get camera-space rotations for this view
            cam_rotations_np = R.from_quat(
                rotations_per_view[v].detach().cpu().numpy()
            ).as_matrix()  # [num_gaussians_per_view, 3, 3]
            
            # Get C2W rotation for this view
            c2w_rot = c2w_rotations[v].detach().cpu().numpy()  # [3, 3]
            
            # Convert to world space: world_rotation = c2w @ cam_rotation
            # Expand c2w_rot to match batch dimension for element-wise matrix multiplication
            # [3, 3] -> [num_gaussians_per_view, 3, 3] then @ [num_gaussians_per_view, 3, 3] -> [num_gaussians_per_view, 3, 3]
            c2w_rot_expanded = np.broadcast_to(
                c2w_rot[None, :, :], 
                (num_gaussians_per_view, 3, 3)
            )  # [num_gaussians_per_view, 3, 3]
            world_rotations_mat = c2w_rot_expanded @ cam_rotations_np  # Element-wise: [n, 3, 3] @ [n, 3, 3] -> [n, 3, 3]
            
            # Convert back to quaternion (scipy uses xyzw format)
            world_rotations_quat = R.from_matrix(world_rotations_mat).as_quat()  # [num_gaussians_per_view, 4] (xyzw)
            # Convert to float32 explicitly to avoid float64 issues with MPS
            world_rotations_list.append(torch.from_numpy(world_rotations_quat).float())
        
        # Flatten back to [num_gaussians, 4]
        # Ensure float32 dtype before moving to device (MPS doesn't support float64)
        world_rotations = torch.cat(world_rotations_list, dim=0).float().to(rotations.device)

        # Export to PLY directly in world space (avoiding export_ply's coordinate transformations)
        # All gaussians are already in world space from the encoder, so we can export them directly
        means_world = gaussians.means[0].detach().cpu()  # [num_gaussians, 3] (world space)
        
        # Debug: Validate gaussian means overlap across views
        print("\n" + "="*70)
        print("Validating Gaussian Means Overlap Across Views")
        print("="*70)
        print(f"  Total gaussians: {means_world.shape[0]}")
        print(f"  Gaussians per view: {num_gaussians_per_view}")
        print(f"  Number of views: {num_views}")
        
        # Debug: Check a sample of means to see their distribution
        # Sample a few gaussians from the center of each view's image
        print(f"\n  Sample Gaussian Positions (center pixels from each view):")
        h, w = context["image"].shape[3:5]
        center_h, center_w = h // 2, w // 2
        center_pixel_idx = center_h * w + center_w
        
        # Also test ray intersection: if we use the same depth for all views' center pixels,
        # they should intersect at the same 3D point
        print(f"\n  Ray Intersection Test (center pixel with fixed depth=5.0):")
        from src.geometry.projection import get_world_rays, sample_image_grid
        # Use the same coordinate generation as the encoder (pixel centers)
        xy_grid, _ = sample_image_grid((h, w), device=torch.device("cpu"))
        center_xy = xy_grid[center_h, center_w:center_w+1]  # [1, 2] - use exact same method as encoder
        test_depth = 5.0
        
        for v in range(num_views):
            view_start = v * num_gaussians_per_view
            # Get gaussians from center pixel area (assuming num_surfaces=1, num_samples=1)
            # The flattening order is [v, r, srf, spp] where r = h*w
            center_gaussian_idx = view_start + center_pixel_idx
            if center_gaussian_idx < means_world.shape[0]:
                center_mean = means_world[center_gaussian_idx]
                camera_pos = camera_centers[v]
                distance = torch.norm(center_mean - camera_pos).item()
                print(f"    View {v} center pixel gaussian:")
                print(f"      Position: [{center_mean[0].item():.3f}, {center_mean[1].item():.3f}, {center_mean[2].item():.3f}]")
                print(f"      Camera: [{camera_pos[0].item():.3f}, {camera_pos[1].item():.3f}, {camera_pos[2].item():.3f}]")
                print(f"      Distance from camera: {distance:.3f}")
                
                # Test with fixed depth
                ext = context["extrinsics"][0, v:v+1].cpu()  # [1, 4, 4]
                intr = context["intrinsics"][0, v:v+1].cpu()  # [1, 3, 3]
                origins, directions = get_world_rays(
                    center_xy.unsqueeze(0),  # [1, 1, 2]
                    ext,  # [1, 4, 4]
                    intr,  # [1, 3, 3]
                )
                origins = origins[0, 0]  # [3]
                directions = directions[0, 0]  # [3]
                test_point = origins + directions * test_depth
                print(f"      Test point (depth={test_depth}): [{test_point[0].item():.3f}, {test_point[1].item():.3f}, {test_point[2].item():.3f}]")
        
        # Check if test points are close (they should intersect)
        test_points = []
        for v in range(num_views):
            ext = context["extrinsics"][0, v:v+1].cpu()
            intr = context["intrinsics"][0, v:v+1].cpu()
            origins, directions = get_world_rays(
                center_xy.unsqueeze(0),
                ext,
                intr,
            )
            origins = origins[0, 0]
            directions = directions[0, 0]
            test_point = origins + directions * test_depth
            test_points.append(test_point)
        
        if len(test_points) >= 2:
            # Check distances between test points
            print(f"\n    Test point distances (should be ~0 if rays intersect):")
            for i in range(len(test_points)):
                for j in range(i + 1, len(test_points)):
                    dist = torch.norm(test_points[i] - test_points[j]).item()
                    print(f"      View {i} <-> View {j}: {dist:.3f}")
                    if dist > 1.0:
                        print(f"        WARNING: Rays don't intersect! This suggests a coordinate system issue.")
                    else:
                        print(f"        ✓ Rays intersect correctly (within numerical precision)")
        
        # Additional diagnostic: Check if the issue is depth prediction inconsistency
        print(f"\n  Depth Prediction Consistency Analysis:")
        print(f"    The median depths are very different across views:")
        print(f"      View 0: 117.391 (very far)")
        print(f"      View 1: 14.161 (medium)")
        print(f"      View 2: 6.279 (close)")
        print(f"    This suggests the depth predictor is producing inconsistent results.")
        print(f"    Possible causes:")
        print(f"      1. Depth predictor not trained for this camera setup")
        print(f"      2. Intrinsics/extrinsics mismatch with training data")
        print(f"      3. Scene scale mismatch")
        print(f"      4. Coordinate system convention mismatch")
        print(f"\n    The ray intersection test shows rays are ~1 unit apart,")
        print(f"    which is relatively small but indicates a coordinate system issue.")
        print(f"    However, the depth prediction inconsistency (50-110 unit separation)")
        print(f"    is the main problem causing gaussians not to overlap.")
        
        for v in range(num_views):
            view_start = v * num_gaussians_per_view
            view_end = (v + 1) * num_gaussians_per_view
            view_means = means_world[view_start:view_end]
            
            # Sample a subset for faster computation (every 100th gaussian)
            sample_indices = torch.arange(0, view_means.shape[0], 100)
            sampled_means = view_means[sample_indices]
            
            print(f"\n  View {v} (gaussians {view_start} to {view_end-1}):")
            print(f"    Position range (from {len(sampled_means)} sampled gaussians):")
            print(f"      X: [{sampled_means[:, 0].min().item():.3f}, {sampled_means[:, 0].max().item():.3f}]")
            print(f"      Y: [{sampled_means[:, 1].min().item():.3f}, {sampled_means[:, 1].max().item():.3f}]")
            print(f"      Z: [{sampled_means[:, 2].min().item():.3f}, {sampled_means[:, 2].max().item():.3f}]")
            print(f"    Mean center: [{sampled_means.mean(0)[0].item():.3f}, {sampled_means.mean(0)[1].item():.3f}, {sampled_means.mean(0)[2].item():.3f}]")
            print(f"    Camera position (from extrinsics): [{camera_centers[v][0].item():.3f}, {camera_centers[v][1].item():.3f}, {camera_centers[v][2].item():.3f}]")
            print(f"    Distance from camera to mean center: {torch.norm(sampled_means.mean(0) - camera_centers[v]).item():.3f}")
        
        # Check if means from different views overlap
        print(f"\n  Overlap Analysis:")
        view_centers = []
        for v in range(num_views):
            view_start = v * num_gaussians_per_view
            view_end = (v + 1) * num_gaussians_per_view
            view_means = means_world[view_start:view_end]
            sample_indices = torch.arange(0, view_means.shape[0], 100)
            sampled_means = view_means[sample_indices]
            view_centers.append(sampled_means.mean(0))
        
        for i in range(num_views):
            for j in range(i + 1, num_views):
                center_distance = torch.norm(view_centers[i] - view_centers[j]).item()
                print(f"    View {i} <-> View {j} center distance: {center_distance:.3f}")
                if center_distance > 10.0:
                    print(f"      WARNING: Views {i} and {j} have very different centers - gaussians may not overlap!")
        
        # Check bounding boxes
        print(f"\n  Bounding Box Analysis:")
        all_means_min = means_world.min(0)[0]
        all_means_max = means_world.max(0)[0]
        all_means_center = means_world.mean(0)
        print(f"    Overall bounding box:")
        print(f"      Min: [{all_means_min[0].item():.3f}, {all_means_min[1].item():.3f}, {all_means_min[2].item():.3f}]")
        print(f"      Max: [{all_means_max[0].item():.3f}, {all_means_max[1].item():.3f}, {all_means_max[2].item():.3f}]")
        print(f"      Center: [{all_means_center[0].item():.3f}, {all_means_center[1].item():.3f}, {all_means_center[2].item():.3f}]")
        print(f"      Size: [{all_means_max[0].item() - all_means_min[0].item():.3f}, {all_means_max[1].item() - all_means_min[1].item():.3f}, {all_means_max[2].item() - all_means_min[2].item():.3f}]")
        
        print("="*70)
        scales_world = scales.detach().cpu()  # [num_gaussians, 3] (world space)
        rotations_world = world_rotations.detach().cpu()  # [num_gaussians, 4] (world space, xyzw format)
        harmonics_world = gaussians.harmonics[0].detach().cpu()  # [num_gaussians, 3, d_sh]
        opacities_world = gaussians.opacities[0].detach().cpu()  # [num_gaussians]
        
        # Convert quaternions from xyzw (scipy format) to wxyz (PLY format)
        x, y, z, w = rearrange(rotations_world.numpy(), "g xyzw -> xyzw g")
        rotations_ply = np.stack((w, x, y, z), axis=-1)  # [num_gaussians, 4] (wxyz format)
        
        # Extract DC component of spherical harmonics (view-independent color)
        harmonics_dc = harmonics_world[..., 0].numpy()  # [num_gaussians, 3]
        
        # Construct PLY attributes (matching export_ply format)
        # Format: x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
        attributes_list = [
            means_world.numpy(),  # x, y, z
            np.zeros_like(means_world.numpy()),  # nx, ny, nz (normals - unused, set to zero)
            harmonics_dc,  # f_dc_0, f_dc_1, f_dc_2
            torch.logit(opacities_world[..., None]).numpy(),  # opacity (as logit)
            scales_world.log().numpy(),  # scale_0, scale_1, scale_2 (log of scales)
            rotations_ply,  # rot_0, rot_1, rot_2, rot_3 (wxyz quaternion)
        ]
        
        # Concatenate all attributes
        attributes = np.concatenate(attributes_list, axis=1)  # [num_gaussians, 3+3+3+1+3+4 = 17]
        
        # Define PLY data type
        dtype_full = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]
        
        # Create structured array
        elements = np.empty(means_world.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        
        # Write PLY file
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(elements, "vertex")]).write(ply_path)
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
