"""
Run encoder using direct config loading (no Hydra) and export to PLY.

Image resolution: 512x960
3 input views: left 90°, head-on, right 90°
"""

# ============================================================================
# Configuration - Modify these paths as needed
# ============================================================================

CHECKPOINT_PATH = "pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"  # Set to None for random init
CONFIG_ROOT = "/content/depthsplat/config"  # Path to config directory
OUTPUT_DIR = "/content/drive/MyDrive/DepthSplat/run-output"

# Input image paths (set to None to use random images)
IMAGE_LEFT_PATH = "/content/drive/MyDrive/DepthSplat/3_new_input/frame_0002.png"   # View 0: 90° left
IMAGE_CENTER_PATH = "/content/drive/MyDrive/DepthSplat/3_new_input/frame_0070.png"  # View 1: Head-on
IMAGE_RIGHT_PATH = "/content/drive/MyDrive/DepthSplat/3_new_input/frame_0140.png"  # View 2: 90° right

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
# Set to None to use computed poses, or provide list of 3 numpy arrays or torch tensors
# Each matrix should be 4x4 in shape
# If provided, must have exactly 3 matrices (one per view)
EXTRINSICS_HARDCODED = None
# Example (uncomment to use - note: numpy is already imported as np):
EXTRINSICS_HARDCODED = [
     # View 0: Left 90° (camera looks down -X axis)
     np.array([
        [0.7534, 0.0369, -0.6566, 4.5406],
        [-0.0366, 0.9992, 0.0142, -0.0882],
        [0.6566, 0.0133, 0.7541, -2.1090],
        [0.0000, 0.0000, 0.0000, 1.0000],
     ], dtype=np.float32),
     # View 1: Head-on (camera looks down +Z axis)
     np.array([
        [-0.1835, -0.0286, 0.9826, -2.6224],
        [0.1672, 0.9841, 0.0599, -0.3042],
        [-0.9687, 0.1753, -0.1758, 2.1813],
        [0.0000, 0.0000, 0.0000, 1.0000]
     ], dtype=np.float32),
     # View 2: Right 90° (camera looks down +X axis)
     np.array([
        [0.9055, -0.0566, 0.4205, -0.0873],
        [0.0506, 0.9984, 0.0254, -0.1815],
        [-0.4212, -0.0017, 0.9070, -2.0622],
        [0.0000, 0.0000, 0.0000, 1.0000]
     ], dtype=np.float32),
 ]

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
from src.model.ply_export import export_ply
from src.misc.image_io import load_image
from einops import rearrange
import math
import torchvision.transforms as tf
from src.geometry.projection import get_fov
from src.dataset.shims.bounds_shim import compute_depth_for_disparity


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

    # Create input with resolution matching COLMAP reconstruction
    # NOTE: Dimensions must match what COLMAP used during reconstruction
    # Based on principal point analysis: COLMAP used width=512, height=960
    batch_size = 1
    num_views = 3  # Left, center, right
    height, width = 960, 512  # FIXED: Match COLMAP dimensions (was 512, 960)

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

    # Create camera poses for 3 views
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
        if len(EXTRINSICS_HARDCODED) != num_views:
            raise ValueError(f"EXTRINSICS_HARDCODED must contain exactly {num_views} matrices, got {len(EXTRINSICS_HARDCODED)}")
        
        extrinsics_list = []
        camera_centers = []
        camera_distances = []
        viewing_directions = []
        
        for i, ext in enumerate(EXTRINSICS_HARDCODED):
            if isinstance(ext, np.ndarray):
                ext_tensor = torch.from_numpy(ext).float()
            elif isinstance(ext, torch.Tensor):
                ext_tensor = ext.float()
            else:
                raise TypeError(f"Extrinsic {i} must be numpy array or torch tensor, got {type(ext)}")
            
            if ext_tensor.shape != (4, 4):
                raise ValueError(f"Extrinsic {i} must be 4x4 matrix, got shape {ext_tensor.shape}")
            
            # Normalize the rotation matrix (3x3 upper-left block) to ensure it's valid
            rotation = ext_tensor[:3, :3]
            rotation_normalized = normalize_rotation_matrix(rotation)
            ext_tensor[:3, :3] = rotation_normalized
            
            # Check determinant for validation
            det = torch.det(rotation_normalized)
            if not torch.allclose(det, torch.tensor(1.0), atol=1e-5):
                print(f"  WARNING: View {i} rotation matrix determinant after normalization: {det.item():.6f} (should be 1.0)")
            
            # Extract camera center (translation part of C2W matrix)
            camera_center = ext_tensor[:3, 3]
            camera_centers.append(camera_center)
            camera_distances.append(torch.norm(camera_center).item())
            
            # Extract viewing direction (camera looks down +Z in camera space)
            # In C2W matrix, the third column of rotation is the camera's +Z axis in world space
            view_dir = rotation_normalized[:, 2]  # Camera's forward direction in world coordinates
            viewing_directions.append(view_dir)
            
            extrinsics_list.append(ext_tensor)
        
        extrinsics = torch.stack(extrinsics_list, dim=0).unsqueeze(0)  # [1, 3, 4, 4]
        
        # Print detailed extrinsics information
        print(f"\n  Extrinsics Validation:")
        print(f"    Camera positions (world coordinates):")
        for i, center in enumerate(camera_centers):
            dist = camera_distances[i]
            print(f"      View {i}: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] (distance from origin: {dist:.3f})")
        
        print(f"\n    Viewing directions (camera forward in world coordinates):")
        for i, view_dir in enumerate(viewing_directions):
            print(f"      View {i}: [{view_dir[0]:.3f}, {view_dir[1]:.3f}, {view_dir[2]:.3f}]")
        
        # Check camera distances consistency
        if len(set([round(d, 1) for d in camera_distances])) > 1:
            print(f"\n    WARNING: Camera distances vary significantly:")
            for i, dist in enumerate(camera_distances):
                print(f"      View {i}: {dist:.3f}")
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
                print(f"      View {i} <-> View {j}: {baseline:.3f}")
        
        # Check viewing angles between cameras
        print(f"\n    Viewing angles between cameras:")
        for i in range(len(viewing_directions)):
            for j in range(i + 1, len(viewing_directions)):
                angle_rad = torch.acos(torch.clamp(torch.dot(viewing_directions[i], viewing_directions[j]), -1.0, 1.0))
                angle_deg = angle_rad * 180 / math.pi
                print(f"      View {i} <-> View {j}: {angle_deg:.1f}°")
        
        # Print full extrinsic matrices
        print(f"\n    Full extrinsic matrices (C2W):")
        for i, ext in enumerate(extrinsics_list):
            print(f"      View {i}:")
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
