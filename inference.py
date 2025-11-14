"""
Run encoder using direct config loading (no Hydra) and export to PLY.

Image resolution: 960x512 (images are automatically resized to this size)
Camera intrinsics and extrinsics are loaded from metadata files (*_metadata.json)
"""

# ============================================================================
# Configuration - Modify these paths as needed
# ============================================================================

CHECKPOINT_PATH = "pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"  # Set to None for random init
CONFIG_ROOT = "config"  # Path to config directory
OUTPUT_DIR = "run-output"

# Input image base path (directory containing images and metadata files)
# Images and metadata files (*_metadata.json) should be in this directory
#IMAGE_BASE_PATH = "/Users/quinton/Desktop/hillman_mov_horizontal"  # Base directory for images
IMAGE_BASE_PATH = "/Users/quinton/repos/Image_sender/received_images"

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

# Toggle detailed validation and diagnostics during PLY export.
# Leave disabled for fastest export.
PLY_EXPORT_VALIDATION = False

# Torch compile / encoder benchmarking options
# Set to True to compile the encoder with torch.compile for faster repeated inference.
ENABLE_TORCH_COMPILE = False

# Number of times to run the encoder for timing/benchmarking.
# Set >1 to measure average runtime; the final run's output is used for export.
NUM_ENCODER_RUNS = 1

import numpy as np

# Camera intrinsics and extrinsics are loaded from metadata files
# See load_intrinsics_extrinsics_from_metadata() function



# Near/Far plane computation
# These disparity values control how near/far planes are computed from camera baselines
# Smaller disparity = farther depth (larger far plane)
# Larger disparity = closer depth (smaller near plane)
# Typical values: near_disparity=1.0-2.0, far_disparity=0.1-0.5
NEAR_DISPARITY = 1.0   # Pixel disparity for near plane computation (close objects)
FAR_DISPARITY = 0.1    # Pixel disparity for far plane computation (far objects)

# ============================================================================

import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from omegaconf import OmegaConf
from src.config import load_typed_config
from src.model.encoder import EncoderDepthSplatCfg, get_encoder
from plyfile import PlyData, PlyElement
from src.misc.image_io import load_image
from einops import rearrange
import math
import time
import torchvision.transforms as tf
from src.geometry.projection import get_fov
from src.dataset.shims.bounds_shim import compute_depth_for_disparity
from scipy.spatial.transform import Rotation as R
import json


@dataclass
class SetupResult:
    encoder: torch.nn.Module
    context: Dict[str, torch.Tensor]
    num_views: int
    output_dir: Path
    camera_centers: List[torch.Tensor]
    camera_distances: List[float]
    num_encoder_runs: int
    ply_export_validation: bool
    device: str


def load_metadata_from_json(image_base_path: str, image_filename: str) -> dict:
    """
    Load camera metadata from JSON file.
    
    Args:
        image_base_path: Base directory containing metadata files
        image_filename: Image filename (e.g., 'frame_0017.png')
    
    Returns:
        Dictionary with 'intrinsics', 'extrinsics', 'image_width', 'image_height', etc.
        Returns None if file not found.
    """
    base_name = Path(image_filename).stem  # Remove extension
    metadata_path = Path(image_base_path) / f"{base_name}_metadata.json"
    
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def load_intrinsics_extrinsics_from_metadata(image_base_path: str) -> tuple:
    """
    Load intrinsics and extrinsics from JSON metadata files in the image directory.
    
    Looks for files matching pattern: {image_name}_metadata.json
    
    Args:
        image_base_path: Base directory containing images and metadata files
    
    Returns:
        Tuple of (intrinsics_dict, extrinsics_dict) where:
        - intrinsics_dict: Dictionary mapping image filenames to [fx, fy, cx, cy]
        - extrinsics_dict: Dictionary mapping image filenames to 4x4 C2W numpy arrays
        Returns (None, None) if no metadata files found or if image_base_path is None
    """
    if image_base_path is None:
        return None, None
    
    base_path = Path(image_base_path)
    if not base_path.exists():
        return None, None
    
    # Find all metadata JSON files
    metadata_files = list(base_path.glob("*_metadata.json"))
    
    if len(metadata_files) == 0:
        return None, None
    
    intrinsics_dict = {}
    extrinsics_dict = {}
    
    for metadata_file in metadata_files:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Extract image filename from metadata filename
        # e.g., "frame_0017_metadata.json" -> "frame_0017.png"
        # We need to find the actual image file to get the extension
        base_name = metadata_file.stem.replace("_metadata", "")
        
        # Try common image extensions
        image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        image_filename = None
        for ext in image_extensions:
            candidate = base_path / f"{base_name}{ext}"
            if candidate.exists():
                image_filename = candidate.name
                break
        
        # If no image found, use .png as default
        if image_filename is None:
            image_filename = f"{base_name}.png"
        
        # Parse intrinsics from flattened 3x3 matrix
        # Format: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        intrinsics_flat = metadata.get("intrinsics", [])
        if len(intrinsics_flat) == 9:
            fx = intrinsics_flat[0]
            fy = intrinsics_flat[4]
            cx = intrinsics_flat[2]
            cy = intrinsics_flat[5]
            intrinsics_dict[image_filename] = [fx, fy, cx, cy]
        else:
            print(f"  WARNING: Invalid intrinsics format in {metadata_file.name}, expected 9 elements, got {len(intrinsics_flat)}")
            continue
        
        # Parse extrinsics from flattened 4x4 matrix
        # Format: 16 elements in row-major order
        extrinsics_flat = metadata.get("extrinsics", [])
        if len(extrinsics_flat) == 16:
            extrinsics_matrix = np.array(extrinsics_flat, dtype=np.float32).reshape(4, 4)
            extrinsics_dict[image_filename] = extrinsics_matrix
        else:
            print(f"  WARNING: Invalid extrinsics format in {metadata_file.name}, expected 16 elements, got {len(extrinsics_flat)}")
            continue
    
    if len(intrinsics_dict) == 0:
        return None, None
    
    return intrinsics_dict, extrinsics_dict


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


def quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two xyzw-format quaternions elementwise.

    Args:
        q1: Tensor of shape [..., 4] in xyzw format.
        q2: Tensor of shape [..., 4] in xyzw format.

    Returns:
        Tensor of shape [..., 4] representing the product q1 * q2 in xyzw format.
    """
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)

    # Hamilton product q = q1 * q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return torch.stack((x, y, z, w), dim=-1)


def get_image_dimensions(image_path: str) -> tuple[int, int]:
    """
    Get the dimensions of an image without loading the full image data.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Tuple of (height, width)
    """
    if image_path is None or not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image to get dimensions
    image = load_image(image_path)  # [3, H, W] in range [0, 1]
    height, width = image.shape[1], image.shape[2]
    return height, width


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


def validate_ply_export(
    context: dict,
    gaussians,
    means_world: torch.Tensor,
    num_views: int,
    num_gaussians_per_view: int,
    camera_centers: list[torch.Tensor],
) -> None:
    """
    Run detailed validation and diagnostics for exported Gaussians.
    Intended for debugging; guarded by PLY_EXPORT_VALIDATION flag.
    """
    print("\n" + "=" * 70)
    print("Validating Gaussian Means Overlap Across Views")
    print("=" * 70)
    print(f"  Total gaussians: {means_world.shape[0]}")
    print(f"  Gaussians per view: {num_gaussians_per_view}")
    print(f"  Number of views: {num_views}")

    print(f"\n  Sample Gaussian Positions (center pixels from each view):")
    h, w = context["image"].shape[3:5]
    center_h, center_w = h // 2, w // 2
    center_pixel_idx = center_h * w + center_w

    print(f"\n  Ray Intersection Test (center pixel with fixed depth=5.0):")
    from src.geometry.projection import get_world_rays, sample_image_grid

    xy_grid, _ = sample_image_grid((h, w), device=torch.device("cpu"))
    center_xy = xy_grid[center_h, center_w:center_w + 1]
    test_depth = 5.0

    for v in range(num_views):
        view_start = v * num_gaussians_per_view
        center_gaussian_idx = view_start + center_pixel_idx
        if center_gaussian_idx < means_world.shape[0]:
            center_mean = means_world[center_gaussian_idx]
            camera_pos = camera_centers[v]
            distance = torch.norm(center_mean - camera_pos).item()
            print(f"    View {v} center pixel gaussian:")
            print(
                f"      Position: [{center_mean[0].item():.3f}, "
                f"{center_mean[1].item():.3f}, {center_mean[2].item():.3f}]"
            )
            print(
                f"      Camera: [{camera_pos[0].item():.3f}, "
                f"{camera_pos[1].item():.3f}, {camera_pos[2].item():.3f}]"
            )
            print(f"      Distance from camera: {distance:.3f}")

            ext = context["extrinsics"][0, v:v + 1].cpu()
            intr = context["intrinsics"][0, v:v + 1].cpu()
            origins, directions = get_world_rays(
                center_xy.unsqueeze(0),
                ext,
                intr,
            )
            origins = origins[0, 0]
            directions = directions[0, 0]
            test_point = origins + directions * test_depth
            print(
                f"      Test point (depth={test_depth}): "
                f"[{test_point[0].item():.3f}, "
                f"{test_point[1].item():.3f}, {test_point[2].item():.3f}]"
            )

    test_points = []
    for v in range(num_views):
        ext = context["extrinsics"][0, v:v + 1].cpu()
        intr = context["intrinsics"][0, v:v + 1].cpu()
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
        print(f"\n    Test point distances (should be ~0 if rays intersect):")
        for i in range(len(test_points)):
            for j in range(i + 1, len(test_points)):
                dist = torch.norm(test_points[i] - test_points[j]).item()
                print(f"      View {i} <-> View {j}: {dist:.3f}")
                if dist > 1.0:
                    print(
                        f"        WARNING: Rays don't intersect! "
                        f"This suggests a coordinate system issue."
                    )
                else:
                    print(
                        f"        ✓ Rays intersect correctly "
                        f"(within numerical precision)"
                    )

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
    print(
        f"\n    The ray intersection test shows rays are ~1 unit apart,\n"
        f"    which is relatively small but indicates a coordinate system issue."
    )
    print(
        f"    However, the depth prediction inconsistency (50-110 unit separation)\n"
        f"    is the main problem causing gaussians not to overlap."
    )

    for v in range(num_views):
        view_start = v * num_gaussians_per_view
        view_end = (v + 1) * num_gaussians_per_view
        view_means = means_world[view_start:view_end]
        sample_indices = torch.arange(0, view_means.shape[0], 100)
        sampled_means = view_means[sample_indices]

        print(f"\n  View {v} (gaussians {view_start} to {view_end - 1}):")
        print(f"    Position range (from {len(sampled_means)} sampled gaussians):")
        print(
            f"      X: [{sampled_means[:, 0].min().item():.3f}, "
            f"{sampled_means[:, 0].max().item():.3f}]"
        )
        print(
            f"      Y: [{sampled_means[:, 1].min().item():.3f}, "
            f"{sampled_means[:, 1].max().item():.3f}]"
        )
        print(
            f"      Z: [{sampled_means[:, 2].min().item():.3f}, "
            f"{sampled_means[:, 2].max().item():.3f}]"
        )
        print(
            f"    Mean center: "
            f"[{sampled_means.mean(0)[0].item():.3f}, "
            f"{sampled_means.mean(0)[1].item():.3f}, "
            f"{sampled_means.mean(0)[2].item():.3f}]"
        )
        print(
            f"    Camera position (from extrinsics): "
            f"[{camera_centers[v][0].item():.3f}, "
            f"{camera_centers[v][1].item():.3f}, "
            f"{camera_centers[v][2].item():.3f}]"
        )
        print(
            f"    Distance from camera to mean center: "
            f"{torch.norm(sampled_means.mean(0) - camera_centers[v]).item():.3f}"
        )

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
                print(
                    f"      WARNING: Views {i} and {j} have very different centers - "
                    f"gaussians may not overlap!"
                )

    print(f"\n  Bounding Box Analysis:")
    all_means_min = means_world.min(0)[0]
    all_means_max = means_world.max(0)[0]
    all_means_center = means_world.mean(0)
    print(f"    Overall bounding box:")
    print(
        f"      Min: [{all_means_min[0].item():.3f}, "
        f"{all_means_min[1].item():.3f}, {all_means_min[2].item():.3f}]"
    )
    print(
        f"      Max: [{all_means_max[0].item():.3f}, "
        f"{all_means_max[1].item():.3f}, {all_means_max[2].item():.3f}]"
    )
    print(
        f"      Center: [{all_means_center[0].item():.3f}, "
        f"{all_means_center[1].item():.3f}, {all_means_center[2].item():.3f}]"
    )
    print(
        f"      Size: [{all_means_max[0].item() - all_means_min[0].item():.3f}, "
        f"{all_means_max[1].item() - all_means_min[1].item():.3f}, "
        f"{all_means_max[2].item() - all_means_min[2].item():.3f}]"
    )
    print("=" * 70)


def setup(
    checkpoint_path: Optional[Union[str, Path]] = None,
    config_root: Union[str, Path] = CONFIG_ROOT,
    image_base_path: Optional[Union[str, Path]] = IMAGE_BASE_PATH,
    encoder_overrides: Optional[Dict[str, Any]] = None,
    output_dir: Union[str, Path] = OUTPUT_DIR,
    num_encoder_runs: Optional[int] = None,
    enable_torch_compile: Optional[bool] = None,
    ply_export_validation: Optional[bool] = None,
    device: Optional[str] = None,
) -> SetupResult:
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATH
    if encoder_overrides is None:
        encoder_overrides = ENCODER_OVERRIDES
    if num_encoder_runs is None:
        num_encoder_runs = NUM_ENCODER_RUNS
    if enable_torch_compile is None:
        enable_torch_compile = ENABLE_TORCH_COMPILE
    if ply_export_validation is None:
        ply_export_validation = PLY_EXPORT_VALIDATION

    config_root = Path(config_root)
    output_dir = Path(output_dir)
    image_base_path = Path(image_base_path) if image_base_path is not None else None
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    # Device selection: prefer CUDA, then MPS (Apple Silicon), then CPU
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")
    
    # Load intrinsics and extrinsics from JSON metadata files
    print("\n" + "="*70)
    print("Loading Camera Metadata")
    print("="*70)
    
    intrinsics_from_metadata, extrinsics_from_metadata = load_intrinsics_extrinsics_from_metadata(image_base_path)
    
    if intrinsics_from_metadata is None or extrinsics_from_metadata is None:
        if image_base_path is not None:
            raise FileNotFoundError(
                f"No metadata files found in {image_base_path}. "
                f"Please ensure metadata files (e.g., '*_metadata.json') are present."
            )
        else:
            raise ValueError(
                "IMAGE_BASE_PATH is None. Please set IMAGE_BASE_PATH to a directory containing images and metadata files."
            )
    
    print(f"  Found {len(extrinsics_from_metadata)} metadata file(s) in {image_base_path}")
    print(f"  Loaded extrinsics for {len(extrinsics_from_metadata)} image(s)")
    
    # For intrinsics, we need to handle per-image intrinsics
    # If all images have the same intrinsics, use the first one
    # Otherwise, we'll need to handle per-image intrinsics later
    intrinsics_values = list(intrinsics_from_metadata.values())
    if len(set(tuple(v) for v in intrinsics_values)) == 1:
        # All images have the same intrinsics
        intrinsics_loaded = intrinsics_values[0]
        print(f"  Loaded intrinsics: fx={intrinsics_loaded[0]:.2f}, fy={intrinsics_loaded[1]:.2f}, "
              f"cx={intrinsics_loaded[2]:.2f}, cy={intrinsics_loaded[3]:.2f}")
    else:
        # Different intrinsics per image - use first one and warn
        intrinsics_loaded = intrinsics_values[0]
        first_image = list(intrinsics_from_metadata.keys())[0]
        print(f"  WARNING: Images have different intrinsics. Using intrinsics from {first_image}")
        print(f"  Loaded intrinsics: fx={intrinsics_loaded[0]:.2f}, fy={intrinsics_loaded[1]:.2f}, "
              f"cx={intrinsics_loaded[2]:.2f}, cy={intrinsics_loaded[3]:.2f}")
    
    # Store extrinsics for later use
    extrinsics_loaded = extrinsics_from_metadata

    # Load encoder config
    print("\n" + "="*70)
    print("Loading Encoder Config")
    print("="*70)
    encoder_cfg = load_encoder_config(str(config_root), encoder_overrides)

    # Initialize encoder
    print("\n" + "="*70)
    print("Initializing Encoder")
    print("="*70)
    encoder, encoder_visualizer = get_encoder(encoder_cfg)
    encoder = encoder.to(device)
    encoder.eval()

    # Optionally compile the encoder for faster repeated inference.
    if enable_torch_compile:
        if hasattr(torch, "compile"):
            try:
                encoder = torch.compile(encoder, mode="reduce-overhead")
                print("Encoder compiled with torch.compile (mode='reduce-overhead').")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}. Continuing without compilation.")
        else:
            print("Warning: torch.compile is not available in this PyTorch version; skipping compilation.")

    print("Encoder initialized successfully!")

    # Load checkpoint if provided
    if checkpoint_path and checkpoint_path.exists():
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

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
    elif checkpoint_path:
        print(f"\nWarning: Checkpoint path '{checkpoint_path}' does not exist. Using randomly initialized weights.")
    else:
        print("\nNo checkpoint path provided. Using randomly initialized weights.")

    # Determine number of views and image paths from loaded extrinsics
    batch_size = 1
    if isinstance(extrinsics_loaded, dict):
        # Extract image filenames from dictionary keys
        image_filenames = list(extrinsics_loaded.keys())
        num_views = len(image_filenames)
        if image_base_path is not None:
            image_base = image_base_path
            image_paths = [str(image_base / filename) for filename in image_filenames]
        else:
            raise ValueError("IMAGE_BASE_PATH is None but extrinsics were loaded from metadata.")
    else:
        raise TypeError(f"Extrinsics from metadata must be a dict, got {type(extrinsics_loaded)}")

    # Load images from paths or generate random ones
    print("\n" + "="*70)
    print("Loading Images")
    print("="*70)
    
    # Expected dimensions: 960 × 512 (width × height)
    EXPECTED_WIDTH = 960
    EXPECTED_HEIGHT = 512
    height, width = EXPECTED_HEIGHT, EXPECTED_WIDTH
    
    # Track original dimensions for intrinsics adjustment
    original_dimensions = None
    
    if image_paths and len(image_paths) > 0:
        try:
            # First, check original dimensions of first image
            first_image_path = image_paths[0]
            original_img = load_image(first_image_path)
            orig_height, orig_width = original_img.shape[1], original_img.shape[2]
            original_dimensions = (orig_height, orig_width)
            
            print(f"  Original image dimensions: {orig_width} × {orig_height}")
            print(f"  Target dimensions: {EXPECTED_WIDTH} × {EXPECTED_HEIGHT}")
            
            # Always resize to expected dimensions
            if orig_width != EXPECTED_WIDTH or orig_height != EXPECTED_HEIGHT:
                print(f"  Resizing images to {EXPECTED_WIDTH} × {EXPECTED_HEIGHT}")
                print(f"  Intrinsics will be adjusted accordingly")
            else:
                print(f"  Image dimensions already match target size")
            
            # Load and resize all images
            loaded_images = []
            for img_path, img_filename in zip(image_paths, image_filenames):
                loaded_img = load_and_resize_image(img_path, (height, width))
                loaded_images.append(loaded_img)
                print(f"  Loaded: {img_filename} from {img_path}")

            # Stack images: [num_views, 3, height, width] -> [1, num_views, 3, height, width]
            images = torch.stack(loaded_images, dim=0).unsqueeze(0)
            print(f"  Final image shape: {images.shape}")
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

    # Use extrinsics from metadata
    print("  Using extrinsics from metadata files")
    
    # Verify all image filenames are in the dictionary
    # (num_views should already match since we set it from the dict length)
    missing_files = [fname for fname in image_filenames if fname not in extrinsics_loaded]
    if missing_files:
        raise ValueError(f"Extrinsics dictionary is missing entries for: {missing_files}")
    
    # Extract extrinsics in the order of image_filenames
    extrinsics_values = [extrinsics_loaded[fname] for fname in image_filenames]
    
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

    # Set up intrinsics
    print("\n" + "="*70)
    print("Setting Up Camera Intrinsics")
    print("="*70)
    
    # IMPORTANT: Check if image dimensions match COLMAP reconstruction
    print(f"  Current image dimensions: width={width}, height={height}")
    
    # Use intrinsics from metadata and adjust if images were resized
    intrinsics_scale_factor_w = 1.0
    intrinsics_scale_factor_h = 1.0
    if original_dimensions is not None:
        orig_height, orig_width = original_dimensions
        if orig_width != width or orig_height != height:
            intrinsics_scale_factor_w = width / orig_width
            intrinsics_scale_factor_h = height / orig_height
            print(f"  Images were resized from {orig_width} × {orig_height} to {width} × {height}")
            print(f"  Intrinsics will be scaled by: width_scale={intrinsics_scale_factor_w:.4f}, height_scale={intrinsics_scale_factor_h:.4f}")
    
    intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
    
    # Use intrinsics from metadata
    print("  Using intrinsics from metadata files (before normalization)")
    if len(intrinsics_loaded) != 4:
        raise ValueError(f"Intrinsics must contain exactly 4 values [fx, fy, cx, cy], got {len(intrinsics_loaded)}")
    
    fx, fy, cx, cy = intrinsics_loaded
    
    # Adjust intrinsics if images were resized
    if original_dimensions is not None and (intrinsics_scale_factor_w != 1.0 or intrinsics_scale_factor_h != 1.0):
        print(f"  Original intrinsics (pixels): fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        fx = fx * intrinsics_scale_factor_w
        fy = fy * intrinsics_scale_factor_h
        cx = cx * intrinsics_scale_factor_w
        cy = cy * intrinsics_scale_factor_h
        print(f"  Adjusted intrinsics (pixels): fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        print(f"    (scaled by {intrinsics_scale_factor_w:.4f} × {intrinsics_scale_factor_h:.4f})")
    else:
        print(f"  Intrinsics (pixels): fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    
    # Validate intrinsics values
    cx_expected = width / 2.0
    cy_expected = height / 2.0
    cx_offset = abs(cx - cx_expected) / width if width > 0 else 0
    cy_offset = abs(cy - cy_expected) / height if height > 0 else 0
    
    print(f"\n  Intrinsics Validation:")
    print(f"    Focal length fx: {fx:.2f} pixels (typical range: 100-5000)")
    print(f"    Focal length fy: {fy:.2f} pixels (typical range: 100-5000)")
    print(f"    Principal point cx: {cx:.2f} pixels (expected center: {cx_expected:.1f})")
    print(f"    Principal point cy: {cy:.2f} pixels (expected center: {cy_expected:.1f})")
    
    # Additional warnings if still off-center
    if cx_offset > 0.1:
        print(f"    WARNING: cx is {cx_offset*100:.1f}% off from center (expected ~{cx_expected:.1f})")
    if cy_offset > 0.1:
        print(f"    WARNING: cy is {cy_offset*100:.1f}% off from center (expected ~{cy_expected:.1f})")
    
    # Check focal length ratio (should be close to 1 for most cameras)
    if abs(fx - fy) / max(fx, fy) > 0.1:
        print(f"    WARNING: fx and fy differ by {(abs(fx-fy)/max(fx,fy)*100):.1f}% (may indicate distortion)")
    
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
    
    return SetupResult(
        encoder=encoder,
        context=context,
        num_views=num_views,
        output_dir=output_dir,
        camera_centers=camera_centers,
        camera_distances=camera_distances,
        num_encoder_runs=num_encoder_runs,
        ply_export_validation=ply_export_validation,
        device=device,
    )


def run_encoder(
    setup_result: SetupResult,
    num_runs: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    visualization_dump: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the encoder inference loop and export the resulting Gaussians to a PLY file.

    Args:
        setup_result: Output of `setup()` containing encoder, context, and metadata.
        num_runs: Optional override for the number of encoder runs.
        output_dir: Optional override for the directory where the PLY will be saved.
        visualization_dump: Optional dictionary to populate with visualization tensors.

    Returns:
        Dictionary containing the encoder result, PLY path (if exported), visualization dump, and timing info.
    """
    encoder = setup_result.encoder
    context = setup_result.context
    num_views = setup_result.num_views
    output_dir_path = Path(output_dir) if output_dir is not None else setup_result.output_dir
    num_runs = num_runs if num_runs is not None else setup_result.num_encoder_runs
    if num_runs <= 0:
        raise ValueError("num_runs must be a positive integer")

    ply_export_validation = setup_result.ply_export_validation
    camera_centers = setup_result.camera_centers
    camera_distances = setup_result.camera_distances
    device = setup_result.device

    if visualization_dump is None:
        visualization_dump = {}

    near = context["near"]
    far = context["far"]

    print("\n" + "=" * 70)
    print("Running Encoder Inference")
    print("=" * 70)
    print(f"  Encoder will run {num_runs} time(s)")

    result: Any = None
    total_encoder_elapsed = 0.0
    encoder_elapsed = 0.0
    encoder_start_wall_time = time.time()
    encoder_end_wall_time = encoder_start_wall_time

    for run_idx in range(num_runs):
        print(f"\n  ----- Encoder run {run_idx + 1}/{num_runs} -----")
        encoder_start_time = time.perf_counter()
        encoder_start_wall_time = time.time()
        print(f"    Run start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(encoder_start_wall_time))}")

        with torch.no_grad():
            result = encoder(
                context=context,
                global_step=0,
                deterministic=False,
                visualization_dump=visualization_dump,
                scene_names=None,
            )

        encoder_end_time = time.perf_counter()
        encoder_end_wall_time = time.time()
        encoder_elapsed = encoder_end_time - encoder_start_time
        total_encoder_elapsed += encoder_elapsed
        print(f"    Run end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(encoder_end_wall_time))}")
        print(f"    Run elapsed time: {encoder_elapsed:.3f} seconds")

    avg_encoder_elapsed = total_encoder_elapsed / max(num_runs, 1)
    print(f"\n  Encoder total time over {num_runs} run(s): {total_encoder_elapsed:.3f} seconds")
    print(f"  Encoder average time per run: {avg_encoder_elapsed:.3f} seconds ({avg_encoder_elapsed/60:.2f} minutes)")

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

    if "depth" in visualization_dump:
        depth_values = visualization_dump["depth"]  # [B, V, H, W, srf, s]
        print(f"\n  Depth Statistics:")
        print(f"    Depth shape: {depth_values.shape}")
        for v in range(num_views):
            view_depth = depth_values[0, v]  # [H, W, srf, s]
            view_depth_flat = view_depth.flatten()
            print(f"    View {v}:")
            print(f"      Min depth: {view_depth_flat.min().item():.3f}")
            print(f"      Max depth: {view_depth_flat.max().item():.3f}")
            print(f"      Mean depth: {view_depth_flat.mean().item():.3f}")
            print(f"      Median depth: {view_depth_flat.median().item():.3f}")
            expected_near = near[0, v].item()
            expected_far = far[0, v].item()
            print(f"      Expected range: [{expected_near:.3f}, {expected_far:.3f}]")
            if view_depth_flat.min().item() < expected_near * 0.5:
                print(f"      WARNING: Min depth ({view_depth_flat.min().item():.3f}) is much less than near plane ({expected_near:.3f})")
            if view_depth_flat.max().item() > expected_far * 2.0:
                print(f"      WARNING: Max depth ({view_depth_flat.max().item():.3f}) is much greater than far plane ({expected_far:.3f})")

    print("\n" + "=" * 70)
    print("Exporting to PLY")
    print("=" * 70)
    ply_export_start_time = time.perf_counter()
    ply_export_start_wall_time = time.time()
    print(f"  PLY export start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ply_export_start_wall_time))}")

    output_dir_path.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir_path / "gaussians.ply"

    ply_export_elapsed: Optional[float] = None
    ply_export_end_wall_time: Optional[float] = None

    if "scales" in visualization_dump and "rotations" in visualization_dump:
        step_timings: Dict[str, float] = {}

        extract_start_time = time.perf_counter()
        scales = visualization_dump["scales"][0]  # [num_gaussians, 3]
        rotations = visualization_dump["rotations"][0]  # [num_gaussians, 4] (xyzw format)
        extract_end_time = time.perf_counter()
        step_timings["extract_scales_rotations"] = extract_end_time - extract_start_time

        extrinsics_start_time = time.perf_counter()
        reference_extrinsics = context["extrinsics"][0, 0].detach().cpu()
        extrinsics_end_time = time.perf_counter()
        step_timings["get_reference_extrinsics"] = extrinsics_end_time - extrinsics_start_time

        rotation_conv_start_time = time.perf_counter()
        total_gaussians = rotations.shape[0]
        num_gaussians_per_view = total_gaussians // num_views

        if total_gaussians % num_views != 0:
            raise ValueError(
                f"Total gaussians ({total_gaussians}) must be divisible by num_views ({num_views})"
            )

        c2w_rotations = context["extrinsics"][0, :, :3, :3]
        c2w_rotations_np = c2w_rotations.detach().cpu().numpy()
        c2w_quats_np = R.from_matrix(c2w_rotations_np).as_quat().astype(np.float32)
        c2w_quats = torch.from_numpy(c2w_quats_np).to(rotations.device)

        view_ids = torch.repeat_interleave(
            torch.arange(num_views, device=rotations.device),
            num_gaussians_per_view,
        )
        if view_ids.shape[0] != total_gaussians:
            raise ValueError(
                f"view_ids length ({view_ids.shape[0]}) does not match total_gaussians ({total_gaussians})"
            )

        c2w_gauss_quats = c2w_quats[view_ids]
        world_rotations = quat_mul_xyzw(c2w_gauss_quats, rotations)

        rotation_conv_end_time = time.perf_counter()
        step_timings["convert_rotations_to_world_space"] = rotation_conv_end_time - rotation_conv_start_time

        extract_props_start_time = time.perf_counter()
        means_world = gaussians.means[0].detach().cpu()

        if ply_export_validation:
            validate_ply_export(
                context=context,
                gaussians=gaussians,
                means_world=means_world,
                num_views=num_views,
                num_gaussians_per_view=num_gaussians_per_view,
                camera_centers=camera_centers,
            )
            print("=" * 70)

        scales_world = scales.detach().cpu()
        rotations_world = world_rotations.detach().cpu()
        harmonics_world = gaussians.harmonics[0].detach().cpu()
        opacities_world = gaussians.opacities[0].detach().cpu()
        extract_props_end_time = time.perf_counter()
        step_timings["extract_gaussian_properties"] = extract_props_end_time - extract_props_start_time

        quaternion_conv_start_time = time.perf_counter()
        x, y, z, w = rearrange(rotations_world.numpy(), "g xyzw -> xyzw g")
        rotations_ply = np.stack((w, x, y, z), axis=-1)
        quaternion_conv_end_time = time.perf_counter()
        step_timings["convert_quaternion_format"] = quaternion_conv_end_time - quaternion_conv_start_time

        harmonics_start_time = time.perf_counter()
        harmonics_dc = harmonics_world[..., 0].numpy()
        harmonics_end_time = time.perf_counter()
        step_timings["extract_harmonics_dc"] = harmonics_end_time - harmonics_start_time

        attributes_start_time = time.perf_counter()
        attributes_list = [
            means_world.numpy(),
            np.zeros_like(means_world.numpy()),
            harmonics_dc,
            torch.logit(opacities_world[..., None]).numpy(),
            scales_world.log().numpy(),
            rotations_ply,
        ]
        attributes = np.concatenate(attributes_list, axis=1)
        attributes_end_time = time.perf_counter()
        step_timings["construct_ply_attributes"] = attributes_end_time - attributes_start_time

        structured_array_start_time = time.perf_counter()
        dtype_full = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]

        elements = np.empty(means_world.shape[0], dtype=dtype_full)
        for i, name in enumerate(elements.dtype.names):
            elements[name] = attributes[:, i]
        structured_array_end_time = time.perf_counter()
        step_timings["create_structured_array"] = structured_array_end_time - structured_array_start_time

        ply_write_start_time = time.perf_counter()
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(elements, "vertex")]).write(ply_path)
        ply_write_end_time = time.perf_counter()
        step_timings["write_ply_file"] = ply_write_end_time - ply_write_start_time

        ply_export_end_time = time.perf_counter()
        ply_export_end_wall_time = time.time()
        ply_export_elapsed = ply_export_end_time - ply_export_start_time
        print(f"✓ Successfully exported {gaussians.means.shape[1]} Gaussians to {ply_path}")
        print(f"  File size: {ply_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"  PLY export end time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ply_export_end_wall_time))}")
        print(f"  PLY export elapsed time: {ply_export_elapsed:.3f} seconds ({ply_export_elapsed/60:.2f} minutes)")

        print("\n" + "=" * 70)
        print("PLY Conversion Step Timings")
        print("=" * 70)
        sorted_timings = sorted(step_timings.items(), key=lambda x: x[1], reverse=True)
        total_step_time = sum(step_timings.values())
        for step_name, elapsed_time in sorted_timings:
            percentage = (elapsed_time / total_step_time * 100) if total_step_time > 0 else 0
            display_name = step_name.replace("_", " ").title()
            print(f"  {display_name:35s}: {elapsed_time:8.3f} seconds ({percentage:5.1f}%)")
        print(f"\n  {'Total (all steps)':35s}: {total_step_time:8.3f} seconds")
        print("=" * 70)
    else:
        print("✗ Warning: visualization_dump does not contain scales/rotations.")
        print("  Cannot export to PLY without this information.")
        print("  This may happen if the encoder config has certain settings.")
        print(f"  Available keys in visualization_dump: {list(visualization_dump.keys())}")
        print("\n  Note: The visualization_dump should be populated by the encoder.")
        print("  If this is missing, check that the encoder is configured correctly.")

    print("\n" + "=" * 70)
    print("Timing Summary")
    print("=" * 70)
    print(f"  Encoder inference:")
    print(f"    Start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(encoder_start_wall_time))}")
    print(f"    End: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(encoder_end_wall_time))}")
    print(f"    Elapsed: {encoder_elapsed:.3f} seconds ({encoder_elapsed/60:.2f} minutes)")
    if ply_export_elapsed is not None and ply_export_end_wall_time is not None:
        print(f"  PLY export:")
        print(f"    Start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ply_export_start_wall_time))}")
        print(f"    End: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ply_export_end_wall_time))}")
        print(f"    Elapsed: {ply_export_elapsed:.3f} seconds ({ply_export_elapsed/60:.2f} minutes)")
        total_elapsed = encoder_elapsed + ply_export_elapsed
        print(f"  Total time: {total_elapsed:.3f} seconds ({total_elapsed/60:.2f} minutes)")
    else:
        print("  PLY export: Not completed (missing visualization data)")
        print(f"  Total time (encoder only): {encoder_elapsed:.3f} seconds ({encoder_elapsed/60:.2f} minutes)")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return {
        "result": result,
        "ply_path": ply_path if ply_export_elapsed is not None else None,
        "visualization_dump": visualization_dump,
        "encoder_total_time": total_encoder_elapsed,
        "encoder_average_time": avg_encoder_elapsed,
        "ply_export_time": ply_export_elapsed,
        "device": device,
        "num_runs": num_runs,
        "camera_distances": camera_distances,
    }


def main() -> None:
    setup_result = setup()
    run_encoder(setup_result)


if __name__ == "__main__":
    main()
