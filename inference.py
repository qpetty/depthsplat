"""
Run encoder using direct config loading (no Hydra) and export to PLY.

Image resolution: 960x512 (images are automatically resized to this size)
Camera intrinsics and extrinsics are loaded from metadata files (*_metadata.json)
"""

# ============================================================================
# Configuration - Modify these paths as needed
# ============================================================================

CHECKPOINT_PATH_BASE = "pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"  # Set to None for random init
CHECKPOINT_PATH_SMALL = "pretrained/depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth"
CHECKPOINT_PATH = CHECKPOINT_PATH_SMALL
CONFIG_ROOT = "config"  # Path to config directory
OUTPUT_DIR = "run-output"

# Input image base path (directory containing images and metadata files)
# Images and metadata files (*_metadata.json) should be in this directory
#IMAGE_BASE_PATH = "/Users/quinton/Desktop/hillman_mov_horizontal"  # Base directory for images
IMAGE_BASE_PATH = "/Users/quinton/repos/Image_sender/received_images"

# Encoder config overrides (set to None to use YAML defaults)
ENCODER_OVERRIDES_BASE = {
    "num_scales": 2,
    "upsample_factor": 4,  # 8 for 448x768 resolution (else branch in DPTHead)
    "lowest_feature_resolution": 8,
    "monodepth_vit_type": "vitb",
    "gaussian_adapter": {
        "gaussian_scale_max": 0.1
    }
}
ENCODER_OVERRIDES_SMALL = {
    "num_scales": 1,
    "upsample_factor": 8,  # 8 for 448x768 resolution (else branch in DPTHead)
    "lowest_feature_resolution": 8,
    "monodepth_vit_type": "vits",
    "gaussian_adapter": {
        "gaussian_scale_max": 0.1
    }
}

ENCODER_OVERRIDES = ENCODER_OVERRIDES_SMALL

# Toggle detailed validation and diagnostics during PLY export.
# Leave disabled for fastest export.
PLY_EXPORT_VALIDATION = False

# Torch compile / encoder benchmarking options
# Set to True to compile the encoder with torch.compile for faster repeated inference.
ENABLE_TORCH_COMPILE = False

# Number of times to run the encoder for timing/benchmarking.
# Set >1 to measure average runtime; the final run's output is used for export.
NUM_ENCODER_RUNS = 1

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
import torch._dynamo  # For config
import copy
import torch.nn as nn

# Enable scalar capture for data-dependent ops like allclose
torch._dynamo.config.capture_scalar_outputs = True

from ast import Set
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf
from src.config import load_typed_config
from src.model.encoder import EncoderDepthSplatCfg, get_encoder
from src.model.types import Gaussians as GaussiansOut
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
import onnxruntime
import coremltools as ct
from torch.export import export


@dataclass
class SetupResult:
    encoder: torch.nn.Module
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


def load_intrinsics_extrinsics_from_manifest_entries(
    image_entries: List[Tuple[Union[str, Path], Union[str, Path]]],
) -> Tuple[Dict[str, List[float]], Dict[str, np.ndarray]]:
    """
    Load intrinsics and extrinsics from explicit metadata/image path pairs.

    Args:
        image_entries: List of (metadata_path, image_path) tuples.

    Returns:
        Tuple of dictionaries keyed by image filename.
    """
    if not image_entries:
        raise ValueError("image_manifest must include at least one entry.")

    intrinsics_dict: Dict[str, List[float]] = {}
    extrinsics_dict: Dict[str, np.ndarray] = {}

    for metadata_path, image_path in image_entries:
        metadata_path = Path(metadata_path)
        image_path = Path(image_path)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        image_filename = image_path.name

        if image_filename in intrinsics_dict:
            raise ValueError(f"Duplicate image entry detected for {image_filename}")

        intrinsics_flat = metadata.get("intrinsics", [])
        if len(intrinsics_flat) != 9:
            raise ValueError(
                f"Invalid intrinsics format in {metadata_path.name}, expected 9 elements, got {len(intrinsics_flat)}"
            )

        fx = intrinsics_flat[0]
        fy = intrinsics_flat[4]
        cx = intrinsics_flat[2]
        cy = intrinsics_flat[5]
        intrinsics_dict[image_filename] = [fx, fy, cx, cy]

        extrinsics_flat = metadata.get("extrinsics", [])
        if len(extrinsics_flat) != 16:
            raise ValueError(
                f"Invalid extrinsics format in {metadata_path.name}, expected 16 elements, got {len(extrinsics_flat)}"
            )

        extrinsics_matrix = np.array(extrinsics_flat, dtype=np.float32).reshape(4, 4)
        extrinsics_dict[image_filename] = extrinsics_matrix

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

onnx_model_name = "depthsplat_encoder.onnx"
generate_onnx = False
run_onnx = False
generate_coreml = False
run_coreml = True
coreml_minimal_outputs = True  # True => Gaussians + viz scales/rotations (skip depths)
coreml_use_fp16_input = True   # Set True to use FP16 input (experimental, may reduce quality)
ort_session = None
coreml_model = None
coreml_output_mapping = None  # Mapping from expected output names to actual CoreML output names


def setup_encoder(checkpoint_path: Optional[Union[str, Path]] = CHECKPOINT_PATH,
    config_root: Union[str, Path] = CONFIG_ROOT,
    encoder_overrides: Optional[Dict[str, Any]] = ENCODER_OVERRIDES,
    enable_torch_compile: Optional[bool] = ENABLE_TORCH_COMPILE) -> SetupResult:

    # Device selection: prefer CUDA, then MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    global ort_session
    if run_onnx:
        # Create an ONNX Runtime session; on macOS this will usually be CPUExecutionProvider
        ort_session = onnxruntime.InferenceSession(
            onnx_model_name,
            providers=[('CoreMLExecutionProvider', {
                        "ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU", 
                        "RequireStaticInputShapes": "0", "EnableOnSubgraphs": "0", "ModelCacheDirectory": "onnx_cache",
                    })],
        )
        print("ONNX Runtime InferenceSession created")

    global coreml_model
    if run_coreml:
        model_path = "depthsplat.mlpackage"
        
        # Try different compute units for optimal performance
        # ALL = CPU+GPU+ANE (Neural Engine), CPU_AND_GPU = CPU+GPU only
        # For some models, GPU-only is faster than including Neural Engine
        compute_unit = ct.ComputeUnit.CPU_AND_GPU  # Try this first, change to ALL if slower
        compute_unit_name = "CPU_AND_GPU"
        
        coreml_model = ct.models.MLModel(model_path, compute_units=compute_unit)
        print(f"Loaded CoreML model from {model_path} with compute units: {compute_unit_name}")
        
        # Load output name mapping if it exists
        mapping_path = "depthsplat_output_mapping.json"
        import json
        import os
        global coreml_output_mapping
        coreml_output_mapping = None
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                coreml_output_mapping = json.load(f)
            print(f"  Loaded output name mapping from {mapping_path}")
        
        # Check model metadata
        spec = coreml_model.get_spec()
        print(f"  Model outputs: {len(spec.description.output)} tensors")
        for output in spec.description.output:
            print(f"    - {output.name}")
        print("  Note: Try switching between CPU_AND_GPU and ALL to find fastest option")
        
        # Warmup run to trigger CoreML compilation/optimization
        # Do 2 warmup runs to ensure compilation is fully cached
        print("Running CoreML warmup (2 iterations)...")
        dummy_input = {}
        for input_desc in coreml_model.get_spec().description.input:
            input_name = input_desc.name
            input_shape = tuple(input_desc.type.multiArrayType.shape)
            dummy_input[input_name] = np.zeros(input_shape, dtype=np.float32)
            
        _ = coreml_model.predict(dummy_input)  # First run: compilation
        _ = coreml_model.predict(dummy_input)  # Second run: cache verification
        print("CoreML warmup complete")

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

    checkpoint_path = Path(checkpoint_path)
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
    return SetupResult(encoder=encoder, device=device)


def run_encoder(
    setup_result: SetupResult,
    images: List[str],
    num_runs: Optional[int] = NUM_ENCODER_RUNS,
    output_dir: Optional[Union[str, Path]] = OUTPUT_DIR,
    visualization_dump: Optional[Dict[str, Any]] = {},
    ply_export_validation: Optional[bool] = PLY_EXPORT_VALIDATION,
) -> Dict[str, Any]:

    device = setup_result.device
    output_dir = Path(output_dir)

    resolved_manifest: Optional[List[Tuple[Path, Path]]] = []
    for image_entry in images:
        image_path = Path(image_entry)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image_stem = image_path.stem
        metadata_name = f"{image_stem}_metadata.json"
        metadata_path = image_path.parent / metadata_name

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        metadata_path = metadata_path.resolve()
        image_path = image_path.resolve()

        resolved_manifest.append((metadata_path, image_path))

    manifest_dirs = {metadata_path.parent for metadata_path, _ in resolved_manifest}
    if len(manifest_dirs) != 1:
        raise ValueError("All metadata files in image_manifest must share the same parent directory.")

    manifest_base = manifest_dirs.pop()
    
    # Load intrinsics and extrinsics from JSON metadata files
    print("\n" + "="*70)
    print("Loading Camera Metadata")
    print("="*70)
    
    if resolved_manifest is not None:
        intrinsics_from_metadata, extrinsics_from_metadata = load_intrinsics_extrinsics_from_manifest_entries(
            resolved_manifest
        )
    else:
        intrinsics_from_metadata, extrinsics_from_metadata = load_intrinsics_extrinsics_from_metadata(image_base_path)
    
    if intrinsics_from_metadata is None or extrinsics_from_metadata is None:
        raise FileNotFoundError(
                f"No metadata files found. "
                f"Please ensure metadata files (e.g., '*_metadata.json') are present."
            )
    
    print(f"  Found {len(extrinsics_from_metadata)} metadata file(s)")
    print(f"  Loaded extrinsics for {len(extrinsics_from_metadata)} image(s)")
    
    # Store intrinsics and extrinsics dictionaries for per-image use
    # We'll use per-image intrinsics when setting up the intrinsics tensor
    intrinsics_loaded_dict = intrinsics_from_metadata
    extrinsics_loaded = extrinsics_from_metadata
    
    # Check if all images have the same intrinsics (for informational purposes)
    intrinsics_values = list(intrinsics_from_metadata.values())
    if len(set(tuple(v) for v in intrinsics_values)) == 1:
        # All images have the same intrinsics
        intrinsics_sample = intrinsics_values[0]
        print(f"  All images use the same intrinsics: fx={intrinsics_sample[0]:.2f}, fy={intrinsics_sample[1]:.2f}, "
              f"cx={intrinsics_sample[2]:.2f}, cy={intrinsics_sample[3]:.2f}")
    else:
        # Different intrinsics per image
        print(f"  Images have different intrinsics (will use per-image values):")
        for img_name, intrinsics_val in intrinsics_from_metadata.items():
            print(f"    {img_name}: fx={intrinsics_val[0]:.2f}, fy={intrinsics_val[1]:.2f}, "
                  f"cx={intrinsics_val[2]:.2f}, cy={intrinsics_val[3]:.2f}")


    # Determine number of views and image paths from loaded extrinsics
    batch_size = 1
    if isinstance(extrinsics_loaded, dict):
        if resolved_manifest is not None:
            image_filenames = [image_path.name for _, image_path in resolved_manifest]
            num_views = len(image_filenames)
            image_paths = [str(image_path) for _, image_path in resolved_manifest]
        else:
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

    if image_paths:
        print(f"  Encoder will use {len(image_paths)} image(s):")
        for idx, (filename, path) in enumerate(zip(image_filenames, image_paths), start=1):
            print(f"    [{idx:02d}] {filename} -> {path}")
    
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
    
    # Track original dimensions per image for proper intrinsics scaling
    # Note: We assume all images are resized to the same target size, but they may have different original sizes
    original_dimensions_per_image = {}
    if image_paths and len(image_paths) > 0:
        for img_path, img_filename in zip(image_paths, image_filenames):
            try:
                orig_img = load_image(img_path)
                orig_h, orig_w = orig_img.shape[1], orig_img.shape[2]
                original_dimensions_per_image[img_filename] = (orig_h, orig_w)
            except Exception as e:
                print(f"    Warning: Could not load {img_filename} to get original dimensions: {e}")
                # Fallback: use the first image's dimensions or target dimensions
                if original_dimensions is not None:
                    original_dimensions_per_image[img_filename] = original_dimensions
                else:
                    original_dimensions_per_image[img_filename] = (height, width)
    
    # Initialize intrinsics tensor
    intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
    
    # Use per-image intrinsics from metadata
    print("  Using per-image intrinsics from metadata files")
    
    # Verify all image filenames have intrinsics
    missing_intrinsics = [fname for fname in image_filenames if fname not in intrinsics_loaded_dict]
    if missing_intrinsics:
        raise ValueError(f"Intrinsics dictionary is missing entries for: {missing_intrinsics}")
    
    # Set intrinsics for each view based on the corresponding image filename
    for view_idx, img_filename in enumerate(image_filenames):
        intrinsics_raw = intrinsics_loaded_dict[img_filename]
        
        if len(intrinsics_raw) != 4:
            raise ValueError(f"Intrinsics for {img_filename} must contain exactly 4 values [fx, fy, cx, cy], got {len(intrinsics_raw)}")
        
        fx, fy, cx, cy = intrinsics_raw
        
        # Adjust intrinsics if this image was resized
        if img_filename in original_dimensions_per_image:
            orig_h, orig_w = original_dimensions_per_image[img_filename]
            if orig_w != width or orig_h != height:
                scale_w = width / orig_w
                scale_h = height / orig_h
                fx = fx * scale_w
                fy = fy * scale_h
                cx = cx * scale_w
                cy = cy * scale_h
        
        # Set normalized intrinsics for this view
        intrinsics[0, view_idx, 0, 0] = fx / width   # fx normalized
        intrinsics[0, view_idx, 1, 1] = fy / height  # fy normalized
        intrinsics[0, view_idx, 0, 2] = cx / width   # cx normalized
        intrinsics[0, view_idx, 1, 2] = cy / height  # cy normalized
        
        # Print intrinsics for this view
        print(f"\n  View {view_idx} ({img_filename}):")
        print(f"    Intrinsics (pixels): fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        print(f"    Normalized: fx={fx/width:.6f}, fy={fy/height:.6f}, cx={cx/width:.6f}, cy={cy/height:.6f}")
        
        # Validate intrinsics values for this view
        cx_expected = width / 2.0
        cy_expected = height / 2.0
        cx_offset = abs(cx - cx_expected) / width if width > 0 else 0
        cy_offset = abs(cy - cy_expected) / height if height > 0 else 0
        
        if cx_offset > 0.1:
            print(f"      WARNING: cx is {cx_offset*100:.1f}% off from center (expected ~{cx_expected:.1f})")
        if cy_offset > 0.1:
            print(f"      WARNING: cy is {cy_offset*100:.1f}% off from center (expected ~{cy_expected:.1f})")
        
        # Check focal length ratio
        if abs(fx - fy) / max(fx, fy) > 0.1:
            print(f"      WARNING: fx and fy differ by {(abs(fx-fy)/max(fx,fy)*100):.1f}% (may indicate distortion)")
    
    # Compute and validate Field of View for all views
    print(f"\n  Field of View (FOV) for all views:")
    for view_idx, img_filename in enumerate(image_filenames):
        fov = get_fov(intrinsics[0, view_idx:view_idx+1])
        fov_deg = fov * 180 / math.pi
        print(f"    View {view_idx} ({img_filename}):")
        print(f"      Horizontal FOV: {fov_deg[0, 0]:.2f}°")
        print(f"      Vertical FOV: {fov_deg[0, 1]:.2f}°")
        
        # Check if FOV is reasonable (typical range: 30-120 degrees)
        if fov_deg[0, 0] < 20 or fov_deg[0, 0] > 150:
            print(f"      WARNING: Horizontal FOV ({fov_deg[0, 0]:.2f}°) is outside typical range (20-150°)")
        if fov_deg[0, 1] < 20 or fov_deg[0, 1] > 150:
            print(f"      WARNING: Vertical FOV ({fov_deg[0, 1]:.2f}°) is outside typical range (20-150°)")
    
    # Print full intrinsic matrices for verification
    print(f"\n  Full intrinsic matrices (3x3) for all views:")
    for view_idx, img_filename in enumerate(image_filenames):
        K = intrinsics[0, view_idx].numpy()
        print(f"    View {view_idx} ({img_filename}):")
        for i in range(3):
            print(f"      [{K[i,0]:8.6f}, {K[i,1]:8.6f}, {K[i,2]:8.6f}]")

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
    
    # Check FOV for all views
    for view_idx, img_filename in enumerate(image_filenames):
        fov_view = get_fov(intrinsics[0, view_idx:view_idx+1])
        fov_deg_view = fov_view * 180 / math.pi
        fov_h = fov_deg_view[0, 0].item()
        if fov_h < 30 or fov_h > 120:
            issues.append(f"View {view_idx} ({img_filename}) FOV ({fov_h:.1f}°) is outside typical range - may indicate incorrect intrinsics")
        elif fov_h < 20:
            issues.append(f"View {view_idx} ({img_filename}) FOV ({fov_h:.1f}°) is very narrow - may indicate telephoto lens or incorrect intrinsics")
    
    if issues:
        print("  Potential issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("  No obvious issues detected.")
    
    print("="*70)

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
    output_dir_path = Path(output_dir) if output_dir is not None else setup_result.output_dir
    num_runs = num_runs if num_runs is not None else setup_result.num_encoder_runs
    if num_runs <= 0:
        raise ValueError("num_runs must be a positive integer")

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
        
        if generate_onnx or run_onnx or generate_coreml or run_coreml:
            class ExportableEncoder(nn.Module):
                def __init__(self, encoder: nn.Module, global_step: int = 0):
                    super().__init__()
                    self.encoder = encoder
                    self.global_step = global_step  # Fixed value; adjust if needed

                def forward(self, context: Dict, visualization_dump: Optional[dict] = None):
                    # Always provide a dict so the underlying encoder populates visualization data.
                    if visualization_dump is None:
                        visualization_dump = {}

                    # Call original forward with fixed args for inference/export
                    raw_output = self.encoder(
                        context=context,
                        global_step=self.global_step,
                        deterministic=True,
                        visualization_dump=visualization_dump,
                        scene_names=None,
                    )

                    # Flatten to dict of tensors only
                    flat_output: Dict[str, torch.Tensor] = {}
                    if isinstance(raw_output, dict):
                        for key, value in raw_output.items():
                            # Special handling for visualization_dump dict
                            if key == "visualization_dump" and isinstance(value, dict):
                                depth_viz = value.get("depth", None)
                                scales_viz = value.get("scales", None)
                                rotations_viz = value.get("rotations", None)

                                if isinstance(depth_viz, torch.Tensor):
                                    flat_output["visualization_dump_depth"] = depth_viz
                                if isinstance(scales_viz, torch.Tensor):
                                    flat_output["visualization_dump_scales"] = scales_viz
                                if isinstance(rotations_viz, torch.Tensor):
                                    assert rotations_viz.shape[-1] == 4, f"Expected viz rotations last dim=4, got {rotations_viz.shape}"
                                    flat_output["visualization_dump_rotations"] = rotations_viz
                                continue

                            if isinstance(value, torch.Tensor):
                                # e.g., 'depths'
                                flat_output[key] = value
                            else:
                                # Treat as Gaussians-like object
                                means = getattr(value, "means", None)
                                covariances = getattr(value, "covariances", None)
                                opacities = getattr(value, "opacities", None)
                                harmonics = getattr(value, "harmonics", None)

                                if isinstance(means, torch.Tensor):
                                    flat_output[f"{key}_means"] = means
                                if isinstance(covariances, torch.Tensor):
                                    flat_output[f"{key}_covariances"] = covariances
                                if isinstance(opacities, torch.Tensor):
                                    flat_output[f"{key}_opacities"] = opacities
                                if isinstance(harmonics, torch.Tensor):
                                    flat_output[f"{key}_harmonics"] = harmonics
                    else:
                        # Fallback (unlikely)
                        if isinstance(raw_output, torch.Tensor):
                            flat_output = {"raw_output": raw_output}

                    return flat_output  # e.g., {'gaussians_means': tensor, ..., 'depths': tensor, 'visualization_dump_*': tensor}
        
            def prepare_context_for_export(context: Dict) -> Dict:
                export_context = copy.deepcopy(context)  # Deep copy to preserve original
                for key, value in export_context.items():
                    if isinstance(value, torch.Tensor):
                        export_context[key] = value.detach().cpu().clone().requires_grad_(False)
                    # Non-tensors (e.g., strings, lists) are fine if not used in computations;
                    # if forward() accesses them dynamically (e.g., if key in context), it may fail—simplify if needed
                return export_context

            context_for_export = prepare_context_for_export(context)  # Your real context, now export-ready

            if run_coreml:
                # Extract image tensor from context
                image_tensor = context_for_export["image"]
                
                # Detailed timing to identify bottlenecks
                t_start = time.perf_counter()
                
                # Pre-convert to numpy with optimal settings to minimize overhead
                # Use contiguous memory and avoid unnecessary copies
                input_dict = {}
                
                # Helper to convert tensor to numpy
                def to_numpy(tensor):
                    if tensor.is_cuda or (hasattr(tensor, 'is_mps') and tensor.is_mps):
                        numpy_val = tensor.cpu().numpy()
                    else:
                        numpy_val = tensor.numpy()
                    
                    # Convert to FP16 if requested and it's float32
                    if coreml_use_fp16_input and numpy_val.dtype == np.float32:
                        numpy_val = numpy_val.astype(np.float16)
                        
                    if not numpy_val.flags['C_CONTIGUOUS']:
                        numpy_val = np.ascontiguousarray(numpy_val)
                    return numpy_val
                
                input_dict["image"] = to_numpy(image_tensor)
                input_dict["extrinsics"] = to_numpy(context_for_export["extrinsics"])
                input_dict["intrinsics"] = to_numpy(context_for_export["intrinsics"])
                input_dict["near"] = to_numpy(context_for_export["near"])
                input_dict["far"] = to_numpy(context_for_export["far"])
                
                t_converted = time.perf_counter()
                
                # Use the predict method - this is synchronous and optimized
                predictions = coreml_model.predict(input_dict)
                
                t_predicted = time.perf_counter()
                
                print(f"Predictions keys: {list(predictions.keys())}")
                
                # Find the means key (might have suffix)
                means_key = None
                for key in predictions.keys():
                    if 'means' in key.lower():
                        means_key = key
                        break
                
                if means_key:
                    print(f"Main output shape: {predictions[means_key].shape}")
                else:
                    print(f"Main output shape: N/A (gaussians_means not found!)")
                    print(f"Available outputs: {list(predictions.keys())}")
                
                print(f"  Timing breakdown:")
                print(f"    Tensor→NumPy conversion: {(t_converted - t_start) * 1000:.2f}ms")
                print(f"    CoreML inference: {(t_predicted - t_converted) * 1000:.2f}ms")
                print(f"    Total: {(t_predicted - t_start) * 1000:.2f}ms")
                if not coreml_minimal_outputs:
                    print(f"  Note: try minimal_outputs=True for potential 10-15% speedup")
                
                # Process CoreML outputs into expected result format
                # Map output keys using the mapping file if available
                output_keys = list(predictions.keys())
                
                # Helper function to find key, using mapping if available
                global coreml_output_mapping
                def find_key(expected_name):
                    # First try using the mapping file
                    if coreml_output_mapping and 'mapping' in coreml_output_mapping:
                        mapped_name = coreml_output_mapping['mapping'].get(expected_name)
                        if mapped_name and mapped_name in output_keys:
                            return mapped_name
                    
                    # Fallback to substring match
                    for key in output_keys:
                        if expected_name.lower() in key.lower():
                            return key
                    return None
                
                # Extract outputs (order depends on coreml_minimal_outputs setting during export)
                # Minimal order: means, covariances, opacities, harmonics, viz_scales, viz_rotations
                # Full order: means, covariances, opacities, harmonics, viz_depth, viz_scales, viz_rotations, depths
                means_key = find_key('gaussians_means')
                covs_key = find_key('gaussians_covariances')
                opac_key = find_key('gaussians_opacities')
                harm_key = find_key('gaussians_harmonics')
                
                # Build detailed error message if required keys are missing
                missing_keys = []
                if not means_key:
                    missing_keys.append('means')
                if not covs_key:
                    missing_keys.append('covariances')
                if not opac_key:
                    missing_keys.append('opacities')
                if not harm_key:
                    missing_keys.append('harmonics')
                
                if missing_keys:
                    print(f"\n{'='*80}")
                    print(f"ERROR: CoreML model is missing required outputs: {missing_keys}")
                    print(f"Available outputs: {output_keys}")
                    print(f"\nThe CoreML model at 'depthsplat.mlpackage' is incomplete or corrupted.")
                    print(f"\nTo fix this, regenerate the CoreML model by:")
                    print(f"  1. Set 'generate_coreml = True' (line ~651)")
                    print(f"  2. Set 'run_coreml = False' (line ~652)")
                    print(f"  3. Run the script again to export a new model")
                    print(f"  4. Once export is complete, set 'generate_coreml = False' and 'run_coreml = True'")
                    print(f"{'='*80}\n")
                    raise ValueError(f"CoreML output missing required keys: {missing_keys}. Found: {output_keys}")
                
                # Convert numpy arrays to torch tensors
                gauss_means = torch.from_numpy(predictions[means_key])
                gauss_covs = torch.from_numpy(predictions[covs_key])
                gauss_opac = torch.from_numpy(predictions[opac_key])
                gauss_harm = torch.from_numpy(predictions[harm_key])
                
                # Reconstruct Gaussians object
                gaussians = GaussiansOut(
                    means=gauss_means,
                    covariances=gauss_covs,
                    harmonics=gauss_harm,
                    opacities=gauss_opac,
                )
                
                # Build optional depth / visualization outputs
                depths_key = find_key('depths')
                vd_depth_key = find_key('visualization_dump_depth')
                vd_scales_key = find_key('visualization_dump_scales')
                vd_rotations_key = find_key('visualization_dump_rotations')
                
                depths = torch.from_numpy(predictions[depths_key]) if depths_key else None
                
                visualization_dump_dict: Optional[Dict[str, torch.Tensor]] = None
                viz_entries: Dict[str, torch.Tensor] = {}
                if vd_depth_key:
                    viz_entries["depth"] = torch.from_numpy(predictions[vd_depth_key])
                if vd_scales_key:
                    viz_entries["scales"] = torch.from_numpy(predictions[vd_scales_key])
                if vd_rotations_key:
                    viz_entries["rotations"] = torch.from_numpy(predictions[vd_rotations_key])
                if viz_entries:
                    visualization_dump_dict = viz_entries
                
                result = {"gaussians": gaussians, "depths": depths, "visualization_dump": visualization_dump_dict}
                    
                    
            if generate_coreml:
                # Use the ExportableEncoder wrapper so that we provide a fixed global_step and
                # a clean tensor-only output dict, then wrap again to expose a simple
                # tensor-in / tuple-of-tensors-out interface for CoreML.
                encoder.to("cpu")
                encoder.eval()

                export_model = ExportableEncoder(encoder, global_step=0)
                export_model.eval()

                class CoreMLEncoderWrapper(nn.Module):
                    def __init__(self, export_model: nn.Module, static_context: Dict[str, torch.Tensor]):
                        super().__init__()
                        self.export_model = export_model

                        # Register static context entries (e.g., intrinsics, poses) as buffers.
                        # These will be treated as constants by CoreML, unless listed in dynamic_keys.
                        self._buffer_keys: List[str] = []
                        self.dynamic_keys = ["extrinsics", "intrinsics", "near", "far"]

                        for key, value in static_context.items():
                            if key in self.dynamic_keys:
                                continue
                            if isinstance(value, torch.Tensor):
                                # Store as buffer; keys are expected to be simple strings.
                                self.register_buffer(key, value)
                                self._buffer_keys.append(key)

                    def forward(self, image: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor, near: torch.Tensor, far: torch.Tensor):
                        # Reconstruct context dict from buffers, overriding the image with
                        # the runtime-provided tensor.
                        context: Dict[str, torch.Tensor] = {}
                        for key in self._buffer_keys:
                            context[key] = getattr(self, key)
                        
                        context["image"] = image
                        context["extrinsics"] = extrinsics
                        context["intrinsics"] = intrinsics
                        context["near"] = near
                        context["far"] = far

                        flat_output = self.export_model(context=context)

                        # Deterministic tuple of outputs for CoreML; order must be stable.
                        # Visualization dumps are skipped during export, so use get() with defaults
                        if coreml_minimal_outputs:
                            # Minimal outputs - Gaussian params plus viz scales/rotations for fast PLY export
                            vd_scales = flat_output.get("visualization_dump_scales")
                            if vd_scales is None:
                                vd_scales = flat_output["gaussians_means"].clone()

                            vd_rotations = flat_output.get("visualization_dump_rotations")
                            if vd_rotations is None:
                                vd_rotations = flat_output["gaussians_covariances"].clone()

                            return (
                                flat_output["gaussians_means"],
                                flat_output["gaussians_covariances"],
                                flat_output["gaussians_opacities"],
                                flat_output["gaussians_harmonics"],
                                vd_scales,
                                vd_rotations,
                            )
                        else:
                            # Full outputs including visualization and depth
                            # IMPORTANT: Must return unique tensor objects - CoreML doesn't handle duplicates well
                            # Use .clone() to create new tensors if fallback values are needed
                            vd_depth = flat_output.get("visualization_dump_depth")
                            if vd_depth is None:
                                vd_depth = flat_output["depths"].clone()
                            
                            vd_scales = flat_output.get("visualization_dump_scales")
                            if vd_scales is None:
                                # Create a unique tensor instead of reusing gaussians_means
                                vd_scales = flat_output["gaussians_means"].clone()
                            
                            vd_rotations = flat_output.get("visualization_dump_rotations")
                            if vd_rotations is None:
                                # Create a unique tensor instead of reusing gaussians_covariances
                                vd_rotations = flat_output["gaussians_covariances"].clone()
                            
                            return (
                                flat_output["gaussians_means"],
                                flat_output["gaussians_covariances"],
                                flat_output["gaussians_opacities"],
                                flat_output["gaussians_harmonics"],
                                vd_depth,
                                vd_scales,
                                vd_rotations,
                                flat_output["depths"],
                            )

                # Instantiate wrapper with current static context.
                coreml_wrapper = CoreMLEncoderWrapper(export_model, static_context=context_for_export)
                coreml_wrapper.eval()

                # Use the main image tensor to define CoreML input shape and trace to TorchScript.
                image_tensor = context_for_export["image"]
                extrinsics_tensor = context_for_export["extrinsics"]
                intrinsics_tensor = context_for_export["intrinsics"]
                near_tensor = context_for_export["near"]
                far_tensor = context_for_export["far"]
                # print(f"Running CoreML trace")
                # with torch.no_grad():
                #     traced_wrapper = torch.jit.trace(
                #         coreml_wrapper,
                #         (image_tensor,),
                #         strict=False,
                #         check_trace=False,  # Skip trace-vs-runtime sanity check; model has dynamic/non-deterministic behavior
                #     )
                # ExportedProgram instead of TorchScript

                import sys
                from e3nn import o3  # ensure e3nn.o3 is loaded

                # Patch e3nn's rotation helpers to remove data-dependent asserts while
                # preserving their numerical behavior.

                # 1) matrix_to_angles (in e3nn.o3._rotation)
                rotation_module = sys.modules.get("e3nn.o3._rotation")
                if rotation_module is None:
                    import e3nn.o3._rotation as rotation_module
                    sys.modules["e3nn.o3._rotation"] = rotation_module

                def patched_matrix_to_angles(R):
                    # Drop the assert on det(R) == 1; compute angles directly using the
                    # same formulas as the original implementation.
                    alpha = torch.atan2(R[..., 2, 1], R[..., 2, 2])
                    beta = torch.asin(-R[..., 2, 0])
                    gamma = torch.atan2(R[..., 1, 0], R[..., 0, 0])
                    return alpha, beta, gamma

                rotation_module.matrix_to_angles = patched_matrix_to_angles

                # Ensure the alias imported in sh_rotation uses the patched version.
                from src.misc import sh_rotation
                sh_rotation.matrix_to_angles = rotation_module.matrix_to_angles

                # 2) so3_generators (in e3nn.o3._wigner), which is used by wigner_D.
                wigner_module = sys.modules.get("e3nn.o3._wigner")
                if wigner_module is None:
                    import e3nn.o3._wigner as wigner_module
                    sys.modules["e3nn.o3._wigner"] = wigner_module

                def patched_so3_generators(l: int) -> torch.Tensor:
                    # Same computation as the original so3_generators, but without the
                    # data-dependent assert on the imaginary part.
                    X = wigner_module.su2_generators(l)
                    Q = wigner_module.change_basis_real_to_complex(l)
                    X = torch.conj(Q.T) @ X @ Q
                    return torch.real(X)

                wigner_module.so3_generators = patched_so3_generators

                # Set flag to skip export-incompatible operations (like rotate_sh with matrix_exp)
                import src.model.encoder.common.gaussian_adapter as gaussian_adapter_module
                gaussian_adapter_module._SKIP_EXPORT_INCOMPATIBLE_OPS = True
                
                print("\n" + "="*80)
                print("DEBUG: Setting _SKIP_EXPORT_INCOMPATIBLE_OPS flag")
                print(f"  Flag value: {gaussian_adapter_module._SKIP_EXPORT_INCOMPATIBLE_OPS}")
                print(f"  Module ID: {id(gaussian_adapter_module)}")
                print("="*80 + "\n")
                
                try:
                    # Export via torch.export (ExportedProgram) after patching.
                    print("Starting torch.export...")
                    exported = export(coreml_wrapper, (image_tensor, extrinsics_tensor, intrinsics_tensor, near_tensor, far_tensor))
                    print("torch.export completed successfully")
                    # Lower TRAINING dialect ops to ATEN/EDGE as required by coremltools.
                    exported = exported.run_decompositions({})
                    
                    # DEBUG: Check for high-dimensional tensors in the exported graph
                    print("\n" + "="*80)
                    print("DEBUG: Checking exported graph for high-dimensional tensors")
                    print("="*80)
                    for node in exported.graph_module.graph.nodes:
                        if hasattr(node, 'meta') and 'val' in node.meta:
                            val = node.meta['val']
                            if hasattr(val, 'shape') and len(val.shape) > 5:
                                print(f"WARNING: Found {len(val.shape)}D tensor!")
                                print(f"  Node: {node.name}")
                                print(f"  Op: {node.op}")
                                print(f"  Target: {node.target}")
                                print(f"  Shape: {val.shape}")
                                print()
                finally:
                    # Reset flag
                    gaussian_adapter_module._SKIP_EXPORT_INCOMPATIBLE_OPS = False

                # Debug: Find all diag operations in the exported graph
                print("\n" + "="*80)
                print("DEBUGGING: Searching for diag operations in exported graph")
                print("="*80)
                gm = exported.graph_module
                diag_nodes = []
                all_ops = set()
                
                def trace_back(node, depth=0, max_depth=10, visited=None):
                    """Recursively trace back through the graph to find the origin."""
                    if visited is None:
                        visited = set()
                    if depth >= max_depth or node in visited:
                        return
                    visited.add(node)
                    
                    indent = "  " * depth
                    if hasattr(node, 'op') and node.op == "call_function":
                        print(f"{indent}← {node.name}: {node.target}")
                        for arg in node.args:
                            if hasattr(arg, 'name') and hasattr(arg, 'op'):
                                trace_back(arg, depth + 1, max_depth, visited)
                
                for node in gm.graph.nodes:
                    if node.op == "call_function":
                        target_str = str(node.target)
                        all_ops.add(target_str)
                        
                        if "diag" in target_str.lower():
                            diag_nodes.append((node, target_str))
                            print(f"\n{'='*60}")
                            print(f"Found diag operation #{len(diag_nodes)}:")
                            print(f"  Node: {node.name}")
                            print(f"  Target: {target_str}")
                            print(f"  Args: {node.args}")
                            print(f"  Kwargs: {node.kwargs}")
                            print(f"\n  Tracing back (up to 10 levels):")
                            trace_back(node, depth=0, max_depth=10)
                
                print(f"\n{'='*80}")
                print(f"Total diag operations found: {len(diag_nodes)}")
                
                if diag_nodes:
                    print("\n" + "="*80)
                    print("WARNING: Found unsupported diag operations in the graph")
                    print("="*80)
                    
                    # Look for common patterns in the traced operations
                    print("\nSearching for matrix operations (linalg, svd, eig, qr, cholesky, etc.)...")
                    matrix_ops = []
                    for op in sorted(all_ops):
                        if any(keyword in op.lower() for keyword in ['linalg', 'svd', 'eig', 'qr', 'cholesky', 'inv', 'solve', 'det', 'slogdet', 'pinv', 'matrix_exp', 'lu']):
                            matrix_ops.append(op)
                            print(f"  Found: {op}")
                    
                    if not matrix_ops:
                        print("  No obvious matrix decomposition operations found.")
                        print("  The diag operations may be coming from torch.export decompositions.")
                    
                    print("\nERROR: Cannot proceed with CoreML conversion due to unsupported diag operations")
                    print("These typically come from operations like:")
                    print("  - torch.linalg.matrix_exp (in e3nn's wigner_D for spherical harmonics)")
                    print("  - Matrix decompositions (SVD, QR, Cholesky, etc.)")
                    print("\nSuggestion: Check that torch.jit.is_tracing() is being used to skip these operations")
                    
                    # Save the graph for inspection
                    try:
                        graph_code = gm.code
                        with open("exported_graph_debug.py", "w") as f:
                            f.write(graph_code)
                        print("\nGraph code saved to: exported_graph_debug.py")
                    except Exception as e:
                        print(f"\nCouldn't save graph code: {e}")
                    
                    return
                else:
                    print("✓ No diag operations found - graph is compatible with CoreML")
                
                print("="*80 + "\n")

                # Replace bicubic upsampling with bilinear upsampling for CoreML compatibility.
                # CoreML's Torch frontend does not support aten::upsample_bicubic2d, but it does
                # support bilinear/nearest Resize. Here we surgically swap the aten op kind
                # while keeping arguments identical.
                print("Replacing bicubic operations with bilinear for CoreML compatibility...")
                try:
                    # Access the graph module directly from ExportedProgram
                    gm = exported.graph_module
                    modified = False
                    
                    # Check for torch.ops.aten.upsample_bicubic2d operations and corresponding bilinear op.
                    try:
                        bicubic_op = torch.ops.aten.upsample_bicubic2d.vec
                    except AttributeError:
                        bicubic_op = None
                    try:
                        bilinear_op = torch.ops.aten.upsample_bilinear2d.vec
                    except AttributeError:
                        bilinear_op = None
                    
                    for node in list(gm.graph.nodes):
                        if node.op == "call_function":
                            target = node.target
                            target_str = str(target)
                            
                            # Check for upsample_bicubic2d operations.
                            # Can appear as torch.ops.aten.upsample_bicubic2d.vec or similar.
                            is_bicubic = False
                            if bicubic_op is not None and target == bicubic_op:
                                is_bicubic = True
                            elif "upsample_bicubic2d" in target_str:
                                is_bicubic = True
                            
                            if is_bicubic:
                                modified = True
                                print(f"    Found bicubic operation: {target_str}")
                                
                                if bilinear_op is None:
                                    raise RuntimeError(
                                        "aten::upsample_bilinear2d.vec is not available in this PyTorch build"
                                    )

                                # Replace with aten::upsample_bilinear2d.vec using the
                                # exact same args/kwargs as the original bicubic op.
                                with gm.graph.inserting_before(node):
                                    new_node = gm.graph.call_function(
                                        bilinear_op,
                                        args=tuple(node.args),
                                        kwargs=dict(node.kwargs),
                                    )
                                
                                node.replace_all_uses_with(new_node)
                                gm.graph.erase_node(node)
                                print("    Replaced with aten.upsample_bilinear2d.vec")
                    
                    if modified:
                        gm.graph.lint()
                        gm.recompile()
                        print("  ✓ Bicubic operations replaced with bilinear")
                    else:
                        print("  No bicubic operations found")
                except Exception as e:
                    print(f"  Warning: Could not replace bicubic operations: {e}")
                    import traceback
                    traceback.print_exc()
                    print("  Attempting to proceed anyway - CoreML conversion may fail if bicubic ops are present")

                print("Debugging and fixing type mismatches for CoreML FP16 compatibility...")
                # Find and fix operations with mixed types (stack, cat, etc.)
                try:
                    gm = exported.graph_module
                    fixed_ops = 0
                    
                    # Operations that are sensitive to type mismatches
                    type_sensitive_ops = ["stack", "cat", "concat"]
                    
                    for node in list(gm.graph.nodes):
                        if node.op == "call_function":
                            target_str = str(node.target)
                            
                            # Look for type-sensitive operations
                            is_sensitive = any(op in target_str.lower() for op in type_sensitive_ops)
                            
                            if is_sensitive:
                                print(f"  Found operation: {node.name} -> {target_str}")
                                
                                # Try to fix type mismatches by casting all inputs to float32
                                if node.args and len(node.args) > 0:
                                    tensors_arg = node.args[0]
                                    if isinstance(tensors_arg, (list, tuple)):
                                        print(f"    Operation has {len(tensors_arg)} inputs, ensuring type consistency...")
                                        
                                        # Create cast operations to ensure all are float32
                                        cast_tensors = []
                                        for i, tensor in enumerate(tensors_arg):
                                            if hasattr(tensor, 'op') and tensor.op == 'placeholder':
                                                # Don't cast placeholders
                                                cast_tensors.append(tensor)
                                            else:
                                                # Insert a cast to float32 before the operation
                                                with gm.graph.inserting_before(node):
                                                    cast_node = gm.graph.call_function(
                                                        torch.ops.aten._to_copy.default,
                                                        args=(tensor,),
                                                        kwargs={"dtype": torch.float32}
                                                    )
                                                    cast_tensors.append(cast_node)
                                        
                                        # Replace the operation's input with casted tensors
                                        new_args = (cast_tensors,) + node.args[1:]
                                        node.args = new_args
                                        fixed_ops += 1
                                        print(f"    ✓ Cast all inputs to float32")
                            
                            # Also look for division operations that might create doubles
                            elif "div" in target_str.lower() or "truediv" in target_str.lower():
                                # Ensure division results are explicitly float32
                                for user in list(node.users.keys()):
                                    # Insert cast after division
                                    with gm.graph.inserting_after(node):
                                        cast_node = gm.graph.call_function(
                                            torch.ops.aten._to_copy.default,
                                            args=(node,),
                                            kwargs={"dtype": torch.float32}
                                        )
                                        user.replace_input_with(node, cast_node)
                                        fixed_ops += 1
                                        break  # Only need to insert once
                    
                    if fixed_ops > 0:
                        gm.graph.lint()
                        gm.recompile()
                        print(f"  ✓ Fixed {fixed_ops} operation(s) with type casting")
                        
                        # Save the fixed graph
                        try:
                            graph_code = gm.code
                            with open("exported_graph_fixed.py", "w") as f:
                                f.write(graph_code)
                            print("  Fixed graph saved to: exported_graph_fixed.py")
                        except Exception as e:
                            print(f"  Couldn't save fixed graph: {e}")
                    else:
                        print("  No operations needed type fixing")
                        
                except Exception as e:
                    print(f"  Warning: Could not fix type mismatches: {e}")
                    import traceback
                    traceback.print_exc()
                    print("  Attempting to proceed anyway")
                
                print("\nConverting to CoreML with optimizations")
                
                # Now that types are fixed, try FP16 first for maximum performance
                fp16_success = False
                
                # Define outputs based on minimal_outputs flag
                if coreml_minimal_outputs:
                    print("  Using minimal outputs (Gaussians + viz scales/rotations)")
                    output_spec = [
                        ct.TensorType(name="gaussians_means"),
                        ct.TensorType(name="gaussians_covariances"),
                        ct.TensorType(name="gaussians_opacities"),
                        ct.TensorType(name="gaussians_harmonics"),
                        ct.TensorType(name="visualization_dump_scales"),
                        ct.TensorType(name="visualization_dump_rotations"),
                    ]
                else:
                    print("  Using full outputs (8 tensors) including visualization")
                    output_spec = [
                        ct.TensorType(name="gaussians_means"),
                        ct.TensorType(name="gaussians_covariances"),
                        ct.TensorType(name="gaussians_opacities"),
                        ct.TensorType(name="gaussians_harmonics"),
                        ct.TensorType(name="visualization_dump_depth"),
                        ct.TensorType(name="visualization_dump_scales"),
                        ct.TensorType(name="visualization_dump_rotations"),
                        ct.TensorType(name="depths"),
                    ]
                
                try:
                    print("  Attempting FP16 conversion (2-4x faster than FP32)...")
                    import logging
                    import sys
                    
                    # Capture CoreML conversion warnings/errors
                    coreml_logger = logging.getLogger('coremltools')
                    coreml_logger.setLevel(logging.DEBUG)
                    handler = logging.StreamHandler(sys.stdout)
                    handler.setLevel(logging.DEBUG)
                    formatter = logging.Formatter('    [CoreML] %(levelname)s: %(message)s')
                    handler.setFormatter(formatter)
                    coreml_logger.addHandler(handler)
                    
                    model = ct.convert(
                        exported,
                        source="pytorch",
                        convert_to="mlprogram",
                        inputs=[
                            ct.TensorType(name="image", shape=image_tensor.shape),
                            ct.TensorType(name="extrinsics", shape=extrinsics_tensor.shape),
                            ct.TensorType(name="intrinsics", shape=intrinsics_tensor.shape),
                            ct.TensorType(name="near", shape=near_tensor.shape),
                            ct.TensorType(name="far", shape=far_tensor.shape),
                        ],
                        outputs=output_spec,
                        compute_precision=ct.precision.FLOAT16,
                        compute_units=ct.ComputeUnit.CPU_AND_GPU,  # Use fastest config
                        minimum_deployment_target=ct.target.macOS13,
                    )
                    print("  ✓ FP16 conversion successful!")
                    fp16_success = True
                except Exception as e:
                    print(f"  ⚠ FP16 conversion still failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print("  Falling back to FP32 with post-conversion FP16 optimization...")
                
                # Fallback to FP32 if FP16 failed
                if not fp16_success:
                    print("  Attempting FP32 conversion...")
                    import logging
                    import sys
                    
                    # Ensure logging is enabled for FP32 conversion too
                    coreml_logger = logging.getLogger('coremltools')
                    coreml_logger.setLevel(logging.DEBUG)
                    if not coreml_logger.handlers:
                        handler = logging.StreamHandler(sys.stdout)
                        handler.setLevel(logging.DEBUG)
                        formatter = logging.Formatter('    [CoreML] %(levelname)s: %(message)s')
                        handler.setFormatter(formatter)
                        coreml_logger.addHandler(handler)
                    
                    model = ct.convert(
                        exported,
                        source="pytorch",
                        convert_to="mlprogram",
                        inputs=[
                            ct.TensorType(name="image", shape=image_tensor.shape),
                            ct.TensorType(name="extrinsics", shape=extrinsics_tensor.shape),
                            ct.TensorType(name="intrinsics", shape=intrinsics_tensor.shape),
                            ct.TensorType(name="near", shape=near_tensor.shape),
                            ct.TensorType(name="far", shape=far_tensor.shape),
                        ],
                        outputs=output_spec,
                        compute_units=ct.ComputeUnit.CPU_AND_GPU,  # Use fastest config
                        minimum_deployment_target=ct.target.macOS13,
                    )
                    print("  ✓ FP32 conversion successful")
                    
                    # Try post-conversion FP16 optimization (more aggressive)
                    try:
                        print("  Attempting aggressive FP16 casting via compression...")
                        import coremltools.optimize.coreml as cto
                        
                        # Use the newer compression API for better FP16 support
                        op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=16)
                        config = cto.OptimizationConfig(global_config=op_config)
                        
                        # Alternative: try direct FP16 casting on the model
                        # This converts compute ops while keeping metadata in FP32
                        from coremltools.models.neural_network.quantization_utils import quantize_weights
                        model = quantize_weights(model, nbits=16, quantization_mode="linear")
                        print("  ✓ Successfully cast operations to FP16")
                        fp16_success = True
                    except Exception as e:
                        print(f"  ℹ Post-conversion FP16 optimization failed: {e}")
                        print("  Using FP32 model (still GPU-accelerated, ~2s inference)")
                
                if fp16_success:
                    print("  🚀 Final model uses FP16 precision for maximum speed")
                
                # FIX: Explicitly rename outputs to match our expected names
                # CoreML's optimization passes can mix up output names, so we force them here
                print("\n  Fixing output names after conversion...")
                spec = model.get_spec()
                actual_output_names = [output.name for output in spec.description.output]
                expected_output_names = [o.name for o in output_spec]
                
                print(f"    Before fix: {actual_output_names}")
                print(f"    Expected: {expected_output_names}")
                
                # WORKAROUND: CoreML's FP16 conversion messes up output names
                # Instead of trying to fix the internal program, we'll:
                # 1. Save a mapping file that maps actual output names to expected names
                # 2. Use this mapping at runtime to correctly identify outputs
                
                # Create output name mapping
                output_name_mapping = {}
                for i in range(min(len(actual_output_names), len(expected_output_names))):
                    actual_name = actual_output_names[i]
                    expected_name = expected_output_names[i]
                    if actual_name != expected_name:
                        output_name_mapping[expected_name] = actual_name
                        print(f"    Mapping: {expected_name} <- {actual_name}")
                
                # Save the mapping to a JSON file alongside the model
                import json
                mapping_path = "depthsplat_output_mapping.json"
                with open(mapping_path, 'w') as f:
                    json.dump({
                        'expected_outputs': expected_output_names,
                        'actual_outputs': actual_output_names,
                        'mapping': output_name_mapping
                    }, f, indent=2)
                print(f"    ✓ Saved output name mapping to {mapping_path}")
                print(f"    NOTE: Runtime code will use this mapping to correctly identify outputs")
                
                # Verify the model outputs before saving
                print("\n" + "="*80)
                print("Verifying CoreML model outputs...")
                spec = model.get_spec()
                actual_outputs = [output.name for output in spec.description.output]
                print(f"Expected outputs: {[o.name for o in output_spec]}")
                print(f"Actual outputs: {actual_outputs}")
                
                missing_outputs = [o.name for o in output_spec if o.name not in actual_outputs]
                if missing_outputs:
                    print(f"\n⚠ WARNING: The following outputs are MISSING from the converted model:")
                    for name in missing_outputs:
                        print(f"  - {name}")
                    print("\nThis indicates an issue during CoreML conversion.")
                    print("Possible causes:")
                    print("  1. The tensor has unsupported operations in its computation path")
                    print("  2. The tensor shape is incompatible with CoreML")
                    print("  3. The tensor dtype is problematic")
                    print("\nInvestigating the exported graph...")
                    
                    # Check the torch.export graph to see if those outputs exist there
                    gm = exported.graph_module
                    output_nodes = [node for node in gm.graph.nodes if node.op == "output"]
                    if output_nodes:
                        output_node = output_nodes[0]
                        print(f"\nTorch export output node args: {output_node.args}")
                        if output_node.args and len(output_node.args) > 0:
                            output_tuple = output_node.args[0]
                            print(f"Number of outputs in torch.export: {len(output_tuple) if isinstance(output_tuple, (list, tuple)) else 'N/A'}")
                            if isinstance(output_tuple, (list, tuple)):
                                # Check for duplicate outputs
                                output_ids = [id(out) for out in output_tuple]
                                unique_ids = set(output_ids)
                                if len(unique_ids) < len(output_tuple):
                                    print(f"\n⚠ WARNING: Found duplicate outputs in the graph!")
                                    print(f"  Total outputs: {len(output_tuple)}, Unique outputs: {len(unique_ids)}")
                                    print(f"  This may cause CoreML conversion issues.")
                                    print(f"\n  Duplicate analysis:")
                                    for i, out in enumerate(output_tuple):
                                        duplicates = [j for j, o in enumerate(output_tuple) if id(o) == id(out) and j != i]
                                        if duplicates:
                                            print(f"    Output {i} ({out.name}): also appears at positions {duplicates}")
                                
                                print(f"\n  Detailed output information:")
                                for i, out in enumerate(output_tuple):
                                    expected_name = output_spec[i].name if i < len(output_spec) else "N/A"
                                    # Get metadata if available
                                    meta_str = ""
                                    if hasattr(out, 'meta') and 'val' in out.meta:
                                        val = out.meta['val']
                                        if hasattr(val, 'shape') and hasattr(val, 'dtype'):
                                            meta_str = f" - shape: {val.shape}, dtype: {val.dtype}"
                                    print(f"    Output {i}: {out.name}{meta_str}")
                                    print(f"      -> Expected name: {expected_name}")
                                    
                    print(f"\n  Possible issue: CoreML may not handle duplicate tensor outputs correctly.")
                    print(f"  The model forward() returns some tensors multiple times (see fallback logic lines 1629-1631).")
                    print(f"  This could cause CoreML to drop outputs during conversion.")
                    
                    print("\n" + "="*80)
                    # Don't raise error - we saved the mapping file
                    print(f"⚠ WARNING: Output names don't match, but we have {len(actual_outputs)} outputs")
                    print(f"  The runtime code will use depthsplat_output_mapping.json to map outputs correctly")
                
                # Check if we have the right number of outputs
                if len(actual_outputs) != len(expected_output_names):
                    print(f"\n⚠ CRITICAL: Output count mismatch!")
                    print(f"  Expected: {len(expected_output_names)} outputs")
                    print(f"  Actual: {len(actual_outputs)} outputs")
                    raise ValueError(f"CoreML conversion produced wrong number of outputs: expected {len(expected_output_names)}, got {len(actual_outputs)}")
                else:
                    print(f"✓ Correct number of outputs: {len(actual_outputs)}")
                print("="*80 + "\n")
                
                model.save("depthsplat.mlpackage")
                print(f"Encoder exported successfully to depthsplat.mlpackage")
                return

            if generate_onnx:
                encoder.to('cpu')
                encoder.eval()
                

                export_model = ExportableEncoder(encoder, global_step=0)

                import sys
                # import torch.onnx
                from e3nn import o3  # Trigger import if not already

                # Find the exact loaded module for _rotation
                rotation_module = sys.modules.get('e3nn.o3._rotation')
                if rotation_module is None:
                    # Fallback: Import directly to load it
                    import e3nn.o3._rotation as rotation_module
                    sys.modules['e3nn.o3._rotation'] = rotation_module

                # Save original
                original_func = rotation_module.matrix_to_angles
                print("Global patch: Original func ID:", id(original_func))

                def patched_matrix_to_angles(R):
                    print("patched_matrix_to_angles called")
                    if torch.onnx.is_in_onnx_export():
                        # Export: Skip assert, compute angles (exact E3NN logic)
                        R = R.clone().detach()  # Tracing-safe
                        # Skip problematic assert
                        alpha = torch.atan2(R[..., 2, 1], R[..., 2, 2])
                        beta = torch.asin(-R[..., 2, 0])
                        gamma = torch.atan2(R[..., 1, 0], R[..., 0, 0])
                        return alpha, beta, gamma
                    else:
                        # Runtime: Original
                        return original_func(R)

                # Apply global patch
                rotation_module.matrix_to_angles = patched_matrix_to_angles
                print("Global patch applied: New func ID:", id(rotation_module.matrix_to_angles))
                print("Patch active for gaussian_adapter/rotate_sh calls")

                # Standalone test (with rotations matching your gaussian_adapter dims, e.g., [B, V, 3, 3])
                test_R = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)  # [1,2,3,3]
                angles = rotation_module.matrix_to_angles(test_R)  # Now uses patched!
                print("Global patch test success: Angles shapes", [a.shape for a in angles])

                # with torch.no_grad():
                #     flat_output = export_model(*export_args)
                #     print("Flattened output keys:", list(flat_output.keys()))
                #     for k, v in flat_output.items():
                #         print(f"  {k}: {v.shape}")
                #     # Expected: gaussians_means: [1,983040,3], etc., + depths: [1,2,512,960]

                # Optional: Print shapes to verify (adjust keys to your actual ones)
                print("Export input shapes:")
                for key, value in context_for_export.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")

                input_names = [f'context.{key}' for key in context_for_export if isinstance(context_for_export[key], torch.Tensor)]
                print(f"ONNX contextinput names: {input_names}")

                output_names = [
                    'gaussians_means',
                    'gaussians_covariances',
                    'gaussians_opacities',
                    'gaussians_harmonics',
                    'visualization_dump_depth',
                    'visualization_dump_scales',
                    'visualization_dump_rotations',
                    'depths'
                ]

                # In run_encoder, after with torch.no_grad(): flat_output = export_model(context=context_for_export)
                # Trace inv_ex calls (requires torch 2.1+)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                    flat_output = export_model(context=context_for_export)
                prof.export_chrome_trace("inv_trace.json")  # View in chrome://tracing
                print("Inv ops in trace: Check inv_trace.json for 'inv_ex' nodes")

                onnx_program = torch.onnx.export(export_model,
                                                args=(),
                                                kwargs={"context": context_for_export},
                                                export_params=True,         # Store trained parameters within the model
                                                input_names=input_names,
                                                output_names=output_names,    # Name for the output node (assuming single output)
                                                dynamo=True,
                                                verbose=True,
                                                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,  # Fallback for custom Functions
                                                do_constant_folding=True,  # Avoid folding errors in checkpoint remnants
                                                optimize=True,  # Skip JIT optimizations that trigger 'Subgraph' pass
                                                report=True,
                                                )
                onnx_program.save(onnx_model_name)
                print(f"Encoder exported successfully to {onnx_model_name}")

                import onnx
                onnx_model = onnx.load(onnx_model_name)
                onnx.checker.check_model(onnx_model)
                return

            if run_onnx:
                global ort_session
                # Build input_feed dict mapping ONNX input names to numpy arrays.
                # Inputs were exported as f"context.{key}" for each tensor-valued entry in context_for_export.
                ort_inputs: Dict[str, np.ndarray] = {}
                for inp in ort_session.get_inputs():
                    name = inp.name  # e.g., "context.image"
                    if not name.startswith("context."):
                        continue
                    ctx_key = name.split(".", 1)[1]
                    if ctx_key not in context_for_export:
                        raise KeyError(
                            f"ONNX input '{name}' expects context key '{ctx_key}', which is missing."
                        )
                    tensor = context_for_export[ctx_key]
                    if hasattr(tensor, "detach"):
                        tensor = tensor.detach().cpu()
                    ort_inputs[name] = tensor.numpy()

                onnx_outputs = ort_session.run(None, ort_inputs)
                print("ONNXRuntime inference completed.")
                print("ONNXRuntime output count:", len(onnx_outputs))

                # Map outputs by name so we're robust to ordering
                output_names = [out.name for out in ort_session.get_outputs()]
                output_map = {name: onnx_outputs[i] for i, name in enumerate(output_names)}

                # Expected outputs from export: gaussians_{means,covariances,opacities,harmonics}, depths
                gauss_means_np = output_map["gaussians_means"]
                gauss_covs_np = output_map["gaussians_covariances"]
                gauss_opac_np = output_map["gaussians_opacities"]
                gauss_harm_np = output_map["gaussians_harmonics"]
                vd_depth_np = output_map["visualization_dump_depth"]
                vd_scales_np = output_map["visualization_dump_scales"]
                vd_rotations_np = output_map["visualization_dump_rotations"]
                depths_np = output_map["depths"]

                # Convert back to torch tensors
                gauss_means = torch.from_numpy(gauss_means_np)
                gauss_covs = torch.from_numpy(gauss_covs_np)
                gauss_opac = torch.from_numpy(gauss_opac_np)
                gauss_harm = torch.from_numpy(gauss_harm_np)
                vd_depth = torch.from_numpy(vd_depth_np)
                vd_scales = torch.from_numpy(vd_scales_np)
                vd_rotations = torch.from_numpy(vd_rotations_np)
                depths = torch.from_numpy(depths_np)

                # Reconstruct a Gaussians-like object compatible with downstream code
                gaussians = GaussiansOut(
                    means=gauss_means,
                    covariances=gauss_covs,
                    harmonics=gauss_harm,
                    opacities=gauss_opac,
                )

                visualization_dump_dict = {
                    "depth": vd_depth,
                    "scales": vd_scales,
                    "rotations": vd_rotations,
                }

                # Keep result structure consistent with the native encoder path
                result = {"gaussians": gaussians, "depths": depths, "visualization_dump": visualization_dump_dict}

        if not run_onnx and not run_coreml:
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

    visualization_dump = None
    if isinstance(result, dict):
        gaussians = result["gaussians"]
        depths = result.get("depths", None)
        visualization_dump = result.get("visualization_dump", None)
        if depths is not None:
            print(f"  Depths: {depths.shape}")
        if gaussians is None:
            raise ValueError("Encoder returned None for gaussians. Check config (train_depth_only should be False).")
    else:
        gaussians = result

    avg_encoder_elapsed = total_encoder_elapsed / max(num_runs, 1)
    print(f"\n  Encoder total time over {num_runs} run(s): {total_encoder_elapsed:.3f} seconds")
    print(f"  Encoder average time per run: {avg_encoder_elapsed:.3f} seconds ({avg_encoder_elapsed/60:.2f} minutes)")


    print(f"\nGaussian Splat Output:")
    print(f"  Means: {gaussians.means.shape}")
    print(f"  Covariances: {gaussians.covariances.shape}")
    print(f"  Harmonics: {gaussians.harmonics.shape}")
    print(f"  Opacities: {gaussians.opacities.shape}")

    if isinstance(visualization_dump, dict) and "depth" in visualization_dump:
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
    elif visualization_dump is None:
        print("\n  Depth Statistics: skipped (visualization dump not available; did you export with minimal CoreML outputs?)")

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

    available_viz_keys = list(visualization_dump.keys()) if isinstance(visualization_dump, dict) else []
    scales_tensor = visualization_dump.get("scales") if isinstance(visualization_dump, dict) else None
    rotations_tensor = visualization_dump.get("rotations") if isinstance(visualization_dump, dict) else None

    def _valid_viz_tensor(tensor: Optional[torch.Tensor], last_dim: int) -> bool:
        return isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 and tensor.shape[-1] == last_dim

    has_viz_scales = _valid_viz_tensor(scales_tensor, 3) and _valid_viz_tensor(rotations_tensor, 4)
    if not has_viz_scales:
        print("✗ Warning: visualization_dump does not contain scales/rotations.")
        if visualization_dump is None:
            print("  Visualization data is unavailable (likely because the CoreML model was exported with minimal outputs).")
        else:
            print("  This may happen if the encoder config has certain settings.")
            if isinstance(rotations_tensor, torch.Tensor):
                print(f"  Rotations tensor shape: {tuple(rotations_tensor.shape)} (expected last dim=4)")
            if isinstance(scales_tensor, torch.Tensor):
                print(f"  Scales tensor shape: {tuple(scales_tensor.shape)} (expected last dim=3)")
        print(f"  Available keys in visualization_dump: {available_viz_keys}")
        print("  Falling back to deriving scales/rotations from the Gaussian covariances.\n")

    try:
        step_timings: Dict[str, float] = {}

        scales_world: torch.Tensor
        rotations_world: torch.Tensor

        if has_viz_scales and scales_tensor is not None and rotations_tensor is not None:
            fetch_start = time.perf_counter()
            scales_world = scales_tensor.detach().reshape(-1, scales_tensor.shape[-1]).cpu()
            rotations_world = rotations_tensor.detach().reshape(-1, rotations_tensor.shape[-1]).cpu()
            total_gaussians = rotations_world.shape[0]
            fetch_end = time.perf_counter()
            step_timings["fetch_visualization_scales_rotations"] = fetch_end - fetch_start
        else:
            # ------------------------------------------------------------------
            # Recover per-Gaussian scales and world-space rotations directly
            # from the exported covariance matrices.
            #
            # For each Gaussian, the covariance has the form:
            #     cov = R @ diag(s^2) @ R^T
            # where R is a 3x3 rotation matrix and s is the per-axis scale.
            # Since cov is symmetric positive definite, eigen-decomposition:
            #     cov = V diag(λ) V^T
            # yields eigenvectors V (rotation) and eigenvalues λ (s^2).
            # ------------------------------------------------------------------
            extract_start_time = time.perf_counter()

            cov_world = gaussians.covariances[0].detach().cpu()  # [num_gaussians, 3, 3]
            eigvals, eigvecs = torch.linalg.eigh(cov_world)      # eigvals: [N,3], eigvecs: [N,3,3]
            eigvals = torch.clamp(eigvals, min=1e-12)

            # Scales are sqrt of eigenvalues
            scales = torch.sqrt(eigvals)  # [num_gaussians, 3]

            # Eigenvectors give an orthonormal rotation matrix per Gaussian.
            rot_mats = eigvecs  # [num_gaussians, 3, 3]

            # Ensure right-handed coordinate system: if det < 0, flip the last column.
            det = torch.det(rot_mats)
            if (det < 0).any():
                flip_mask = det < 0
                rot_mats[flip_mask, :, 2] *= -1.0

            # Convert rotation matrices to xyzw quaternions using SciPy's convention.
            rot_quats_np = R.from_matrix(rot_mats.numpy()).as_quat().astype(np.float32)
            rotations_world = torch.from_numpy(rot_quats_np)  # [num_gaussians, 4] (x, y, z, w)
            scales_world = scales.detach().cpu()
            total_gaussians = rotations_world.shape[0]

            extract_end_time = time.perf_counter()
            step_timings["extract_scales_rotations"] = extract_end_time - extract_start_time

        num_gaussians_per_view = total_gaussians // num_views
        if total_gaussians % num_views != 0:
            raise ValueError(
                f"Total gaussians ({total_gaussians}) must be divisible by num_views ({num_views})"
            )

        extrinsics_start_time = time.perf_counter()
        reference_extrinsics = context["extrinsics"][0, 0].detach().cpu()
        extrinsics_end_time = time.perf_counter()
        step_timings["get_reference_extrinsics"] = extrinsics_end_time - extrinsics_start_time

        rotation_conv_start_time = time.perf_counter()
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
    except Exception as e:
        print("✗ Error: Failed to export PLY from Gaussian outputs.")
        print(f"  Reason: {e}")
        print("  Tip: Re-run with --debug for a full stack trace or export the CoreML model with full outputs.")
        ply_path = None
        ply_export_elapsed = None

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
        print("  PLY export: Not completed (see warnings above)")
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
    setup_result = setup_encoder()
    # INSERT_YOUR_CODE
    # Get list of all image files in IMAGE_BASE_PATH directory (common image extensions)
    import os

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_dir = IMAGE_BASE_PATH if isinstance(IMAGE_BASE_PATH, str) else str(IMAGE_BASE_PATH)
    all_files = os.listdir(image_dir)
    images = [
        os.path.join(image_dir, f)
        for f in all_files
        if os.path.splitext(f)[1].lower() in image_extensions
           and os.path.isfile(os.path.join(image_dir, f))
    ]
    print(f"Found {len(images)} image(s) in {image_dir}:")
    for img_path in images:
        print(f"  {img_path}")
    run_encoder(setup_result, images)


if __name__ == "__main__":
    main()
