"""
Run encoder using direct config loading (no Hydra) and export to PLY.

This module can be imported and used programmatically:

    from inference import DepthSplatInference
    
    # Initialize encoder (uses module-level config variables)
    model = DepthSplatInference()
    
    # Run inference from file paths
    ply_bytes = model.run_from_paths("/path/to/images")
    
    # Or run inference from pre-loaded data
    ply_bytes = model.run_from_data(
        images=images_tensor,  # [num_views, 3, H, W]
        intrinsics_list=intrinsics_list,  # List of 3x3 numpy arrays
        extrinsics_list=extrinsics_list,  # List of 4x4 numpy arrays
    )

Image resolution: 512x960
Input views: determined by image files in the input directory
"""

# ============================================================================
# Configuration - Modify these paths as needed
# ============================================================================

CHECKPOINT_PATH = "pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"  # Set to None for random init
CONFIG_ROOT = "/content/depthsplat/config"  # Path to config directory
OUTPUT_DIR = "/content/drive/MyDrive/DepthSplat/run-output"

# Base directory containing the images and metadata files
# All .png and .jpg images in this directory will be processed
# For each image, a corresponding *_metadata.json file must exist in the same directory
# (e.g., for "dude_1.png", there must be "dude_1_metadata.json")
# Camera intrinsics and extrinsics will be loaded from these metadata files.
IMAGE_BASE_PATH = "/content/drive/MyDrive/DepthSplat/3_new_input"

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
import io
from dataclasses import dataclass
from typing import Optional

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
import json


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    target_height: int = 512
    target_width: int = 960
    near_disparity: float = 1.0
    far_disparity: float = 0.1
    verbose: bool = True


def load_metadata_from_json(image_path: Path) -> dict:
    """
    Load camera intrinsics and extrinsics from JSON metadata file.
    
    Metadata file should be named: {image_stem}_metadata.json
    For example, for "dude_1.png", the metadata file should be "dude_1_metadata.json"
    
    Expected JSON structure:
    {
        "intrinsics": [fx, 0, cx, 0, fy, cy, 0, 0, 1],  # 3x3 matrix in row-major order
        "extrinsics": [r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz, 0, 0, 0, 1],  # 4x4 matrix in row-major order
        "image_width": int,
        "image_height": int
    }
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing:
            - intrinsics: 3x3 numpy array
            - extrinsics: 4x4 numpy array (camera-to-world matrix)
            - image_width: int
            - image_height: int
    """
    # Construct metadata filename
    metadata_path = image_path.parent / f"{image_path.stem}_metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"Expected metadata file for image: {image_path}"
        )
    
    # Load JSON
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Parse intrinsics (3x3 matrix in row-major order)
    if "intrinsics" not in metadata:
        raise ValueError(f"Metadata file {metadata_path} missing 'intrinsics' field")
    
    intrinsics_flat = metadata["intrinsics"]
    if len(intrinsics_flat) != 9:
        raise ValueError(f"Intrinsics must have 9 elements (3x3 matrix), got {len(intrinsics_flat)}")
    
    intrinsics = np.array(intrinsics_flat, dtype=np.float32).reshape(3, 3)
    
    # Parse extrinsics (4x4 matrix in row-major order)
    if "extrinsics" not in metadata:
        raise ValueError(f"Metadata file {metadata_path} missing 'extrinsics' field")
    
    extrinsics_flat = metadata["extrinsics"]
    if len(extrinsics_flat) != 16:
        raise ValueError(f"Extrinsics must have 16 elements (4x4 matrix), got {len(extrinsics_flat)}")
    
    extrinsics = np.array(extrinsics_flat, dtype=np.float32).reshape(4, 4)
    
    # Get image dimensions from metadata
    image_width = metadata.get("image_width")
    image_height = metadata.get("image_height")
    
    return {
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "image_width": image_width,
        "image_height": image_height,
    }


def normalize_rotation_matrix(R_mat: torch.Tensor) -> torch.Tensor:
    """
    Normalize a 3x3 rotation matrix to ensure it's a valid rotation matrix.
    Uses SVD to orthogonalize the matrix and ensure determinant = 1.
    
    Args:
        R_mat: 3x3 rotation matrix (may not be perfectly orthogonal)
    
    Returns:
        Normalized 3x3 rotation matrix with determinant = 1
    """
    U, _, Vt = torch.linalg.svd(R_mat)
    R_normalized = U @ Vt
    
    det = torch.det(R_normalized)
    if det < 0:
        U_mod = U.clone()
        U_mod[:, -1] *= -1
        R_normalized = U_mod @ Vt
    
    return R_normalized


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

    image = load_image(image_path)
    target_height, target_width = target_size
    resize_transform = tf.Resize((target_height, target_width), antialias=True)
    image = resize_transform(image)

    return image


def _load_encoder_config(config_root: str, overrides: dict = None) -> EncoderDepthSplatCfg:
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

    if overrides:
        print("Applying config overrides:")
        for key, value in overrides.items():
            if isinstance(value, dict) and key in encoder_cfg_dict:
                for nested_key, nested_value in value.items():
                    print(f"  {key}.{nested_key}: {nested_value}")
                    encoder_cfg_dict[key][nested_key] = nested_value
            else:
                print(f"  {key}: {value}")
                encoder_cfg_dict[key] = value

    encoder_cfg = load_typed_config(encoder_cfg_dict, EncoderDepthSplatCfg)

    print(f"Encoder config loaded successfully!")
    print(f"  name: {encoder_cfg.name}")
    print(f"  num_scales: {encoder_cfg.num_scales}")
    print(f"  monodepth_vit_type: {encoder_cfg.monodepth_vit_type}")
    print(f"  gaussian_scale_max: {encoder_cfg.gaussian_adapter.gaussian_scale_max}")

    return encoder_cfg


class DepthSplatInference:
    """
    DepthSplat inference class for generating Gaussian splats from images.
    
    Uses module-level configuration variables:
        - CHECKPOINT_PATH: Path to model checkpoint
        - CONFIG_ROOT: Path to config directory
        - ENCODER_OVERRIDES: Dictionary of config overrides
    
    Example:
        from inference import DepthSplatInference
        
        model = DepthSplatInference()
        ply_bytes = model.run_from_paths("/path/to/images")
    """
    
    def __init__(self):
        """Initialize the encoder using module-level configuration."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load encoder config
        print("\n" + "="*70)
        print("Loading Encoder Config")
        print("="*70)
        encoder_cfg = _load_encoder_config(CONFIG_ROOT, ENCODER_OVERRIDES)
        
        # Initialize encoder
        print("\n" + "="*70)
        print("Initializing Encoder")
        print("="*70)
        self.encoder, _ = get_encoder(encoder_cfg)
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
        print("Encoder initialized successfully!")
        
        # Load checkpoint if provided
        if CHECKPOINT_PATH and Path(CHECKPOINT_PATH).exists():
            print(f"\nLoading checkpoint from {CHECKPOINT_PATH}")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                encoder_state_dict = {
                    k.replace('encoder.', ''): v
                    for k, v in state_dict.items()
                    if k.startswith('encoder.')
                }
                self.encoder.load_state_dict(encoder_state_dict, strict=False)
            else:
                self.encoder.load_state_dict(checkpoint, strict=False)
            print("Checkpoint loaded successfully!")
        elif CHECKPOINT_PATH:
            print(f"\nWarning: Checkpoint path '{CHECKPOINT_PATH}' does not exist. Using randomly initialized weights.")
        else:
            print("\nNo checkpoint path provided. Using randomly initialized weights.")
    
    def _prepare_camera_data(
        self,
        extrinsics_list: list[np.ndarray],
        intrinsics_list: list[np.ndarray],
        original_sizes: list[tuple[int, int]],
        target_height: int,
        target_width: int,
        image_filenames: list[str],
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[float]]:
        """Prepare camera extrinsics and intrinsics tensors from lists."""
        num_views = len(extrinsics_list)
        
        camera_centers = []
        camera_distances = []
        viewing_directions = []
        processed_extrinsics = []
        
        for i, ext in enumerate(extrinsics_list):
            img_name = image_filenames[i] if i < len(image_filenames) else f"view_{i}"
            
            ext_tensor = torch.from_numpy(ext).float()
            
            rotation = ext_tensor[:3, :3]
            rotation_normalized = normalize_rotation_matrix(rotation)
            ext_tensor[:3, :3] = rotation_normalized
            
            det = torch.det(rotation_normalized)
            if verbose and not torch.allclose(det, torch.tensor(1.0), atol=1e-5):
                print(f"  WARNING: View {i} ({img_name}) rotation matrix determinant: {det.item():.6f}")
            
            camera_center = ext_tensor[:3, 3]
            camera_centers.append(camera_center)
            camera_distances.append(torch.norm(camera_center).item())
            
            view_dir = rotation_normalized[:, 2]
            viewing_directions.append(view_dir)
            
            processed_extrinsics.append(ext_tensor)
        
        extrinsics = torch.stack(processed_extrinsics, dim=0).unsqueeze(0)
        
        processed_intrinsics = []
        
        for i, K in enumerate(intrinsics_list):
            img_name = image_filenames[i] if i < len(image_filenames) else f"view_{i}"
            original_width, original_height = original_sizes[i]
            
            fx_orig = float(K[0, 0])
            fy_orig = float(K[1, 1])
            cx_orig = float(K[0, 2])
            cy_orig = float(K[1, 2])
            
            scale_x = target_width / original_width
            scale_y = target_height / original_height
            
            fx = fx_orig * scale_x
            fy = fy_orig * scale_y
            cx = cx_orig * scale_x
            cy = cy_orig * scale_y
            
            K_normalized = torch.eye(3, dtype=torch.float32)
            K_normalized[0, 0] = fx / target_width
            K_normalized[1, 1] = fy / target_height
            K_normalized[0, 2] = cx / target_width
            K_normalized[1, 2] = cy / target_height
            
            processed_intrinsics.append(K_normalized)
            
            if verbose:
                print(f"  View {i} ({img_name}):")
                print(f"    Original: {original_width}x{original_height}, fx={fx_orig:.2f}, fy={fy_orig:.2f}")
                print(f"    Scaled to {target_width}x{target_height}: fx={fx:.2f}, fy={fy:.2f}")
        
        intrinsics = torch.stack(processed_intrinsics, dim=0).unsqueeze(0)
        
        return extrinsics, intrinsics, camera_centers, viewing_directions, camera_distances
    
    def _compute_near_far_planes(
        self,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
        camera_centers: list[torch.Tensor],
        camera_distances: list[float],
        near_disparity: float = 1.0,
        far_disparity: float = 0.1,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute near and far planes based on camera baselines."""
        batch_size = 1
        num_views = extrinsics.shape[1]
        
        if camera_centers and len(camera_centers) >= 2:
            if verbose:
                print(f"  Computing Near/Far Planes from Camera Baselines:")
                print(f"    Using disparity values: near={near_disparity}px, far={far_disparity}px")
            
            near_computed = compute_depth_for_disparity(
                extrinsics,
                intrinsics,
                (height, width),
                near_disparity,
            )
            far_computed = compute_depth_for_disparity(
                extrinsics,
                intrinsics,
                (height, width),
                far_disparity,
            )
            
            origins = extrinsics[:, :, :3, 3]
            deltas = (origins[:, None, :, :] - origins[:, :, None, :]).norm(dim=-1)
            max_baseline = deltas.max().item()
            deltas_positive = deltas[deltas > 1e-6]
            min_baseline = deltas_positive.min().item() if len(deltas_positive) > 0 else max_baseline
            
            use_computed = True
            if near_computed[0].item() > 100 * max_baseline:
                if verbose:
                    print(f"    WARNING: Computed near plane too large, using fallback")
                use_computed = False
            
            if far_computed[0].item() > 1000 * max_baseline:
                if verbose:
                    print(f"    WARNING: Computed far plane too large, using fallback")
                use_computed = False
            
            near_valid = (near_computed[0].item() > 0 and 
                         not torch.isnan(near_computed[0]) and 
                         not torch.isinf(near_computed[0]))
            
            far_valid = (far_computed[0].item() > 0 and 
                        not torch.isnan(far_computed[0]) and 
                        not torch.isinf(far_computed[0]))
            
            if not near_valid or not use_computed:
                near_fallback = max(0.5, 0.5 * min_baseline)
                if camera_distances:
                    min_camera_dist = min(camera_distances)
                    near_fallback = max(near_fallback, 0.1 * min_camera_dist)
                near = torch.ones(batch_size, num_views, dtype=torch.float32) * near_fallback
            else:
                near = near_computed.unsqueeze(1).repeat(1, num_views)
            
            if not far_valid or not use_computed:
                far_fallback = 15.0 * max_baseline
                far_fallback = min(200.0, far_fallback)
                if camera_distances:
                    max_camera_dist = max(camera_distances)
                    far_fallback = max(far_fallback, 5.0 * max_camera_dist)
                far = torch.ones(batch_size, num_views, dtype=torch.float32) * far_fallback
            elif far_computed[0].item() <= near[0, 0].item():
                far_fallback = 15.0 * max_baseline
                far_fallback = min(200.0, far_fallback)
                if camera_distances:
                    max_camera_dist = max(camera_distances)
                    far_fallback = max(far_fallback, 5.0 * max_camera_dist)
                far = torch.ones(batch_size, num_views, dtype=torch.float32) * far_fallback
            else:
                far = far_computed.unsqueeze(1).repeat(1, num_views)
            
            if verbose:
                print(f"    Near plane: {near[0, 0].item():.6f}")
                print(f"    Far plane: {far[0, 0].item():.6f}")
        else:
            if verbose:
                print(f"  Cannot compute from baselines (need at least 2 cameras)")
                print(f"  Using fixed values: near=0.1, far=100.0")
            near = torch.ones(batch_size, num_views, dtype=torch.float32) * 0.1
            far = torch.ones(batch_size, num_views, dtype=torch.float32) * 100.0
        
        return near, far
    
    def _gaussians_to_ply_bytes(
        self,
        gaussians,
        visualization_dump: dict,
        context: dict,
        num_views: int,
    ) -> bytes:
        """Convert gaussians to PLY format and return as bytes."""
        if "scales" not in visualization_dump or "rotations" not in visualization_dump:
            raise ValueError(
                "visualization_dump does not contain scales/rotations. "
                "Cannot export to PLY without this information."
            )
        
        scales = visualization_dump["scales"][0]
        rotations = visualization_dump["rotations"][0]
        
        total_gaussians = rotations.shape[0]
        num_gaussians_per_view = total_gaussians // num_views
        
        if total_gaussians % num_views != 0:
            raise ValueError(
                f"Total gaussians ({total_gaussians}) must be divisible by num_views ({num_views})"
            )
        
        rotations_per_view = rotations.view(num_views, num_gaussians_per_view, 4)
        c2w_rotations = context["extrinsics"][0, :, :3, :3].detach().cpu()
        
        world_rotations_list = []
        for v in range(num_views):
            cam_rotations_np = R.from_quat(
                rotations_per_view[v].detach().cpu().numpy()
            ).as_matrix()
            
            c2w_rot = c2w_rotations[v].detach().cpu().numpy()
            c2w_rot_expanded = np.broadcast_to(
                c2w_rot[None, :, :], 
                (num_gaussians_per_view, 3, 3)
            )
            world_rotations_mat = c2w_rot_expanded @ cam_rotations_np
            world_rotations_quat = R.from_matrix(world_rotations_mat).as_quat()
            world_rotations_list.append(torch.from_numpy(world_rotations_quat))
        
        world_rotations = torch.cat(world_rotations_list, dim=0).to(rotations.device)
        
        means_world = gaussians.means[0].detach().cpu()
        scales_world = scales.detach().cpu()
        rotations_world = world_rotations.detach().cpu()
        harmonics_world = gaussians.harmonics[0].detach().cpu()
        opacities_world = gaussians.opacities[0].detach().cpu()
        
        x, y, z, w = rearrange(rotations_world.numpy(), "g xyzw -> xyzw g")
        rotations_ply = np.stack((w, x, y, z), axis=-1)
        
        harmonics_dc = harmonics_world[..., 0].numpy()
        
        attributes_list = [
            means_world.numpy(),
            np.zeros_like(means_world.numpy()),
            harmonics_dc,
            torch.logit(opacities_world[..., None]).numpy(),
            scales_world.log().numpy(),
            rotations_ply,
        ]
        
        attributes = np.concatenate(attributes_list, axis=1)
        
        dtype_full = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ]
        
        elements = np.empty(means_world.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        
        buffer = io.BytesIO()
        PlyData([PlyElement.describe(elements, "vertex")]).write(buffer)
        buffer.seek(0)
        return buffer.read()
    
    def run_from_paths(
        self,
        image_base_path: str,
        config: Optional[InferenceConfig] = None,
    ) -> bytes:
        """
        Run inference from image file paths.
        
        Loads images and metadata from a directory, runs the encoder, and returns
        the resulting Gaussian splat as PLY bytes.
        
        Args:
            image_base_path: Directory containing images and metadata files.
                For each image (e.g., "view_0.png"), a corresponding metadata file
                (e.g., "view_0_metadata.json") must exist.
            config: Inference configuration. Uses defaults if None.
        
        Returns:
            PLY file contents as bytes.
        """
        if config is None:
            config = InferenceConfig()
        
        image_base = Path(image_base_path)
        
        if not image_base.exists():
            raise FileNotFoundError(f"Image base path does not exist: {image_base}")
        
        if not image_base.is_dir():
            raise ValueError(f"Image base path must be a directory: {image_base}")
        
        # Discover all images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.PNG', '*.JPG', '*.jpeg', '*.JPEG']:
            image_files.extend(sorted(image_base.glob(ext)))
        
        if not image_files:
            raise ValueError(f"No image files found in: {image_base}")
        
        if config.verbose:
            print(f"Found {len(image_files)} image(s) in {image_base}")
        
        # Load images and metadata
        loaded_images = []
        intrinsics_list = []
        extrinsics_list = []
        original_sizes = []
        image_filenames = []
        
        for image_path in image_files:
            try:
                metadata = load_metadata_from_json(image_path)
                loaded_img = load_and_resize_image(
                    str(image_path), 
                    (config.target_height, config.target_width)
                )
                
                loaded_images.append(loaded_img)
                intrinsics_list.append(metadata["intrinsics"])
                extrinsics_list.append(metadata["extrinsics"])
                original_sizes.append((metadata["image_width"], metadata["image_height"]))
                image_filenames.append(image_path.name)
                
                if config.verbose:
                    print(f"  Loaded: {image_path.name}")
            except Exception as e:
                if config.verbose:
                    print(f"  Skipping {image_path.name}: {e}")
                continue
        
        if not loaded_images:
            raise ValueError(f"No valid image/metadata pairs found in: {image_base}")
        
        # Stack images
        images = torch.stack(loaded_images, dim=0)
        
        return self.run_from_data(
            images=images,
            intrinsics_list=intrinsics_list,
            extrinsics_list=extrinsics_list,
            original_sizes=original_sizes,
            image_filenames=image_filenames,
            config=config,
        )
    
    def run_from_data(
        self,
        images: torch.Tensor,
        intrinsics_list: list[np.ndarray],
        extrinsics_list: list[np.ndarray],
        original_sizes: Optional[list[tuple[int, int]]] = None,
        image_filenames: Optional[list[str]] = None,
        config: Optional[InferenceConfig] = None,
    ) -> bytes:
        """
        Run inference from pre-loaded image data.
        
        Args:
            images: Image tensor of shape [num_views, 3, H, W] in range [0, 1].
            intrinsics_list: List of 3x3 intrinsic matrices (numpy arrays).
                These should be for the original image size before any resizing.
            extrinsics_list: List of 4x4 camera-to-world matrices (numpy arrays).
            original_sizes: List of (width, height) tuples for original image sizes.
                If None, assumes images are already at target resolution and uses
                the tensor dimensions.
            image_filenames: List of image filenames for logging. Optional.
            config: Inference configuration. Uses defaults if None.
        
        Returns:
            PLY file contents as bytes.
        """
        if config is None:
            config = InferenceConfig()
        
        num_views = images.shape[0]
        height, width = images.shape[2], images.shape[3]
        
        if original_sizes is None:
            original_sizes = [(width, height)] * num_views
        
        if image_filenames is None:
            image_filenames = [f"view_{i}" for i in range(num_views)]
        
        # Resize images if needed
        if height != config.target_height or width != config.target_width:
            resize_transform = tf.Resize(
                (config.target_height, config.target_width), 
                antialias=True
            )
            images = resize_transform(images)
            height, width = config.target_height, config.target_width
        
        # Add batch dimension
        images = images.unsqueeze(0)
        
        # Prepare camera data
        extrinsics, intrinsics, camera_centers, viewing_directions, camera_distances = \
            self._prepare_camera_data(
                extrinsics_list=extrinsics_list,
                intrinsics_list=intrinsics_list,
                original_sizes=original_sizes,
                target_height=height,
                target_width=width,
                image_filenames=image_filenames,
                verbose=config.verbose,
            )
        
        # Compute near/far planes
        near, far = self._compute_near_far_planes(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            height=height,
            width=width,
            camera_centers=camera_centers,
            camera_distances=camera_distances,
            near_disparity=config.near_disparity,
            far_disparity=config.far_disparity,
            verbose=config.verbose,
        )
        
        # Prepare context
        context = {
            "image": images.to(self.device),
            "extrinsics": extrinsics.to(self.device),
            "intrinsics": intrinsics.to(self.device),
            "near": near.to(self.device),
            "far": far.to(self.device),
        }
        
        # Run encoder
        visualization_dump = {}
        
        with torch.no_grad():
            result = self.encoder(
                context=context,
                global_step=0,
                deterministic=False,
                visualization_dump=visualization_dump,
                scene_names=None,
            )
        
        # Handle both dict and direct gaussians return
        if isinstance(result, dict):
            gaussians = result["gaussians"]
            if gaussians is None:
                raise ValueError("Encoder returned None for gaussians.")
        else:
            gaussians = result
        
        if config.verbose:
            print(f"Gaussian output: {gaussians.means.shape[1]} gaussians")
        
        # Convert to PLY bytes
        return self._gaussians_to_ply_bytes(
            gaussians=gaussians,
            visualization_dump=visualization_dump,
            context=context,
            num_views=num_views,
        )


def main():
    """Main function for command-line usage."""
    # Initialize model
    model = DepthSplatInference()
    
    # Run inference
    config = InferenceConfig(
        target_height=512,
        target_width=960,
        near_disparity=NEAR_DISPARITY,
        far_disparity=FAR_DISPARITY,
        verbose=True,
    )
    
    ply_bytes = model.run_from_paths(
        image_base_path=IMAGE_BASE_PATH,
        config=config,
    )

    # Save PLY to file
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "gaussians.ply"
    
    with open(ply_path, 'wb') as f:
        f.write(ply_bytes)
    
    print(f"\n✓ Successfully exported to {ply_path}")
    print(f"  File size: {len(ply_bytes) / (1024*1024):.2f} MB")

    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
