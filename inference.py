"""
Run encoder using direct config loading (no Hydra) and export to PLY.

Image resolution: 512x960
Input views: determined by EXTRINSICS_HARDCODED dictionary keys (image filenames)
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

# TensorRT Configuration
# NOTE: TensorRT compilation may fail with xformers and complex models
# If compilation fails, the script will automatically fall back to PyTorch
USE_TENSORRT = True  # Enable TensorRT optimization
TENSORRT_MODEL_PATH = "depthsplat_encoder_trt.ts"  # Path to save/load TensorRT model
TENSORRT_ENGINE_PATH = "depthsplat_encoder.engine"  # Path for TensorRT engine (ONNX path)
TENSORRT_ONNX_PATH = "depthsplat_encoder.onnx"  # Path for intermediate ONNX model
TENSORRT_FP16 = True  # Use FP16 precision (faster, slightly less accurate)
TENSORRT_USE_TORCH_COMPILE = False  # Use torch.compile instead (PyTorch 2.0+)
TENSORRT_USE_ONNX = True  # Use ONNX->TensorRT path (more reliable for complex models)
BENCHMARK_RUNS = 10  # Number of runs for benchmarking
WARMUP_RUNS = 5  # Number of warmup runs before benchmarking

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
import time
from typing import Optional

# Try to import torch_tensorrt if TensorRT is enabled
if USE_TENSORRT:
    try:
        import torch_tensorrt
        TENSORRT_AVAILABLE = True
        print(f"torch_tensorrt version: {torch_tensorrt.__version__}")
    except ImportError:
        TENSORRT_AVAILABLE = False
        print("Warning: torch_tensorrt not found. Install with: pip install torch-tensorrt")
        print("Falling back to standard PyTorch inference.")
else:
    TENSORRT_AVAILABLE = False


def patch_dinov2_pos_embed(model):
    """
    Patch DINOv2's positional embedding interpolation to avoid complex numbers.
    The original uses antialias=True which can trigger complex number operations via FFT.
    
    Must be called AFTER the encoder is created.
    """
    import types
    import torch.nn.functional as F
    
    def simple_interpolate_pos_encoding(self, x, w, h):
        """
        Simplified positional embedding interpolation without complex operations.
        Uses bilinear interpolation instead of bicubic with antialias.
        """
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        
        if npatch == N and w == h:
            return self.pos_embed
        
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        
        # Use sqrt(N) to get original grid size
        M = int(N ** 0.5)
        
        # Reshape and interpolate using bilinear (avoids complex numbers)
        patch_pos_embed = patch_pos_embed.reshape(1, M, M, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(h0, w0),
            mode="bilinear",  # Use bilinear instead of bicubic to avoid complex ops
            align_corners=False,
        )
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)
    
    patched = False
    try:
        dinov2_model = model.depth_predictor.pretrained
        
        # Patch the interpolate_pos_encoding method
        if hasattr(dinov2_model, 'interpolate_pos_encoding'):
            dinov2_model.interpolate_pos_encoding = types.MethodType(
                simple_interpolate_pos_encoding, dinov2_model
            )
            patched = True
            print("  ✓ Patched DINOv2 positional embedding interpolation (avoiding complex ops)")
    except Exception as e:
        print(f"  Warning: Could not patch DINOv2 pos embed: {e}")
    
    return patched


def patch_dinov2_attention(model):
    """
    Monkey-patch DINOv2's attention instances to use PyTorch native attention instead of xformers.
    This is needed for TensorRT compatibility since xformers uses SymInt which JIT can't trace.
    
    Args:
        model: The encoder model containing DINOv2 (must have depth_predictor.pretrained)
    
    Must be called AFTER the encoder is created (so DINOv2 modules are loaded).
    """
    import torch.nn.functional as F
    import types
    
    def pytorch_attention_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use PyTorch native scaled dot product attention
        x = F.scaled_dot_product_attention(q, k, v)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    # Find and patch all Attention instances in the DINOv2 model
    patched_count = 0
    
    try:
        # Get the DINOv2 model (pretrained attribute of depth_predictor)
        dinov2_model = model.depth_predictor.pretrained
        
        # Iterate through all modules and patch Attention instances
        for name, module in dinov2_model.named_modules():
            # Check if this is a DINOv2 Attention module by checking for the attributes we need
            if (hasattr(module, 'qkv') and 
                hasattr(module, 'num_heads') and 
                hasattr(module, 'proj') and 
                hasattr(module, 'proj_drop') and
                'attn' in name.lower()):
                # Bind our new forward method to this specific instance
                module.forward = types.MethodType(pytorch_attention_forward, module)
                patched_count += 1
        
        if patched_count > 0:
            print(f"  ✓ Patched {patched_count} DINOv2 attention instances to use PyTorch native attention")
        else:
            print("  Warning: No DINOv2 attention instances found to patch")
            
    except AttributeError as e:
        print(f"  Warning: Could not access DINOv2 model: {e}")
        print("  Trying alternative patching method...")
        
        # Fallback: try to patch via class
        try:
            import sys
            if 'dinov2.layers.attention' in sys.modules:
                dinov2_attention = sys.modules['dinov2.layers.attention']
                dinov2_attention.Attention.forward = pytorch_attention_forward
                print("  ✓ Patched DINOv2 Attention class (fallback method)")
                return True
        except Exception as e2:
            print(f"  Warning: Fallback patching also failed: {e2}")
    
    return patched_count > 0


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


class TensorRTEncoderWrapper(torch.nn.Module):
    """
    Wrapper for TensorRT compiled encoder that maintains the original interface.
    This allows the TensorRT model to be used as a drop-in replacement.
    """
    def __init__(self, trt_model):
        super().__init__()
        self.trt_model = trt_model
    
    def forward(self, context, global_step=0, deterministic=False, visualization_dump=None, scene_names=None):
        """
        Forward pass that matches the original encoder interface.
        
        The TensorRT model returns (means, covariances, harmonics, opacities),
        which we need to package back into the expected format.
        """
        # Extract inputs from context
        image = context["image"]
        extrinsics = context["extrinsics"]
        intrinsics = context["intrinsics"]
        near = context["near"]
        far = context["far"]
        
        # Run TensorRT model
        means, covariances, harmonics, opacities = self.trt_model(
            image, extrinsics, intrinsics, near, far
        )
        
        # Create a mock Gaussians object
        from src.model.types import Gaussians
        gaussians = Gaussians(
            means=means,
            covariances=covariances,
            harmonics=harmonics,
            opacities=opacities,
        )
        
        # Return in the same format as the original encoder
        return {"gaussians": gaussians}


class TensorRTEngineWrapper(torch.nn.Module):
    """
    Wrapper for TensorRT engine that runs inference using the TensorRT Python API.
    This is used with the ONNX->TensorRT path for more reliable compilation.
    """
    def __init__(self, engine_path: str, output_names: list):
        super().__init__()
        import tensorrt as trt
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.output_names = output_names
        
        # Load the engine
        print(f"  Loading TensorRT engine from {engine_path}...")
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Get binding info
        self.input_names = []
        self.output_shapes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                shape = self.engine.get_tensor_shape(name)
                self.output_shapes[name] = shape
        
        print(f"  ✓ Engine loaded with {len(self.input_names)} inputs, {len(self.output_shapes)} outputs")
    
    def forward(self, image, extrinsics, intrinsics, near, far):
        import tensorrt as trt
        
        # Set input shapes (for dynamic shapes)
        inputs = {
            "image": image,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "near": near,
            "far": far,
        }
        
        # Set input tensor addresses
        for name in self.input_names:
            tensor = inputs[name].contiguous()
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
        
        # Allocate output tensors
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = torch.float32  # Assume float32 outputs
            output = torch.empty(tuple(shape), dtype=dtype, device="cuda")
            outputs[name] = output
            self.context.set_tensor_address(name, output.data_ptr())
        
        # Run inference
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        
        # Return outputs in order
        return tuple(outputs[name] for name in self.output_names)


class TensorRTEngineEncoderWrapper(torch.nn.Module):
    """
    High-level wrapper that makes TensorRTEngineWrapper compatible with the encoder interface.
    """
    def __init__(self, engine_wrapper: TensorRTEngineWrapper):
        super().__init__()
        self.engine_wrapper = engine_wrapper
    
    def forward(self, context, global_step=0, deterministic=False, visualization_dump=None, scene_names=None):
        """Forward pass that matches the original encoder interface."""
        # Extract inputs from context
        image = context["image"]
        extrinsics = context["extrinsics"]
        intrinsics = context["intrinsics"]
        near = context["near"]
        far = context["far"]
        
        # Run TensorRT engine
        means, covariances, harmonics, opacities = self.engine_wrapper(
            image, extrinsics, intrinsics, near, far
        )
        
        # Create Gaussians object
        from src.model.types import Gaussians
        gaussians = Gaussians(
            means=means,
            covariances=covariances,
            harmonics=harmonics,
            opacities=opacities,
        )
        
        return {"gaussians": gaussians}


def compile_to_tensorrt_via_onnx(
    model: torch.nn.Module,
    example_inputs: dict,
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
) -> Optional[torch.nn.Module]:
    """
    Compile a PyTorch model to TensorRT via ONNX.
    
    This is more reliable than direct TorchScript->TensorRT conversion for complex models.
    
    Args:
        model: PyTorch model to compile (used for copying weights)
        example_inputs: Dictionary of example inputs for tracing
        onnx_path: Path to save intermediate ONNX model
        engine_path: Path to save TensorRT engine
        fp16: Whether to use FP16 precision
        
    Returns:
        TensorRT wrapped model or None if compilation failed
    """
    import os
    
    print("\n" + "="*70)
    print("Compiling Model to TensorRT via ONNX")
    print("="*70)
    print(f"  Target precision: {'FP16' if fp16 else 'FP32'}")
    print(f"  ONNX path: {onnx_path}")
    print(f"  Engine path: {engine_path}")
    
    # Check if engine already exists
    if engine_path.exists():
        print(f"  Found existing TensorRT engine at {engine_path}")
        try:
            output_names = ["means", "covariances", "harmonics", "opacities"]
            engine_wrapper = TensorRTEngineWrapper(str(engine_path), output_names)
            return TensorRTEngineEncoderWrapper(engine_wrapper)
        except Exception as e:
            print(f"  Failed to load existing engine: {e}")
            print("  Will rebuild engine...")
    
    # Set environment variable for PyTorch attention
    os.environ["FORCE_PYTORCH_ATTENTION"] = "1"
    
    try:
        print("  Re-creating encoder with PyTorch native attention...")
        
        # Re-create encoder with PyTorch attention
        encoder_cfg = load_encoder_config(CONFIG_ROOT, ENCODER_OVERRIDES)
        pytorch_encoder, _ = get_encoder(encoder_cfg)
        pytorch_encoder = pytorch_encoder.to("cuda").eval()
        
        # Patch DINOv2's attention and positional embedding interpolation
        patch_dinov2_attention(pytorch_encoder)
        patch_dinov2_pos_embed(pytorch_encoder)  # Avoid complex numbers in pos embed interpolation
        
        # Copy weights from original model
        pytorch_encoder.load_state_dict(model.state_dict())
        print("  ✓ Encoder re-created with PyTorch native attention")
        
        # Create a wrapper for ONNX export
        class EncoderONNXWrapper(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
            
            def forward(self, image, extrinsics, intrinsics, near, far):
                context = {
                    "image": image,
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "near": near,
                    "far": far,
                }
                result = self.encoder(
                    context=context,
                    global_step=0,
                    deterministic=True,
                    visualization_dump={},
                    scene_names=None,
                )
                gaussians = result["gaussians"]
                return gaussians.means, gaussians.covariances, gaussians.harmonics, gaussians.opacities
        
        wrapped_model = EncoderONNXWrapper(pytorch_encoder).cuda().eval()
        
        # Step 1: Export to ONNX
        print("  Exporting model to ONNX...")
        input_names = ["image", "extrinsics", "intrinsics", "near", "far"]
        output_names = ["means", "covariances", "harmonics", "opacities"]
        
        batch_size, num_views, channels, height, width = example_inputs["image"].shape
        
        # Define dynamic axes for flexible batch/view sizes
        dynamic_axes = {
            "image": {0: "batch", 1: "views"},
            "extrinsics": {0: "batch", 1: "views"},
            "intrinsics": {0: "batch", 1: "views"},
            "near": {0: "batch", 1: "views"},
            "far": {0: "batch", 1: "views"},
            "means": {0: "batch"},
            "covariances": {0: "batch"},
            "harmonics": {0: "batch"},
            "opacities": {0: "batch"},
        }
        
        onnx_export_success = False
        
        # Try multiple ONNX export methods
        # Method 1: Try torch.onnx.dynamo_export (PyTorch 2.x, handles more ops)
        print("    Trying dynamo-based ONNX export...")
        try:
            export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
            onnx_program = torch.onnx.dynamo_export(
                wrapped_model,
                example_inputs["image"],
                example_inputs["extrinsics"],
                example_inputs["intrinsics"],
                example_inputs["near"],
                example_inputs["far"],
                export_options=export_options,
            )
            onnx_program.save(str(onnx_path))
            onnx_export_success = True
            print("    ✓ Dynamo ONNX export successful")
        except Exception as e:
            print(f"    Dynamo export failed: {e}")
            print("    Trying legacy ONNX export...")
        
        # Method 2: Try legacy torch.onnx.export with verbose mode
        if not onnx_export_success:
            try:
                with torch.no_grad():
                    torch.onnx.export(
                        wrapped_model,
                        (
                            example_inputs["image"],
                            example_inputs["extrinsics"],
                            example_inputs["intrinsics"],
                            example_inputs["near"],
                            example_inputs["far"],
                        ),
                        str(onnx_path),
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        opset_version=17,
                        do_constant_folding=True,
                        export_params=True,
                        verbose=False,
                    )
                onnx_export_success = True
                print("    ✓ Legacy ONNX export successful")
            except RuntimeError as e:
                if "complex" in str(e).lower():
                    print(f"    ✗ ONNX export failed due to complex numbers: {e}")
                    print("\n    The model contains operations with complex numbers (likely in DINOv2).")
                    print("    This is a known limitation of ONNX export.")
                    print("\n    Workaround options:")
                    print("    1. Use torch.compile with inductor backend (set TENSORRT_USE_TORCH_COMPILE = True)")
                    print("    2. Use a different depth predictor backbone")
                    print("    3. Patch DINOv2 to avoid complex operations")
                else:
                    print(f"    ✗ ONNX export failed: {e}")
                raise
        
        if not onnx_export_success:
            raise RuntimeError("All ONNX export methods failed")
        
        print(f"  ✓ ONNX model exported to {onnx_path}")
        print(f"    File size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Step 2: Convert ONNX to TensorRT
        print("  Converting ONNX to TensorRT engine (this may take several minutes)...")
        
        try:
            import tensorrt as trt
        except ImportError:
            print("  ✗ tensorrt package not found. Install with: pip install tensorrt")
            print("  Falling back to standard PyTorch inference")
            return None
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Build the engine
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX
        with open(str(onnx_path), "rb") as f:
            if not parser.parse(f.read()):
                print("  ✗ Failed to parse ONNX model:")
                for i in range(parser.num_errors):
                    print(f"    {parser.get_error(i)}")
                return None
        
        print(f"    ONNX parsed successfully: {network.num_inputs} inputs, {network.num_outputs} outputs")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
        
        if fp16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("    FP16 mode enabled")
            else:
                print("    Warning: FP16 not supported on this platform")
        
        # Set optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()
        
        # Set shape ranges for each input
        # Min, Optimal, Max shapes
        profile.set_shape("image", 
                         (1, 2, 3, height, width),      # min
                         (batch_size, num_views, 3, height, width),  # opt
                         (batch_size, num_views * 2, 3, height, width))  # max
        profile.set_shape("extrinsics",
                         (1, 2, 4, 4),
                         (batch_size, num_views, 4, 4),
                         (batch_size, num_views * 2, 4, 4))
        profile.set_shape("intrinsics",
                         (1, 2, 3, 3),
                         (batch_size, num_views, 3, 3),
                         (batch_size, num_views * 2, 3, 3))
        profile.set_shape("near",
                         (1, 2),
                         (batch_size, num_views),
                         (batch_size, num_views * 2))
        profile.set_shape("far",
                         (1, 2),
                         (batch_size, num_views),
                         (batch_size, num_views * 2))
        
        config.add_optimization_profile(profile)
        
        # Build engine
        print("    Building TensorRT engine...")
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("  ✗ Failed to build TensorRT engine")
            return None
        
        # Save engine
        with open(str(engine_path), "wb") as f:
            f.write(serialized_engine)
        
        print(f"  ✓ TensorRT engine saved to {engine_path}")
        print(f"    File size: {engine_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Create and return wrapper
        engine_wrapper = TensorRTEngineWrapper(str(engine_path), output_names)
        return TensorRTEngineEncoderWrapper(engine_wrapper)
        
    except Exception as e:
        print(f"  ✗ ONNX->TensorRT compilation failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n  Falling back to standard PyTorch inference")
        return None
        
    finally:
        # Clean up
        if "FORCE_PYTORCH_ATTENTION" in os.environ:
            del os.environ["FORCE_PYTORCH_ATTENTION"]


def compile_with_torch_compile(
    model: torch.nn.Module,
    backend: str = "inductor",
) -> torch.nn.Module:
    """
    Compile a PyTorch model using torch.compile (PyTorch 2.0+).
    This is more robust than TensorRT for complex models with xformers.
    
    Args:
        model: PyTorch model to compile
        backend: Compilation backend ("inductor", "cudagraphs", etc.)
        
    Returns:
        Compiled model
    """
    print("\n" + "="*70)
    print("Compiling Model with torch.compile")
    print("="*70)
    print(f"  Backend: {backend}")
    
    try:
        # Check PyTorch version
        torch_version = torch.__version__.split('+')[0]
        major, minor = map(int, torch_version.split('.')[:2])
        
        if major < 2:
            print(f"  ✗ torch.compile requires PyTorch 2.0+, found {torch.__version__}")
            print("  Falling back to standard PyTorch")
            return None
        
        print("  Compiling model (first run will be slower)...")
        
        # Compile the model
        compiled_model = torch.compile(
            model,
            backend=backend,
            mode="max-autotune",  # Optimize for performance
        )
        
        print("  ✓ Model compiled successfully!")
        print("  Note: First inference will trigger actual compilation")
        
        return compiled_model
        
    except Exception as e:
        print(f"  ✗ torch.compile failed: {e}")
        print("  Falling back to standard PyTorch")
        import traceback
        traceback.print_exc()
        return None


def compile_to_tensorrt(
    model: torch.nn.Module,
    example_inputs: dict,
    save_path: Path,
    fp16: bool = True,
) -> Optional[torch.nn.Module]:
    """
    Compile a PyTorch model to TensorRT.
    
    This function re-creates the encoder with PyTorch native attention (instead of xformers)
    to ensure compatibility with JIT tracing required by TensorRT.
    
    Args:
        model: PyTorch model to compile (used for copying weights)
        example_inputs: Dictionary of example inputs for tracing
        save_path: Path to save the compiled model
        fp16: Whether to use FP16 precision
        
    Returns:
        TensorRT wrapped model or None if compilation failed
    """
    import os
    
    if not TENSORRT_AVAILABLE:
        print("TensorRT compilation skipped: torch_tensorrt not available")
        return None
    
    print("\n" + "="*70)
    print("Compiling Model to TensorRT")
    print("="*70)
    print(f"  Target precision: {'FP16' if fp16 else 'FP32'}")
    print(f"  Save path: {save_path}")
    
    # Set environment variable BEFORE creating the encoder
    # This makes CrossAttention use PyTorch native attention instead of xformers
    os.environ["FORCE_PYTORCH_ATTENTION"] = "1"
    
    try:
        print("  Re-creating encoder with PyTorch native attention...")
        
        # Re-create encoder with PyTorch attention (env var is now set)
        encoder_cfg = load_encoder_config(CONFIG_ROOT, ENCODER_OVERRIDES)
        pytorch_encoder, _ = get_encoder(encoder_cfg)
        pytorch_encoder = pytorch_encoder.to("cuda").eval()
        
        # Patch DINOv2's attention instances after the model is loaded (it uses xformers internally)
        # We must patch the actual instances, not just the class, since the model is already instantiated
        patch_dinov2_attention(pytorch_encoder)
        patch_dinov2_pos_embed(pytorch_encoder)  # Avoid complex numbers in pos embed interpolation
        
        # Copy weights from original model
        pytorch_encoder.load_state_dict(model.state_dict())
        print("  ✓ Encoder re-created with PyTorch native attention")
        
        # Create a wrapper that accepts separate tensor inputs
        # We pass shapes explicitly to avoid TensorRT shape analysis issues
        class EncoderWrapper(torch.nn.Module):
            def __init__(self, encoder, batch_size: int, num_views: int, height: int, width: int):
                super().__init__()
                self.encoder = encoder
                # Register shapes as buffers so they're constants in the traced graph
                self.register_buffer('_batch_size', torch.tensor(batch_size, dtype=torch.int64))
                self.register_buffer('_num_views', torch.tensor(num_views, dtype=torch.int64))
                self.register_buffer('_height', torch.tensor(height, dtype=torch.int64))
                self.register_buffer('_width', torch.tensor(width, dtype=torch.int64))
            
            def forward(self, image, extrinsics, intrinsics, near, far):
                context = {
                    "image": image,
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "near": near,
                    "far": far,
                }
                # Create a minimal visualization_dump
                visualization_dump = {}
                result = self.encoder(
                    context=context,
                    global_step=0,
                    deterministic=False,
                    visualization_dump=visualization_dump,
                    scene_names=None,
                )
                
                # Return only the core gaussian data
                if isinstance(result, dict):
                    gaussians = result["gaussians"]
                else:
                    gaussians = result
                
                # Return the essential gaussian properties as separate tensors
                return gaussians.means, gaussians.covariances, gaussians.harmonics, gaussians.opacities
        
        batch_size, num_views, channels, height, width = example_inputs["image"].shape
        wrapped_model = EncoderWrapper(pytorch_encoder, batch_size, num_views, height, width).cuda().eval()
        
        # Trace the model with example inputs
        print("  Tracing model with example inputs...")
        with torch.no_grad():
            traced_model = torch.jit.trace(
                wrapped_model,
                (
                    example_inputs["image"],
                    example_inputs["extrinsics"],
                    example_inputs["intrinsics"],
                    example_inputs["near"],
                    example_inputs["far"],
                ),
                strict=False,  # Allow non-tensor inputs and more flexible tracing
            )
            
            # Freeze the model to convert shape-dependent values to constants
            # This helps TensorRT understand static shapes during graph partitioning
            traced_model = torch.jit.freeze(traced_model)
        
        print("  Model traced and frozen successfully!")
        print("  Compiling with TensorRT (this may take several minutes)...")
        
        # Configure TensorRT compilation
        enabled_precisions = {torch.float}
        if fp16:
            enabled_precisions.add(torch.half)
        
        # Compile to TensorRT
        # Note: We use explicit shapes since the model expects specific input sizes
        # (batch_size, num_views, height, width were extracted earlier during wrapper creation)
        
        trt_inputs = [
            torch_tensorrt.Input(
                shape=[batch_size, num_views, channels, height, width],
                dtype=torch.float32,
            ),  # image
            torch_tensorrt.Input(
                shape=[batch_size, num_views, 4, 4],
                dtype=torch.float32,
            ),  # extrinsics
            torch_tensorrt.Input(
                shape=[batch_size, num_views, 3, 3],
                dtype=torch.float32,
            ),  # intrinsics
            torch_tensorrt.Input(
                shape=[batch_size, num_views],
                dtype=torch.float32,
            ),  # near
            torch_tensorrt.Input(
                shape=[batch_size, num_views],
                dtype=torch.float32,
            ),  # far
        ]
        
        # Try TorchScript-based compilation first
        use_dynamo_fallback = False
        try:
            trt_model = torch_tensorrt.compile(
                traced_model,
                ir="torchscript",  # Explicitly use TorchScript IR since we traced the model
                inputs=trt_inputs,
                enabled_precisions=enabled_precisions,
                truncate_long_and_double=True,
                workspace_size=1 << 30,  # 1GB workspace
                require_full_compilation=False,  # Allow fallback for unsupported ops
                min_block_size=1,  # Minimize subgraph fragmentation
            )
        except Exception as ts_error:
            print(f"  TorchScript compilation failed: {ts_error}")
            print("  Trying torch.compile with torch_tensorrt backend...")
            
            # Fall back to torch.compile with torch_tensorrt backend (Dynamo-based)
            # This handles dynamic shapes better
            try:
                trt_model = torch.compile(
                    wrapped_model,
                    backend="torch_tensorrt",
                    options={
                        "enabled_precisions": enabled_precisions,
                        "truncate_long_and_double": True,
                        "debug": True,
                        "min_block_size": 1,
                    }
                )
                
                # Run a warmup pass to trigger compilation
                print("  Running warmup pass to trigger compilation...")
                with torch.no_grad():
                    _ = trt_model(
                        example_inputs["image"],
                        example_inputs["extrinsics"],
                        example_inputs["intrinsics"],
                        example_inputs["near"],
                        example_inputs["far"],
                    )
                use_dynamo_fallback = True
                print("  ✓ Dynamo + TensorRT compilation successful!")
            except Exception as dynamo_error:
                print(f"  Dynamo compilation also failed: {dynamo_error}")
                raise ts_error  # Re-raise original error
        
        if not use_dynamo_fallback:
            print("  Compilation successful!")
            
            # Save the compiled model (only for TorchScript-based)
            print(f"  Saving TensorRT model to {save_path}...")
            torch.jit.save(trt_model, str(save_path))
            
            print(f"  ✓ TensorRT model saved successfully!")
            print(f"  File size: {save_path.stat().st_size / (1024*1024):.2f} MB")
        else:
            print("  Note: Dynamo-compiled models cannot be saved to disk")
            print("  The model will be recompiled on each run")
        
        # Wrap the TensorRT model to match the original encoder interface
        wrapped_trt = TensorRTEncoderWrapper(trt_model)
        
        return wrapped_trt
        
    except Exception as e:
        print(f"  ✗ TensorRT compilation failed: {e}")
        
        # Check for common issues and provide helpful guidance
        error_str = str(e)
        if "SymInt" in error_str or "unsupported argument type" in error_str:
            print("\n  This error is typically caused by:")
            print("    - xformers operations that use symbolic integers")
            print("    - Dynamic shapes in the model")
            print("    - Incompatibility between JIT tracing and certain operations")
            print("\n  Recommended solutions:")
            print("    1. Use torch.compile instead: Set TENSORRT_USE_TORCH_COMPILE = True")
            print("    2. Or disable optimization: Set USE_TENSORRT = False")
            print("    3. See TENSORRT_USAGE.md for more details")
        
        print(f"\n  Falling back to standard PyTorch inference")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Clean up - restore xformers for normal inference
        if "FORCE_PYTORCH_ATTENTION" in os.environ:
            del os.environ["FORCE_PYTORCH_ATTENTION"]


def benchmark_inference(
    model: torch.nn.Module,
    context: dict,
    num_warmup: int = 10,
    num_runs: int = 100,
    model_name: str = "Model",
) -> dict:
    """
    Benchmark model inference time.
    
    Args:
        model: Model to benchmark
        context: Input context dictionary
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        model_name: Name of the model for display
        
    Returns:
        Dictionary with timing statistics
    """
    print("\n" + "="*70)
    print(f"Benchmarking {model_name} Inference")
    print("="*70)
    
    # Warmup runs
    print(f"  Running {num_warmup} warmup iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(context=context, global_step=0, deterministic=False, visualization_dump={}, scene_names=None)
    
    # Synchronize CUDA before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark runs
    print(f"  Running {num_runs} benchmark iterations...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            
            _ = model(context=context, global_step=0, deterministic=False, visualization_dump={}, scene_names=None)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    # Compute statistics
    times = np.array(times)
    stats = {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "median": float(np.median(times)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
    }
    
    print(f"\n  Timing Statistics ({num_runs} runs):")
    print(f"    Mean:   {stats['mean']*1000:.2f} ms")
    print(f"    Median: {stats['median']*1000:.2f} ms")
    print(f"    Std:    {stats['std']*1000:.2f} ms")
    print(f"    Min:    {stats['min']*1000:.2f} ms")
    print(f"    Max:    {stats['max']*1000:.2f} ms")
    print(f"    P95:    {stats['p95']*1000:.2f} ms")
    print(f"    P99:    {stats['p99']*1000:.2f} ms")
    print(f"    FPS:    {1.0/stats['mean']:.2f}")
    
    return stats


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

    # Load images and metadata
    print("\n" + "="*70)
    print("Loading Images and Metadata")
    print("="*70)
    
    # Set up paths
    image_base = Path(IMAGE_BASE_PATH)
    
    if not image_base.exists():
        raise FileNotFoundError(f"IMAGE_BASE_PATH does not exist: {image_base}")
    
    if not image_base.is_dir():
        raise ValueError(f"IMAGE_BASE_PATH must be a directory: {image_base}")
    
    # Discover all .png and .jpg images in the directory
    image_files = []
    for ext in ['*.png', '*.jpg', '*.PNG', '*.JPG', '*.jpeg', '*.JPEG']:
        image_files.extend(sorted(image_base.glob(ext)))
    
    if not image_files:
        raise ValueError(f"No image files (.png, .jpg) found in: {image_base}")
    
    print(f"  Found {len(image_files)} image(s) in {image_base}")
    
    # Load all images and metadata
    batch_size = 1
    num_views = len(image_files)
    loaded_images = []
    metadata_list = []
    image_filenames = []
    
    # Target resolution for inference (will resize images to this)
    # Can be adjusted based on your needs
    target_height, target_width = 512, 960
    
    for image_path in image_files:
        image_filename = image_path.name
        
        # Load metadata
        print(f"\n  Loading: {image_filename}")
        try:
            metadata = load_metadata_from_json(image_path)
            metadata_list.append(metadata)
        except FileNotFoundError as e:
            print(f"    ✗ Skipping {image_filename}: {e}")
            continue
        except Exception as e:
            print(f"    ✗ Error loading metadata for {image_filename}: {e}")
            continue
        
        # Load and resize image
        try:
            loaded_img = load_and_resize_image(str(image_path), (target_height, target_width))
            loaded_images.append(loaded_img)
            image_filenames.append(image_filename)
            
            print(f"    ✓ Image loaded: {image_path}")
            print(f"    ✓ Metadata loaded from: {image_path.parent / f'{image_path.stem}_metadata.json'}")
            print(f"    Original size: {metadata['image_width']}x{metadata['image_height']}")
            print(f"    Resized to: {target_width}x{target_height}")
        except Exception as e:
            print(f"    ✗ Error loading image {image_filename}: {e}")
            # Remove the metadata we just added since the image failed
            metadata_list.pop()
            continue
    
    if not loaded_images:
        raise ValueError(f"No valid image/metadata pairs found in: {image_base}")
    
    # Update num_views based on successfully loaded images
    num_views = len(loaded_images)
    
    # Stack images: [num_views, 3, height, width] -> [1, num_views, 3, height, width]
    images = torch.stack(loaded_images, dim=0).unsqueeze(0)
    height, width = target_height, target_width
    
    print(f"\n  Successfully loaded {num_views} image(s) with metadata")
    print(f"  Final image tensor shape: {images.shape}")

    # Create camera poses from metadata
    print("\n" + "="*70)
    print("Setting Up Camera Poses from Metadata")
    print("="*70)

    # Initialize variables for validation
    camera_centers = []
    camera_distances = []
    viewing_directions = []
    
    extrinsics_list = []
    
    for i, metadata in enumerate(metadata_list):
        img_name = image_filenames[i]
        
        # Get extrinsics from metadata (already a 4x4 numpy array)
        ext = metadata["extrinsics"]
        ext_tensor = torch.from_numpy(ext).float()
        
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
        img_name = image_filenames[i]
        print(f"      View {i} ({img_name}): [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] (distance from origin: {dist:.3f})")
    
    print(f"\n    Viewing directions (camera forward in world coordinates):")
    for i, view_dir in enumerate(viewing_directions):
        img_name = image_filenames[i]
        print(f"      View {i} ({img_name}): [{view_dir[0]:.3f}, {view_dir[1]:.3f}, {view_dir[2]:.3f}]")
    
    # Check camera distances consistency
    if len(set([round(d, 1) for d in camera_distances])) > 1:
        print(f"\n    NOTE: Camera distances vary:")
        for i, dist in enumerate(camera_distances):
            img_name = image_filenames[i]
            print(f"      View {i} ({img_name}): {dist:.3f}")
    
    # Check if cameras are too close or too far
    avg_distance = sum(camera_distances) / len(camera_distances)
    if avg_distance < 0.01:
        print(f"\n    WARNING: Cameras are very close to origin (avg distance: {avg_distance:.3f})")
        print(f"    This may cause numerical issues. Consider scaling the scene.")
    elif avg_distance > 1000:
        print(f"\n    WARNING: Cameras are very far from origin (avg distance: {avg_distance:.3f})")
        print(f"    This may cause precision issues. Consider scaling the scene.")
    
    # Check baseline (distance between cameras) if we have multiple views
    if num_views > 1:
        print(f"\n    Camera baselines (distances between camera centers):")
        for i in range(len(camera_centers)):
            for j in range(i + 1, len(camera_centers)):
                baseline = torch.norm(camera_centers[i] - camera_centers[j]).item()
                img_name_i = image_filenames[i]
                img_name_j = image_filenames[j]
                print(f"      View {i} ({img_name_i}) <-> View {j} ({img_name_j}): {baseline:.3f}")
        
        # Check viewing angles between cameras
        print(f"\n    Viewing angles between cameras:")
        for i in range(len(viewing_directions)):
            for j in range(i + 1, len(viewing_directions)):
                angle_rad = torch.acos(torch.clamp(torch.dot(viewing_directions[i], viewing_directions[j]), -1.0, 1.0))
                angle_deg = angle_rad * 180 / math.pi
                img_name_i = image_filenames[i]
                img_name_j = image_filenames[j]
                print(f"      View {i} ({img_name_i}) <-> View {j} ({img_name_j}): {angle_deg:.1f}°")
    
    # Print full extrinsic matrices
    print(f"\n    Full extrinsic matrices (C2W):")
    for i, ext in enumerate(extrinsics_list):
        img_name = image_filenames[i]
        print(f"      View {i} ({img_name}):")
        ext_np = ext.numpy()
        for row in ext_np:
            print(f"        [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}, {row[3]:8.4f}]")

    # Set up intrinsics from metadata
    print("\n" + "="*70)
    print("Setting Up Camera Intrinsics from Metadata")
    print("="*70)
    
    print(f"  Target inference dimensions: width={width}, height={height}")
    
    # Process intrinsics for each view
    intrinsics_list = []
    
    for i, metadata in enumerate(metadata_list):
        img_name = image_filenames[i]
        
        # Get intrinsics from metadata (3x3 numpy array)
        K = metadata["intrinsics"]
        
        # Extract focal lengths and principal point from metadata
        # Intrinsics are for the original image size in metadata
        original_width = metadata["image_width"]
        original_height = metadata["image_height"]
        
        fx_orig = float(K[0, 0])
        fy_orig = float(K[1, 1])
        cx_orig = float(K[0, 2])
        cy_orig = float(K[1, 2])
        
        print(f"\n  View {i} ({img_name}):")
        print(f"    Original image size: {original_width}x{original_height}")
        print(f"    Original intrinsics: fx={fx_orig:.2f}, fy={fy_orig:.2f}, cx={cx_orig:.2f}, cy={cy_orig:.2f}")
        
        # Scale intrinsics to match resized image dimensions
        # Focal lengths scale proportionally with image dimensions
        # Principal point also scales proportionally
        scale_x = width / original_width
        scale_y = height / original_height
        
        fx = fx_orig * scale_x
        fy = fy_orig * scale_y
        cx = cx_orig * scale_x
        cy = cy_orig * scale_y
        
        print(f"    Scaled to {width}x{height}: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
        # Validate scaled intrinsics
        cx_expected = width / 2.0
        cy_expected = height / 2.0
        cx_offset = abs(cx - cx_expected) / width if width > 0 else 0
        cy_offset = abs(cy - cy_expected) / height if height > 0 else 0
        
        if cx_offset > 0.15 or cy_offset > 0.15:
            print(f"    ⚠️  WARNING: Principal point far from center!")
            print(f"        Principal point: cx={cx:.1f}, cy={cy:.1f}")
            print(f"        Expected center: cx={cx_expected:.1f}, cy={cy_expected:.1f}")
        
        # Create normalized intrinsics matrix for this view
        K_normalized = torch.eye(3, dtype=torch.float32)
        K_normalized[0, 0] = fx / width   # fx normalized
        K_normalized[1, 1] = fy / height  # fy normalized
        K_normalized[0, 2] = cx / width   # cx normalized
        K_normalized[1, 2] = cy / height  # cy normalized
        
        intrinsics_list.append(K_normalized)
        
        print(f"    Normalized: fx={fx/width:.6f}, fy={fy/height:.6f}, cx={cx/width:.6f}, cy={cy/height:.6f}")
    
    # Stack all intrinsics: [num_views, 3, 3] -> [1, num_views, 3, 3]
    intrinsics = torch.stack(intrinsics_list, dim=0).unsqueeze(0)
    
    # Compute and validate Field of View for each view
    print(f"\n  Field of View (FOV) for each view:")
    for i in range(num_views):
        img_name = image_filenames[i]
        fov = get_fov(intrinsics[0, i:i+1])  # Get FOV for this view
        fov_deg = fov * 180 / math.pi
        print(f"    View {i} ({img_name}):")
        print(f"      Horizontal FOV: {fov_deg[0, 0]:.2f}°")
        print(f"      Vertical FOV: {fov_deg[0, 1]:.2f}°")
        
        # Check if FOV is reasonable (typical range: 30-120 degrees)
        if fov_deg[0, 0] < 20 or fov_deg[0, 0] > 150:
            print(f"      WARNING: Horizontal FOV ({fov_deg[0, 0]:.2f}°) is outside typical range (20-150°)")
        if fov_deg[0, 1] < 20 or fov_deg[0, 1] > 150:
            print(f"      WARNING: Vertical FOV ({fov_deg[0, 1]:.2f}°) is outside typical range (20-150°)")

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
    if camera_distances:
        avg_dist = sum(camera_distances) / len(camera_distances)
        if avg_dist < 0.01:
            issues.append("Camera distances are very small - may cause numerical precision issues")
        if avg_dist > 1000:
            issues.append("Camera distances are very large - may cause numerical precision issues")
    
    # Check FOV for all views
    for i in range(num_views):
        img_name = image_filenames[i]
        fov = get_fov(intrinsics[0, i:i+1])
        fov_deg = fov * 180 / math.pi
        fov_h = fov_deg[0, 0].item()
        if fov_h < 30 or fov_h > 120:
            issues.append(f"View {i} ({img_name}): FOV ({fov_h:.1f}°) is outside typical range")
        elif fov_h < 20:
            issues.append(f"View {i} ({img_name}): FOV ({fov_h:.1f}°) is very narrow")
    
    if issues:
        print("  Potential issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("  No obvious issues detected.")
    
    print("="*70)

    # Prepare visualization dump to capture scales and rotations for PLY export
    visualization_dump = {}

    # Model Optimization Setup
    use_optimization = USE_TENSORRT and (TENSORRT_AVAILABLE or TENSORRT_USE_TORCH_COMPILE)
    trt_model_path = Path(TENSORRT_MODEL_PATH)
    encoder_to_use = encoder
    optimization_name = "PyTorch (Standard)"
    
    if use_optimization:
        # Use torch.compile if requested (more robust for complex models)
        if TENSORRT_USE_TORCH_COMPILE:
            print("\n" + "="*70)
            print("Optimization Setup (torch.compile)")
            print("="*70)
            
            compiled_encoder = compile_with_torch_compile(encoder, backend="inductor")
            
            if compiled_encoder is not None:
                encoder_to_use = compiled_encoder
                optimization_name = "torch.compile (Inductor)"
            else:
                use_optimization = False
                
        # Use TensorRT compilation (requires compatible model)
        elif TENSORRT_AVAILABLE:
            print("\n" + "="*70)
            print("Optimization Setup (TensorRT)")
            print("="*70)
            
            # Check for ONNX-based TensorRT engine first (preferred)
            engine_path = Path(TENSORRT_ENGINE_PATH)
            if TENSORRT_USE_ONNX and engine_path.exists():
                print(f"  Found existing TensorRT engine: {engine_path}")
                print(f"  File size: {engine_path.stat().st_size / (1024*1024):.2f} MB")
                
                try:
                    print("  Loading TensorRT engine...")
                    output_names = ["means", "covariances", "harmonics", "opacities"]
                    engine_wrapper = TensorRTEngineWrapper(str(engine_path), output_names)
                    trt_encoder = TensorRTEngineEncoderWrapper(engine_wrapper)
                    print("  ✓ TensorRT engine loaded successfully!")
                    encoder_to_use = trt_encoder
                    optimization_name = "TensorRT (ONNX)"
                except Exception as e:
                    print(f"  ✗ Failed to load TensorRT engine: {e}")
                    print("  Will try to rebuild...")
                    use_optimization = False
            
            # Check for TorchScript-based TensorRT model
            elif trt_model_path.exists():
                print(f"  Found existing TensorRT model: {trt_model_path}")
                print(f"  File size: {trt_model_path.stat().st_size / (1024*1024):.2f} MB")
                
                try:
                    print("  Loading TensorRT model...")
                    trt_model_raw = torch.jit.load(str(trt_model_path)).cuda().eval()
                    # Wrap the loaded model to match the encoder interface
                    trt_encoder = TensorRTEncoderWrapper(trt_model_raw)
                    print("  ✓ TensorRT model loaded successfully!")
                    encoder_to_use = trt_encoder
                    optimization_name = "TensorRT"
                except Exception as e:
                    print(f"  ✗ Failed to load TensorRT model: {e}")
                    print("  Falling back to standard PyTorch inference")
                    use_optimization = False
            else:
                print(f"  TensorRT model not found: {trt_model_path}")
                print("  Compiling model to TensorRT...")
                
                # Prepare example inputs for compilation
                example_inputs = {
                    "image": context["image"],
                    "extrinsics": context["extrinsics"],
                    "intrinsics": context["intrinsics"],
                    "near": context["near"],
                    "far": context["far"],
                }
                
                trt_encoder = None
                
                # Try ONNX->TensorRT path first (more reliable for complex models)
                if TENSORRT_USE_ONNX:
                    print("  Using ONNX->TensorRT path (more reliable for complex models)...")
                    onnx_path = Path(TENSORRT_ONNX_PATH)
                    engine_path = Path(TENSORRT_ENGINE_PATH)
                    
                    trt_encoder = compile_to_tensorrt_via_onnx(
                        encoder,
                        example_inputs,
                        onnx_path,
                        engine_path,
                        fp16=TENSORRT_FP16,
                    )
                
                # Fall back to TorchScript path if ONNX fails or is disabled
                if trt_encoder is None and not TENSORRT_USE_ONNX:
                    print("  Using TorchScript->TensorRT path...")
                    trt_encoder = compile_to_tensorrt(
                        encoder,
                        example_inputs,
                        trt_model_path,
                        fp16=TENSORRT_FP16,
                    )
                
                if trt_encoder is not None:
                    encoder_to_use = trt_encoder
                    optimization_name = "TensorRT"
                else:
                    print("\n  TensorRT compilation failed. Consider using torch.compile instead:")
                    print("    Set TENSORRT_USE_TORCH_COMPILE = True at the top of inference.py")
                    use_optimization = False
    
    # Benchmark inference if requested
    if BENCHMARK_RUNS > 0:
        # Benchmark original model
        pytorch_stats = benchmark_inference(
            encoder,
            context,
            num_warmup=WARMUP_RUNS,
            num_runs=BENCHMARK_RUNS,
            model_name="PyTorch (Standard)",
        )
        
        # Benchmark optimized model if available
        if use_optimization and encoder_to_use != encoder:
            optimized_stats = benchmark_inference(
                encoder_to_use,
                context,
                num_warmup=WARMUP_RUNS,
                num_runs=BENCHMARK_RUNS,
                model_name=optimization_name,
            )
            
            # Print speedup comparison
            speedup = pytorch_stats["mean"] / optimized_stats["mean"]
            print("\n" + "="*70)
            print("Performance Comparison")
            print("="*70)
            print(f"  PyTorch:   {pytorch_stats['mean']*1000:.2f} ms")
            print(f"  {optimization_name}: {optimized_stats['mean']*1000:.2f} ms")
            print(f"  Speedup:   {speedup:.2f}x faster")
            print("="*70)

    # Run encoder
    print("\n" + "="*70)
    print("Running Encoder Inference")
    print("="*70)
    using_optimized = use_optimization and encoder_to_use != encoder
    print(f"  Using: {optimization_name}")
    
    # Note: For PLY export, we need visualization_dump
    # TensorRT wrapper doesn't populate it, so we need to run PyTorch encoder once
    # torch.compile should populate it correctly
    needs_viz_pass = using_optimized and optimization_name == "TensorRT"
    if needs_viz_pass:
        print("  Note: TensorRT model will be used for main inference")
        print("  Standard PyTorch will run once more to capture visualization data for PLY export")
    
    inference_start = time.perf_counter()
    with torch.no_grad():
        result = encoder_to_use(
            context=context,
            global_step=0,
            deterministic=False,
            visualization_dump=visualization_dump if not needs_viz_pass else {},
            scene_names=None,
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_end = time.perf_counter()
    inference_time = inference_end - inference_start
    
    print(f"  Inference completed in {inference_time*1000:.2f} ms")
    
    # If using TensorRT wrapper, run the original encoder to populate visualization_dump for PLY export
    if needs_viz_pass:
        print("\n  Running PyTorch encoder to capture visualization data for PLY export...")
        with torch.no_grad():
            _ = encoder(
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
            world_rotations_list.append(torch.from_numpy(world_rotations_quat))
        
        # Flatten back to [num_gaussians, 4]
        world_rotations = torch.cat(world_rotations_list, dim=0).to(rotations.device)

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
