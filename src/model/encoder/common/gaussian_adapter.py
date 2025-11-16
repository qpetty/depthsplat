from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F

from ....geometry.projection import get_world_rays, _invert_camera_intrinsics
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance

# Global flag to skip operations that are incompatible with export/CoreML
# Set this to True before calling torch.export.export() or similar
_SKIP_EXPORT_INCOMPATIBLE_OPS = False


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"] | None,
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
    ) -> Gaussians:
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        scales = torch.clamp(F.softplus(scales - 4.),
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
            )

        assert input_images is not None

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        
        # Check if we need to reduce dimensions to avoid 6D+ tensors (CoreML limit is 5D)
        is_export_mode = torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or _SKIP_EXPORT_INCOMPATIBLE_OPS
        
        # DEBUG: Print export mode status
        print(f"[GaussianAdapter] Export mode check:")
        print(f"  torch.onnx.is_in_onnx_export() = {torch.onnx.is_in_onnx_export()}")
        print(f"  torch.jit.is_tracing() = {torch.jit.is_tracing()}")
        print(f"  _SKIP_EXPORT_INCOMPATIBLE_OPS = {_SKIP_EXPORT_INCOMPATIBLE_OPS}")
        print(f"  is_export_mode = {is_export_mode}")
        print(f"  opacities.shape = {opacities.shape}")
        print(f"  opacities.ndim = {opacities.ndim}")
        
        # CRITICAL: ALWAYS squeeze dimensions when in export mode
        # The ndim check doesn't work reliably during torch.export tracing (symbolic shapes)
        # Since this model always produces 5D+ tensors, we just squeeze unconditionally during export
        if is_export_mode:
            print("[GaussianAdapter] SQUEEZING dimensions for export!")
            # CoreML Export Mode: Reduce tensor dimensions to avoid 6D+ tensors
            # ================================================================
            # CoreML has a HARD LIMIT of 5 dimensions. The original code creates 6D and 7D tensors:
            #   - broadcast_to((*opacities.shape, 3, d_sh)) creates [B,V,H*W,1,1,3,d_sh] (7D!) ❌
            #   - rearrange(images, "b v c h w -> b v (h w) () () c") creates [B,V,H*W,1,1,C] (6D!) ❌
            #
            # Solution: Squeeze singleton dimensions (from num_surfaces=1 and padding)
            #   - Target shapes stay at 5D or less ✓
            #   - Exception: extrinsics/intrinsics keep ONE spatial singleton for broadcasting ✓
            #
            # Input shapes (with num_surfaces=1 and padding):
            # opacities: [B, V, H*W, srf=1, 1, 1] (6D)
            # sh: [B, V, H*W, srf=1, 1, 3, d_sh] (7D!)
            # scales: [B, V, H*W, srf=1, 1, 3] (6D)
            # rotations: [B, V, H*W, srf=1, 1, 4] (6D)
            # extrinsics: [B, V, 1, 1, 1, 4, 4] (7D!)
            # intrinsics: [B, V, 1, 1, 1, 3, 3] (7D!)
            
            # Remove singleton dimensions carefully - don't remove B or V dimensions
            # We only want to remove the singleton dims from num_surfaces and padding
            # 
            # opacities: [B, V, H*W, srf=1, 1, 1] -> [B, V, H*W]
            # Squeeze from the right side only to preserve B and V
            original_ndim = opacities.dim()
            target_ndim = 3  # We want [B, V, H*W]
            for _ in range(original_ndim - target_ndim):
                opacities = opacities.squeeze(-1)
            
            # scales: [B, V, H*W, 1, 1, 3] -> [B, V, H*W, 3]
            # Squeeze singleton dims between H*W and 3
            original_scales_ndim = scales.dim()
            target_scales_ndim = 4  # [B, V, H*W, 3]
            for _ in range(original_scales_ndim - target_scales_ndim):
                scales = scales.squeeze(-2)
            
            # rotations: [B, V, H*W, 1, 1, 4] -> [B, V, H*W, 4]
            # Squeeze singleton dims between H*W and 4
            original_rotations_ndim = rotations.dim()
            target_rotations_ndim = 4  # [B, V, H*W, 4]
            for _ in range(original_rotations_ndim - target_rotations_ndim):
                rotations = rotations.squeeze(-2)
            
            print(f"[GaussianAdapter] After squeezing scales and rotations:")
            print(f"  scales.shape = {scales.shape}, ndim = {scales.ndim}")
            print(f"  rotations.shape = {rotations.shape}, ndim = {rotations.ndim}")
            
            # sh: [B, V, H*W, 1, 1, 3, d_sh] -> [B, V, H*W, 3, d_sh]
            # Squeeze the singleton dims between H*W and 3
            # Need to squeeze 2 singleton dimensions
            original_sh_ndim = sh.dim()
            target_sh_ndim = 5  # [B, V, H*W, 3, d_sh]
            for _ in range(original_sh_ndim - target_sh_ndim):
                # Squeeze from position -3 which is between spatial dims and feature dims
                sh = sh.squeeze(-3)
            
            # Broadcast: sh to [B, V, H*W, 3, d_sh] using opacities shape [B, V, H*W]
            target_shape = (*opacities.shape, 3, self.d_sh)
            # Verify shapes are compatible before broadcasting
            if sh.shape != target_shape:
                sh = sh.broadcast_to(target_shape)
            sh = sh * self.sh_mask
            
            if input_images is not None:
                # [B, V, C, H, W] -> [B, V, H*W, C] (4D)
                imgs = rearrange(input_images, "b v c h w -> b v (h w) c")
                sh[..., 0] = sh[..., 0] + RGB2SH(imgs)
            
            # Update extrinsics, intrinsics, coordinates, depths to remove singleton dims
            # extrinsics: [B, V, 1, 1, 1, 4, 4] (7D) -> [B, V, 1, 4, 4] (5D)
            # Keep ONE singleton for spatial broadcasting (needed for c2w_rotations @ covariances)
            original_extrinsics_ndim = extrinsics.dim()
            target_extrinsics_ndim = 5  # [B, V, 1, 4, 4] - keep one spatial singleton
            for _ in range(original_extrinsics_ndim - target_extrinsics_ndim):
                extrinsics = extrinsics.squeeze(-3)
            
            # intrinsics: [B, V, 1, 1, 1, 3, 3] (7D) -> [B, V, 1, 3, 3] (5D)
            # Keep ONE singleton for spatial broadcasting
            if intrinsics is not None:
                original_intrinsics_ndim = intrinsics.dim()
                target_intrinsics_ndim = 5  # [B, V, 1, 3, 3] - keep one spatial singleton
                for _ in range(original_intrinsics_ndim - target_intrinsics_ndim):
                    intrinsics = intrinsics.squeeze(-3)
            
            # coordinates: [B, V, H*W, srf=1, 1, 2] (6D) -> [B, V, H*W, 2] (4D)
            original_coordinates_ndim = coordinates.dim()
            target_coordinates_ndim = 4  # [B, V, H*W, 2]
            for _ in range(original_coordinates_ndim - target_coordinates_ndim):
                coordinates = coordinates.squeeze(-2)
            
            # depths: [B, V, H*W, srf=1, 1] (5D) -> [B, V, H*W] (3D)
            if depths is not None:
                original_depths_ndim = depths.dim()
                target_depths_ndim = 3  # [B, V, H*W]
                for _ in range(original_depths_ndim - target_depths_ndim):
                    depths = depths.squeeze(-1)
                
            # Note: We don't add back singleton dimensions - downstream code and the
            # Gaussians dataclass should work with the squeezed shapes
        else:
            # Normal mode: use original logic with higher-dimensional tensors
            print("[GaussianAdapter] NOT in export mode - using original logic (WARNING: may create 6D+ tensors!)")
            sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

            if input_images is not None:
                # [B, V, H*W, 1, 1, 3]
                imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
                # init sh with input images
                sh[..., 0] = sh[..., 0] + RGB2SH(imgs)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        rotated_harmonics = sh  # Identity rotation
        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or _SKIP_EXPORT_INCOMPATIBLE_OPS:
            # Export: Skip SH rotation (use unrotated harmonics; minimal impact for tracing)
            # This avoids matrix_exp which decomposes to unsupported diag operations in CoreML
            print("Export mode: Skipping rotate_sh (uses matrix_exp with unsupported diag operations)")
        else:
            rotated_harmonics = rotate_sh(sh, c2w_rotations[..., None, :, :])

        # Handle rotations broadcast
        # In export mode with squeezed tensors:
        #   scales.shape is [B, V, H*W, 3] (4D)
        #   scales.shape[:-1] is [B, V, H*W] (3D)
        #   rotations is [B, V, H*W, 4] (4D)
        #   broadcast_to [B, V, H*W, 4] is a no-op ✓
        # In normal mode:
        #   scales.shape is [B, V, H*W, srf, 1, 3] (6D)
        #   scales.shape[:-1] is [B, V, H*W, srf, 1] (5D)
        #   rotations is [B, V, H*W, srf, 1, 4] (6D)
        #   broadcast_to [B, V, H*W, srf, 1, 4] is a no-op ✓
        
        print(f"[GaussianAdapter] Before rotations broadcast:")
        print(f"  scales.shape = {scales.shape}, ndim = {scales.ndim}")
        print(f"  rotations.shape = {rotations.shape}, ndim = {rotations.ndim}")
        print(f"  target shape for broadcast: {(*scales.shape[:-1], 4)}")
        
        rotations_broadcast = rotations.broadcast_to((*scales.shape[:-1], 4))
        
        print(f"  rotations_broadcast.shape = {rotations_broadcast.shape}, ndim = {rotations_broadcast.ndim}")

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotated_harmonics,
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations_broadcast,
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        # NOTE: Use analytical intrinsics inverse so that ONNX export does not
        # introduce aten.linalg_inv_ex.
        intrinsics_inv = _invert_camera_intrinsics(intrinsics)
        # Replace einsum with matmul for CoreML compatibility
        # einsum("... i j, j -> ... i") is equivalent to matmul
        xy_multipliers = multiplier * torch.matmul(intrinsics_inv, pixel_size.unsqueeze(-1)).squeeze(-1)
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0
