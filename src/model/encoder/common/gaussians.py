import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    # The formula is: R @ diag(s^2) @ R^T
    # We can avoid creating the diagonal matrix by using broadcasting:
    # R @ diag(s^2) @ R^T = (R * s^2.unsqueeze(-2)) @ R^T
    # where the multiplication broadcasts s^2 across the rows of R
    
    scale_sq = scale * scale  # [..., 3] - element-wise square
    rotation = quaternion_to_matrix(rotation_xyzw)  # [..., 3, 3]
    
    # Scale each column of R by the corresponding s^2 value
    # rotation has shape [..., 3, 3], scale_sq has shape [..., 3]
    # We want to scale column i by scale_sq[i]
    scaled_rotation = rotation * scale_sq.unsqueeze(-2)  # [..., 3, 3]
    
    # Now compute (R * s^2) @ R^T
    return scaled_rotation @ rearrange(rotation, "... i j -> ... j i")
