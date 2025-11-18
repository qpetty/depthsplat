from math import isqrt

import torch
from e3nn.o3 import matrix_to_angles, wigner_D
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def rotate_sh(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype

    *_, n = sh_coefficients.shape
    alpha, beta, gamma = matrix_to_angles(rotations)
    
    # MPS doesn't support float64/complex128, so we need to compute wigner_D on CPU
    # if we're on MPS, then move the result back
    compute_device = device
    if device.type == "mps":
        # Move angles to CPU for wigner_D computation
        alpha = alpha.cpu()
        beta = beta.cpu()
        gamma = gamma.cpu()
        compute_device = torch.device("cpu")
    
    result = []
    for degree in range(isqrt(n)):
        with torch.device(compute_device):
            sh_rotations = wigner_D(degree, alpha, beta, gamma).type(dtype)
        
        # Move back to original device if we computed on CPU
        if compute_device != device:
            sh_rotations = sh_rotations.to(device)
        
        sh_rotated = einsum(
            sh_rotations,
            sh_coefficients[..., degree**2 : (degree + 1) ** 2],
            "... i j, ... j -> ... i",
        )
        result.append(sh_rotated)

    return torch.cat(result, dim=-1)


def rotate_sh_coreml(
    sh_coefficients: Float[Tensor, "B V P 3 D"],
    rotations: Float[Tensor, "B V 3 3"]
) -> Float[Tensor, "B V P 3 D"]:
    """
    CoreML-compatible SH rotation (avoiding e3nn/matrix_exp).
    Supports l=1 (Degree 1). Degree 2+ is passed through unrotated for now to ensure
    CoreML compatibility and speed, while fixing the most noticeable linear light rotation.
    
    Args:
        sh_coefficients: [B, V, P, C, D] where P=H*W, C=3(RGB), D=SH_coeffs
        rotations: [B, V, 3, 3]
    """
    # Flatten B and V into a single batch dimension N = B*V to keep rank low
    B, V, P, C, D = sh_coefficients.shape
    N = B * V
    
    # sh_flat: [N, P, C, D] (Rank 4)
    sh_flat = sh_coefficients.reshape(N, P, C, D)
    # rot_flat: [N, 3, 3] (Rank 3)
    rot_flat = rotations.reshape(N, 3, 3)
    
    # Use torch.clone() to ensure we don't modify in place in a way that upsets autograd/tracing
    # However, for in-place assignment to work reliably with symbolic tracing, we sometimes need
    # to be careful.
    sh_out = sh_flat.clone()
    
    # --- Degree 1 (Indices 1, 2, 3) ---
    if D >= 4:
        # e3nn basis for l=1 is (y, z, x)
        # We extract components, rotate them as (x, y, z), and put them back.
        
        # Extract current (y, z, x)
        sh1_y = sh_flat[..., 1]
        sh1_z = sh_flat[..., 2]
        sh1_x = sh_flat[..., 3]
        
        # Stack as (x, y, z) -> [N, P, C, 3]
        xyz = torch.stack([sh1_x, sh1_y, sh1_z], dim=-1)
        
        # Reshape to [N, P*C, 3] for batch matrix multiply
        xyz_flat_for_matmul = xyz.reshape(N, P*C, 3)
        
        # Rotate: xyz_new = xyz @ R.T
        # [N, P*C, 3] @ [N, 3, 3].transpose(1, 2) -> [N, P*C, 3] @ [N, 3, 3] -> [N, P*C, 3]
        # Note: rot_flat is [N, 3, 3]
        R_T = rot_flat.transpose(1, 2)
        
        xyz_new_flat = torch.matmul(xyz_flat_for_matmul, R_T)
        
        # Reshape back to [N, P, C, 3]
        xyz_new = xyz_new_flat.reshape(N, P, C, 3)
        
        # Unpack new (x, y, z)
        x_new = xyz_new[..., 0]
        y_new = xyz_new[..., 1]
        z_new = xyz_new[..., 2]
        
        # Assign back as (y, z, x) using slices
        # In symbolic tracing, slicing like sh_out[..., 1] = ... can be tricky if shapes aren't static.
        # Instead, we construct the new tensor by concatenating parts.
        # This is safer for export.
        
        # Parts:
        # 0: DC (unchanged)
        # 1: y_new
        # 2: z_new
        # 3: x_new
        # 4+: Higher degrees (unchanged)
        
        parts = []
        # sh_flat is [N, P, C, D]
        
        # DC component [N, P, C, 1]
        parts.append(sh_flat[..., 0:1]) 
        
        # Rotated components need to be [N, P, C, 1]
        # y_new, z_new, x_new are [N, P, C]
        parts.append(y_new.unsqueeze(-1)) 
        parts.append(z_new.unsqueeze(-1))
        parts.append(x_new.unsqueeze(-1))
        
        if D > 4:
             parts.append(sh_flat[..., 4:]) # Higher degrees [N, P, C, D-4]
             
        sh_out = torch.cat(parts, dim=-1)

    # --- Degree 2 (Indices 4-8) ---
    # Skipped for CoreML speed/compatibility/stability.
    # l=1 covers the dominant directional lighting.
    
    # Reshape back to original 5D shape
    return sh_out.reshape(B, V, P, C, D)


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    from e3nn.o3 import spherical_harmonics
    from matplotlib import cm
    from scipy.spatial.transform.rotation import Rotation as R

    device = torch.device("cuda")

    # Generate random spherical harmonics coefficients.
    degree = 4
    coefficients = torch.rand((degree + 1) ** 2, dtype=torch.float32, device=device)

    def plot_sh(sh_coefficients, path: Path) -> None:
        phi = torch.linspace(0, torch.pi, 100, device=device)
        theta = torch.linspace(0, 2 * torch.pi, 100, device=device)
        phi, theta = torch.meshgrid(phi, theta, indexing="xy")
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        xyz = torch.stack([x, y, z], dim=-1)
        sh = spherical_harmonics(list(range(degree + 1)), xyz, True)
        result = einsum(sh, sh_coefficients, "... n, n -> ...")
        result = (result - result.min()) / (result.max() - result.min())

        # Set the aspect ratio to 1 so our sphere looks spherical
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            x.cpu().numpy(),
            y.cpu().numpy(),
            z.cpu().numpy(),
            rstride=1,
            cstride=1,
            facecolors=cm.seismic(result.cpu().numpy()),
        )
        # Turn off the axis planes
        ax.set_axis_off()
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(path)

    for i, angle in enumerate(torch.linspace(0, 2 * torch.pi, 30)):
        rotation = torch.tensor(
            R.from_euler("x", angle.item()).as_matrix(), device=device
        )
        plot_sh(rotate_sh(coefficients, rotation), Path(f"sh_rotation/{i:0>3}.png"))

    print("Done!")
