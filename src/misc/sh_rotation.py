from math import isqrt

import torch
try:
    from e3nn.o3 import matrix_to_angles, wigner_D
    from einops import einsum
except ImportError:
    pass # Handle missing imports for CoreML environment if needed
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


def _construct_l2_rotation_matrix(R: Tensor) -> Tensor:
    """
    Constructs the 5x5 Wigner-D matrix for l=2 from a 3x3 rotation matrix R.
    Uses the Kronecker product method for the real SH basis.
    R: [N, 3, 3] (Rotation on [x, y, z])
    Returns: [N, 5, 5]
    """
    N = R.shape[0]
    device = R.device
    
    # 1. Convert R to the basis used by e3nn l=1: [y, z, x]
    # The permutation P maps [x, y, z] -> [y, z, x]
    # R_new = P @ R @ P.T
    # This corresponds to shuffling rows and columns indices: 0->2, 1->0, 2->1 ?
    # x is idx 0, y is idx 1, z is idx 2.
    # Target: y(1), z(2), x(0).
    # So we want rows/cols 1, 2, 0.
    
    idx = torch.tensor([1, 2, 0], device=device)
    # R_yzx = R[:, idx][:, :, idx] # Broad slicing can be tricky with trace
    
    # Manual shuffle to be safe
    # R has shape [N, 3, 3]
    R_yzx = R.index_select(1, idx).index_select(2, idx)
    
    # 2. Compute Kronecker product K = R_yzx (x) R_yzx
    # Shape [N, 9, 9]
    # K_ij,kl = R_ik * R_jl
    # We can use reshaping:
    # R: [N, 3, 3]
    # K: [N, 3, 3, 3, 3] -> [N, 9, 9]
    
    # Outer product of R with itself
    # [N, 3, 1, 3, 1] * [N, 1, 3, 1, 3] -> [N, 3, 3, 3, 3]
    K = R_yzx.unsqueeze(2).unsqueeze(4) * R_yzx.unsqueeze(1).unsqueeze(3)
    K = K.reshape(N, 9, 9)
    
    # 3. Define Basis transformation matrix C (5x9)
    # Mapping from u(x)u basis (y^2, yz, yx, ...) to SH basis
    # Basis: [y, z, x]
    # Kron: y^2(0), yz(1), yx(2), zy(3), z^2(4), zx(5), xy(6), xz(7), x^2(8)
    #
    # SH Basis (e3nn l=2):
    # 0 (xy): 0.5*sqrt(3) * (yx + xy) -> indices 2, 6
    # 1 (yz): 0.5*sqrt(3) * (yz + zy) -> indices 1, 3
    # 2 (z^2): z^2 - 0.5(x^2+y^2) -> 1.0*4 - 0.5*8 - 0.5*0
    # 3 (xz): 0.5*sqrt(3) * (zx + xz) -> indices 5, 7
    # 4 (x^2-y^2): 0.5*sqrt(3) * (x^2 - y^2) -> 0.5*sqrt(3)*8 - 0.5*sqrt(3)*0
    
    # Note: we use 0.5*... because xy appears twice in the symmetric tensor u(x)u.
    # To project symmetric basis, we sum the components.
    
    sqrt3 = 1.73205080757
    sqrt3_2 = sqrt3 / 2.0
    
    # Create C matrix [5, 9]
    # We build it on CPU/device once
    C = torch.zeros(5, 9, device=device, dtype=R.dtype)
    
    # 0: xy -> indices 2 (yx) and 6 (xy)
    C[0, 2] = sqrt3_2
    C[0, 6] = sqrt3_2
    
    # 1: yz -> indices 1 (yz) and 3 (zy)
    C[1, 1] = sqrt3_2
    C[1, 3] = sqrt3_2
    
    # 2: 3z^2-1 -> z^2(4) - 0.5*x^2(8) - 0.5*y^2(0)
    C[2, 4] = 1.0
    C[2, 8] = -0.5
    C[2, 0] = -0.5
    
    # 3: xz -> indices 5 (zx) and 7 (xz)
    C[3, 5] = sqrt3_2
    C[3, 7] = sqrt3_2
    
    # 4: x^2-y^2 -> 0.5*sqrt(3)*x^2(8) - 0.5*sqrt(3)*y^2(0)
    C[4, 8] = sqrt3_2
    C[4, 0] = -sqrt3_2
    
    # 4. Compute D = (1/1.5) * C @ K @ C.T
    # Scale factor 1.5 comes from norm of basis vectors in product space
    scale = 1.0 / 1.5
    
    # [5, 9] @ [N, 9, 9] -> [N, 5, 9]
    # Use matmul with broadcasting
    # C is [5, 9], K is [N, 9, 9]
    # We want C K C^T
    
    # Reshape C for broadcast: [1, 5, 9]
    C_broad = C.unsqueeze(0)
    
    # Temp = C @ K
    # [1, 5, 9] @ [N, 9, 9] -> [N, 5, 9]
    Temp = torch.matmul(C_broad, K)
    
    # Result = Temp @ C.T
    # [N, 5, 9] @ [1, 9, 5] -> [N, 5, 5]
    D = torch.matmul(Temp, C_broad.transpose(1, 2))
    
    return D * scale


def rotate_sh_coreml(
    sh_coefficients: Float[Tensor, "B V P 3 D"],
    rotations: Float[Tensor, "B V 3 3"]
) -> Float[Tensor, "B V P 3 D"]:
    """
    CoreML-compatible SH rotation (avoiding e3nn/matrix_exp).
    Supports l=1 and l=2.
    
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
    
    # --- Degree 1 (Indices 1, 2, 3) ---
    # Use torch.clone() to ensure we don't modify in place
    sh_out_parts = []
    
    # 0: DC (unchanged)
    sh_out_parts.append(sh_flat[..., 0:1])
    
    if D >= 4:
        # l=1: indices 1, 2, 3
        # e3nn basis for l=1 is (y, z, x)
        sh1_y = sh_flat[..., 1]
        sh1_z = sh_flat[..., 2]
        sh1_x = sh_flat[..., 3]
        
        # Stack as (x, y, z) -> [N, P, C, 3]
        xyz = torch.stack([sh1_x, sh1_y, sh1_z], dim=-1)
        
        # Reshape to [N, P*C, 3] for batch matrix multiply
        xyz_flat_for_matmul = xyz.reshape(N, P*C, 3)
        
        # Rotate: xyz_new = xyz @ R.T
        # [N, P*C, 3] @ [N, 3, 3].transpose(1, 2) -> [N, P*C, 3]
        R_T = rot_flat.transpose(1, 2)
        xyz_new_flat = torch.matmul(xyz_flat_for_matmul, R_T)
        
        # Reshape back to [N, P, C, 3]
        xyz_new = xyz_new_flat.reshape(N, P, C, 3)
        
        # Unpack new (x, y, z)
        x_new = xyz_new[..., 0]
        y_new = xyz_new[..., 1]
        z_new = xyz_new[..., 2]
        
        sh_out_parts.append(y_new.unsqueeze(-1)) 
        sh_out_parts.append(z_new.unsqueeze(-1))
        sh_out_parts.append(x_new.unsqueeze(-1))
        
        # --- Degree 2 (Indices 4, 5, 6, 7, 8) ---
        if D >= 9:
            # Compute 5x5 Wigner D matrix for l=2
            # D2: [N, 5, 5]
            D2 = _construct_l2_rotation_matrix(rot_flat)
            
            # Extract l=2 coefficients: [N, P, C, 5]
            sh2 = sh_flat[..., 4:9]
            
            # Flatten for matmul: [N, P*C, 5]
            sh2_flat = sh2.reshape(N, P*C, 5)
            
            # Apply rotation: sh2_new = sh2 @ D2.T
            # [N, P*C, 5] @ [N, 5, 5].T -> [N, P*C, 5]
            sh2_new_flat = torch.matmul(sh2_flat, D2.transpose(1, 2))
            
            # Reshape back
            sh2_new = sh2_new_flat.reshape(N, P, C, 5)
            
            sh_out_parts.append(sh2_new)
            
            # Higher degrees
            if D > 9:
                sh_out_parts.append(sh_flat[..., 9:])
        elif D > 4:
             sh_out_parts.append(sh_flat[..., 4:])
             
    elif D > 1:
        sh_out_parts.append(sh_flat[..., 1:])

    sh_out = torch.cat(sh_out_parts, dim=-1)

    # Reshape back to original 5D shape
    return sh_out.reshape(B, V, P, C, D)


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    # from e3nn.o3 import spherical_harmonics # Avoid import if testing coreml logic without e3nn
    from matplotlib import cm
    from scipy.spatial.transform.rotation import Rotation as R

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Test the CoreML rotation vs e3nn if available
    try:
        from e3nn import o3
        print("Testing l=2 rotation consistency...")
        
        # Create a random rotation
        rot_np = R.random().as_matrix()
        rot_torch = torch.tensor(rot_np, dtype=torch.float32, device=device).unsqueeze(0) # [1, 3, 3]
        
        # Create random SH coeffs (l=2 only)
        # 1 batch, 1 view, 1 pixel, 1 channel, 9 coeffs
        coeffs = torch.randn(1, 1, 1, 1, 9, device=device)
        
        # Rotate using CoreML logic
        coeffs_rot_coreml = rotate_sh_coreml(coeffs, rot_torch.unsqueeze(0))
        
        # Rotate using e3nn logic (via rotate_sh)
        # Note: rotate_sh expects [batch, n] coeffs and [batch, 3, 3] rot
        coeffs_flat = coeffs.reshape(1, 9)
        rot_flat = rot_torch
        coeffs_rot_e3nn = rotate_sh(coeffs_flat, rot_flat)
        
        diff = (coeffs_rot_coreml.reshape(1, 9) - coeffs_rot_e3nn).abs().max().item()
        print(f"Max difference between CoreML and e3nn rotation: {diff:.6f}")
        
        if diff < 1e-5:
            print("PASS: Rotation matches e3nn!")
        else:
            print("FAIL: Rotation mismatch.")
            
    except ImportError:
        print("e3nn not found, skipping consistency test.")

    print("Done!")
