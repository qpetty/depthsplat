"""
Helper script for setting up camera parameters for 3-view person photography.

This provides reasonable starting values that you can tweak based on your actual setup.
"""

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


def create_rotation_matrix_y(angle_degrees: float) -> torch.Tensor:
    """Create a rotation matrix around the Y-axis (vertical axis).
    
    Args:
        angle_degrees: Rotation angle in degrees (positive = counterclockwise when looking down -Y)
    
    Returns:
        3x3 rotation matrix
    """
    angle_rad = np.deg2rad(angle_degrees)
    rotation = R.from_euler('y', angle_rad, degrees=False)
    return torch.tensor(rotation.as_matrix(), dtype=torch.float32)


def create_camera_pose(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Create a 4x4 camera-to-world pose matrix.
    
    Args:
        rotation: 3x3 rotation matrix (camera orientation)
        translation: 3D translation vector (camera position in world coordinates)
    
    Returns:
        4x4 homogeneous transformation matrix (camera-to-world)
    """
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def setup_cameras_person_photography(
    width: int = 960,
    height: int = 512,
    camera_distance: float = 2.5,  # Distance from person (in world units)
    left_angle: float = 45.0,      # Left view angle in degrees (try 30-60)
    right_angle: float = -45.0,    # Right view angle in degrees (try -30 to -60)
    fx_pixels: float = 800.0,      # Focal length in pixels (adjust based on your camera)
    fy_pixels: float = None,       # If None, uses fx_pixels
    cx_pixels: float = None,       # If None, uses width/2
    cy_pixels: float = None,       # If None, uses height/2
    batch_size: int = 1,
    num_views: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Set up camera parameters for 3-view person photography.
    
    RECOMMENDED STARTING VALUES:
    - camera_distance: 2.0-4.0 (2.5 is a good starting point)
      * Increase if person appears too large in images
      * Decrease if person appears too small
    - left_angle: 30-60 degrees (45 is a good starting point)
      * Typical portrait photos are not full 90° side views
      * Adjust based on how much side profile you want
    - right_angle: -30 to -60 degrees (symmetric to left)
    - fx_pixels: 400-1200 (800 is reasonable for many cameras)
      * Wide-angle lenses: 400-600
      * Standard lenses: 600-900
      * Telephoto lenses: 900-1200
      * Check your camera's EXIF data for actual focal length
      * For phone cameras: typically 1000-1500 pixels for 1080p images
    
    Returns:
        extrinsics: [batch_size, num_views, 4, 4] camera-to-world poses
        intrinsics: [batch_size, num_views, 3, 3] normalized camera intrinsics
    """
    
    # Set defaults
    if fy_pixels is None:
        fy_pixels = fx_pixels
    if cx_pixels is None:
        cx_pixels = width / 2.0
    if cy_pixels is None:
        cy_pixels = height / 2.0
    
    # View 0: Left view
    # Camera is positioned to the left, rotated to look at the person
    rotation_left = create_rotation_matrix_y(left_angle)
    # Position camera to the left of origin
    # When rotated, camera looks toward origin where person stands
    translation_left = torch.tensor([-camera_distance, 0.0, 0.0], dtype=torch.float32)
    pose_left = create_camera_pose(rotation_left, translation_left)
    
    # View 1: Head-on (center view)
    # Camera looks down +Z axis (standard OpenCV convention)
    rotation_center = torch.eye(3, dtype=torch.float32)
    translation_center = torch.tensor([0.0, 0.0, camera_distance], dtype=torch.float32)
    pose_center = create_camera_pose(rotation_center, translation_center)
    
    # View 2: Right view
    rotation_right = create_rotation_matrix_y(right_angle)
    translation_right = torch.tensor([camera_distance, 0.0, 0.0], dtype=torch.float32)
    pose_right = create_camera_pose(rotation_right, translation_right)
    
    # Stack all poses
    extrinsics = torch.stack([pose_left, pose_center, pose_right], dim=0)
    extrinsics = extrinsics.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 3, 4, 4]
    
    # Normalized intrinsics (same for all views)
    intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    intrinsics = intrinsics.repeat(batch_size, num_views, 1, 1)
    
    # Normalize by image dimensions
    intrinsics[:, :, 0, 0] = fx_pixels / width   # fx normalized
    intrinsics[:, :, 1, 1] = fy_pixels / height  # fy normalized
    intrinsics[:, :, 0, 2] = cx_pixels / width   # cx normalized
    intrinsics[:, :, 1, 2] = cy_pixels / height  # cy normalized
    
    return extrinsics, intrinsics


def print_camera_info(extrinsics: torch.Tensor, intrinsics: torch.Tensor, width: int, height: int):
    """Print camera parameters for debugging."""
    print(f"\n{'='*60}")
    print(f"Camera Setup Summary")
    print(f"{'='*60}")
    
    for i in range(extrinsics.shape[1]):
        pose = extrinsics[0, i]
        position = pose[:3, 3]
        rotation = pose[:3, :3]
        
        # Extract viewing direction (camera's -Z axis in world coordinates)
        # In OpenCV convention, camera looks down +Z in camera space
        # So the viewing direction in world space is the third column of rotation
        view_dir = rotation[:, 2]
        
        print(f"\nView {i}:")
        print(f"  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        print(f"  View direction: [{view_dir[0]:.3f}, {view_dir[1]:.3f}, {view_dir[2]:.3f}]")
        print(f"  Distance from origin: {position.norm():.3f}")
        
        # Extract angle from center
        if i == 0:
            angle = np.rad2deg(np.arctan2(position[0], position[2]))
            print(f"  Angle from center: {angle:.1f}° (left)")
        elif i == 2:
            angle = np.rad2deg(np.arctan2(position[0], position[2]))
            print(f"  Angle from center: {angle:.1f}° (right)")
    
    print(f"\nIntrinsics (normalized):")
    fx_norm = intrinsics[0, 0, 0, 0].item()
    fy_norm = intrinsics[0, 0, 1, 1].item()
    cx_norm = intrinsics[0, 0, 0, 2].item()
    cy_norm = intrinsics[0, 0, 1, 2].item()
    
    print(f"  fx: {fx_norm:.4f} (={fx_norm * width:.1f} pixels)")
    print(f"  fy: {fy_norm:.4f} (={fy_norm * height:.1f} pixels)")
    print(f"  cx: {cx_norm:.4f} (={cx_norm * width:.1f} pixels)")
    print(f"  cy: {cy_norm:.4f} (={cy_norm * height:.1f} pixels)")
    
    # Calculate field of view
    fov_x = 2 * np.arctan(width / (2 * fx_norm * width))
    fov_y = 2 * np.arctan(height / (2 * fy_norm * height))
    print(f"  FOV: {np.rad2deg(fov_x):.1f}° x {np.rad2deg(fov_y):.1f}°")


if __name__ == "__main__":
    # Example usage with recommended starting values
    width, height = 960, 512
    
    print("="*60)
    print("RECOMMENDED STARTING VALUES FOR PERSON PHOTOGRAPHY")
    print("="*60)
    print("\n1. CAMERA DISTANCE: Start with 2.5, adjust between 2.0-4.0")
    print("   - Too close: person appears too large, may be cropped")
    print("   - Too far: person appears too small, lacks detail")
    print("\n2. VIEW ANGLES: Start with ±45°, adjust between 30°-60°")
    print("   - 90° = full side profile (very extreme)")
    print("   - 45° = three-quarter view (common for portraits)")
    print("   - 30° = slight angle (more natural)")
    print("\n3. FOCAL LENGTH: Start with 800 pixels, adjust based on camera")
    print("   - Check EXIF data from your images")
    print("   - Typical phone: 1000-1500 pixels")
    print("   - Typical DSLR: 600-1200 pixels")
    print("\n4. PRINCIPAL POINT: Usually at image center")
    print("   - Only adjust if you know your camera has offset")
    
    print("\n" + "="*60)
    print("Example Setup:")
    print("="*60)
    
    extrinsics, intrinsics = setup_cameras_person_photography(
        width=width,
        height=height,
        camera_distance=2.5,  # GOOD STARTING VALUE
        left_angle=45.0,      # GOOD STARTING VALUE (not 90°)
        right_angle=-45.0,    # GOOD STARTING VALUE (not -90°)
        fx_pixels=800.0,      # ADJUST based on your camera
    )
    
    print_camera_info(extrinsics, intrinsics, width, height)
    
    print("\n" + "="*60)
    print("TROUBLESHOOTING:")
    print("="*60)
    print("If person appears:")
    print("  - Too large/close: Increase camera_distance (try 3.0, 3.5, 4.0)")
    print("  - Too small/far: Decrease camera_distance (try 2.0, 1.5)")
    print("  - Views don't match: Adjust left_angle and right_angle")
    print("  - Distorted: Check focal length matches your camera")
    print("  - Off-center: Adjust cx_pixels and cy_pixels")

