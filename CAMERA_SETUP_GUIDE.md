# Camera Parameter Setup Guide for 3-View Person Photography

## Critical Issues in Your Current Code

### 1. **Camera Distance = 0.0** ❌
This is the biggest problem! With `camera_distance = 0.0`, all three cameras are positioned at the origin (where the person stands), which will produce invalid or identical images.

**Fix:** Use `camera_distance = 2.5` as a starting value (adjust between 2.0-4.0 based on your images).

### 2. **Rotation Angles = 90°** ⚠️
90-degree angles create extreme side profiles. Most portrait photography uses smaller angles.

**Fix:** Start with `±45°` instead of `±90°` for more natural three-quarter views.

### 3. **Camera Positioning Logic** ⚠️
Your current setup has the cameras positioned incorrectly relative to their rotations.

**Fix:** See the corrected code below.

## Recommended Starting Values

```python
camera_distance = 2.5   # Distance from person (2.0-4.0 range)
left_angle = 45.0       # Left view angle (30-60 degrees)
right_angle = -45.0     # Right view angle (-30 to -60 degrees)
fx_pixels = 800.0       # Focal length (adjust based on your camera)
```

## Corrected Code

Here's the corrected version of your camera setup:

```python
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

def create_rotation_matrix_y(angle_degrees: float) -> torch.Tensor:
    """Create rotation matrix around Y-axis."""
    angle_rad = np.deg2rad(angle_degrees)
    rotation = R.from_euler('y', angle_rad, degrees=False)
    return torch.tensor(rotation.as_matrix(), dtype=torch.float32)

def create_camera_pose(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Create 4x4 camera-to-world pose matrix."""
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose

# RECOMMENDED STARTING VALUES
camera_distance = 2.5   # CHANGE FROM 0.0!
left_angle = 45.0       # CHANGE FROM 90.0!
right_angle = -45.0     # CHANGE FROM -90.0!

# View 0: Left view
# Camera positioned to the left, rotated to look at person
rotation_left = create_rotation_matrix_y(left_angle)
translation_left = torch.tensor([-camera_distance, 0.0, 0.0], dtype=torch.float32)
pose_left = create_camera_pose(rotation_left, translation_left)

# View 1: Head-on (camera looks down +Z axis)
rotation_center = torch.eye(3, dtype=torch.float32)
translation_center = torch.tensor([0.0, 0.0, camera_distance], dtype=torch.float32)
pose_center = create_camera_pose(rotation_center, translation_center)

# View 2: Right view
rotation_right = create_rotation_matrix_y(right_angle)
translation_right = torch.tensor([camera_distance, 0.0, 0.0], dtype=torch.float32)
pose_right = create_camera_pose(rotation_right, translation_right)

# Stack all poses
extrinsics = torch.stack([pose_left, pose_center, pose_right], dim=0).unsqueeze(0)

# Intrinsics (normalized)
fx, fy = 800.0, 800.0  # Adjust based on your camera
cx, cy = width / 2.0, height / 2.0

intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1)
intrinsics[:, :, 0, 0] = fx / width   # fx normalized
intrinsics[:, :, 1, 1] = fy / height  # fy normalized
intrinsics[:, :, 0, 2] = cx / width   # cx normalized
intrinsics[:, :, 1, 2] = cy / height  # cy normalized
```

## Parameter Tuning Guide

### Camera Distance (`camera_distance`)
- **Too small (1.0-1.5):** Person appears too large, may be cropped
- **Good range (2.0-3.0):** Person fills frame nicely
- **Too large (4.0+):** Person appears too small, lacks detail

**How to adjust:** Look at your images. If the person's head is cut off or too large, increase distance. If too small, decrease.

### View Angles (`left_angle`, `right_angle`)
- **90°:** Full side profile (very extreme, rarely used in portraits)
- **60°:** Strong three-quarter view
- **45°:** Standard three-quarter view (recommended starting point)
- **30°:** Subtle angle (more natural)

**How to adjust:** Match the actual angles in your photos. If your "left" photo shows more profile than expected, increase the angle.

### Focal Length (`fx_pixels`, `fy_pixels`)
- **Phone cameras (1080p):** Typically 1000-1500 pixels
- **DSLR cameras:** Varies by lens (400-1200 pixels common)
- **Wide-angle:** 400-600 pixels
- **Standard:** 600-900 pixels
- **Telephoto:** 900-1200 pixels

**How to find:** Check EXIF data from your images using:
```python
from PIL import Image
from PIL.ExifTags import TAGS

img = Image.open('your_image.jpg')
exif = img._getexif()
# Look for focal length in EXIF data
```

Or use a tool like `exiftool`:
```bash
exiftool your_image.jpg | grep "Focal Length"
```

### Principal Point (`cx`, `cy`)
- **Usually:** Image center (width/2, height/2)
- **Only adjust if:** Your camera has known offset (rare)

## Common Issues and Solutions

### Issue: Person appears at wrong scale
**Solution:** Adjust `camera_distance`. Increase if too large, decrease if too small.

### Issue: Views don't match photo angles
**Solution:** Adjust `left_angle` and `right_angle` to match the actual viewing angles in your photos.

### Issue: Distorted appearance
**Solution:** Verify `fx_pixels` matches your camera's actual focal length from EXIF data.

### Issue: Person appears off-center
**Solution:** Adjust `cx` and `cy` if you know your camera has offset principal point.

## Testing Your Setup

1. **Start with recommended values:**
   - `camera_distance = 2.5`
   - `left_angle = 45.0`, `right_angle = -45.0`
   - `fx = 800.0` (or from your camera EXIF)

2. **Render/test your setup** and observe:
   - Does the person appear at the right scale?
   - Do the viewing angles match your photos?
   - Is there any distortion?

3. **Iteratively adjust:**
   - Scale issues → adjust `camera_distance`
   - Angle issues → adjust `left_angle`/`right_angle`
   - Distortion → adjust `fx`/`fy`

## Using the Helper Script

Run `camera_setup_helper.py` to generate camera parameters with recommended starting values:

```python
from camera_setup_helper import setup_cameras_person_photography, print_camera_info

extrinsics, intrinsics = setup_cameras_person_photography(
    width=960,
    height=512,
    camera_distance=2.5,  # Start here
    left_angle=45.0,      # Start here
    right_angle=-45.0,    # Start here
    fx_pixels=800.0,      # Adjust based on your camera
)

print_camera_info(extrinsics, intrinsics, width=960, height=512)
```

## Coordinate System Notes

This codebase uses **OpenCV convention**:
- **X-axis:** Right
- **Y-axis:** Down (or up, depending on convention)
- **Z-axis:** Forward (camera looks down +Z)
- **Camera position:** Translation vector in world coordinates
- **Camera orientation:** Rotation matrix (camera-to-world)

The extrinsics matrix is a **camera-to-world** transformation (4x4 homogeneous matrix).

