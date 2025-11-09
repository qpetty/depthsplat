import numpy as np
from scipy.spatial.transform import Rotation as R
# Assuming read_write_model.py is in the same directory
#from read_write_model import read_cameras_binary, read_images_binary, read_points3d_binary
from read_write_model import read_cameras_binary, read_images_binary

# Define the path to your sparse model directory
model_path = "/Users/quinton/Desktop/colmap_output/sparse/0/"
image_names_to_print = ["frame_0002.png", "frame_0035.png", "frame_0070.png", "frame_0105.png", "frame_0140.png"]

# Read the binary files
cameras = read_cameras_binary(model_path + "cameras.bin")
images = read_images_binary(model_path + "images.bin")
#points3D = read_points3d_binary(model_path + "points3D.bin")

for ndx in [1]:
# Access data (e.g., camera with ID 1 intrinsics, image with ID 1 extrinsics)
    camera_1_params = cameras[ndx].params
    image_1_qvec = images[ndx].qvec
    image_1_tvec = images[ndx].tvec

print(f"Camera 1 parameters: {camera_1_params}")
print(f"Image 1 translation vector: {image_1_tvec}")

# Get the camera object for Camera ID 1 (assuming it exists)
camera_id_to_print = 1
if camera_id_to_print in cameras:
    camera = cameras[camera_id_to_print]

    print(f"--- Camera ID {camera.id} Intrinsics ---")
    print(f"  Model:  {camera.model}")
    print(f"  Width:  {camera.width} pixels")
    print(f"  Height: {camera.height} pixels")
    print(f"  Params: {camera.params}") # This is a numpy array

    # For better legibility, you can print each parameter with a descriptive label
    # The parameters vary by model (e.g., PINHOLE has fx, fy, cx, cy)
    if camera.model == 'PINHOLE' and len(camera.params) == 4:
        fx, fy, cx, cy = camera.params
        print("\nDetailed Parameters (PINHOLE model):")
        print(f"  Focal Length X (fx): {fx:.4f}")
        print(f"  Focal Length Y (fy): {fy:.4f}")
        print(f"  Principal Point X (cx): {cx:.4f}")
        print(f"  Principal Point Y (cy): {cy:.4f}")

    # You can also construct and print the K matrix legibly
    K = np.array([
        [camera.params[0], 0, camera.params[2]],
        [0, camera.params[1], camera.params[3]],
        [0, 0, 1]
    ])
    print("\nIntrinsic Matrix (K):")
    # Using numpy to print with specific precision and formatting
    np.set_printoptions(precision=4, suppress=True)
    print(K)
#    np.set_printoptions(10, edgeitems=3, linewidth=75, suppress=False, nanval='nan', infval='inf', negative_infval='-inf') # Reset to default if needed

else:
    print(f"Camera ID {camera_id_to_print} not found in the reconstruction.")

print(f"\n--- Extrinsics for Specified Images ---")
np.set_printoptions(precision=4, suppress=True) # For legible numpy output

## Iterate through all registered images to find the specified ones
#for img_id, img_data in images.items():
#    if img_data.name in image_names_to_print:
#        print(f"\nImage Name: {img_data.name} (Image ID: {img_id})")
#        
#        # 1. Access the quaternion and translation vector
#        qvec = img_data.qvec
#        tvec = img_data.tvec
#
#        print(f"  Quaternion (QW, QX, QY, QZ): {qvec}")
#        print(f"  Translation Vector (TX, TY, TZ): {tvec}")
#
#        # 2. Convert to a 4x4 Extrinsic Matrix (World-to-Camera pose)
#        # COLMAP provides R and t such that X_camera = R @ X_world + t
#        rotation_matrix = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix() # Note order is QX, QY, QZ, QW for scipy input
#
#        # Create the 4x4 extrinsic matrix [R | t; 0 0 0 | 1]
#        extrinsic_matrix_world_to_cam = np.eye(4)
#        extrinsic_matrix_world_to_cam[:3, :3] = rotation_matrix
#        extrinsic_matrix_world_to_cam[:3, 3] = tvec
#
#        print("\n  4x4 World-to-Camera Extrinsic Matrix:")
#        print(extrinsic_matrix_world_to_cam)
#
#        # 3. (Optional) Get the Camera Position in World Coordinates
#        # The camera center C = -R_transpose @ t
#        camera_center_world = -rotation_matrix.T @ tvec
#        print(f"\n  Camera Center Position (World Coords): {camera_center_world}")

# Iterate through all registered images to find the specified ones
extrinsics_list = []  # Store C2W matrices for EXTRINSICS_HARDCODED

for img_id, img_data in images.items():
    if img_data.name in image_names_to_print:
        print(f"\nImage Name: {img_data.name} (Image ID: {img_id})")
        
        # 1. Access the quaternion and translation vector
        qvec = img_data.qvec  # COLMAP format: [QW, QX, QY, QZ]
        tvec = img_data.tvec  # Translation vector [TX, TY, TZ]
        
        print(f"  Quaternion (QW, QX, QY, QZ): {qvec}")
        print(f"  Translation Vector (TX, TY, TZ): {tvec}")
        
        # 2. Convert quaternion to rotation matrix
        # COLMAP quaternion is [QW, QX, QY, QZ]
        # scipy expects [QX, QY, QZ, QW]
        rotation_matrix = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
        
        # 3. Create World-to-Camera (W2C) matrix
        # COLMAP provides: X_camera = R @ X_world + t
        w2c_matrix = np.eye(4, dtype=np.float32)
        w2c_matrix[:3, :3] = rotation_matrix
        w2c_matrix[:3, 3] = tvec
        
        print("\n  4x4 World-to-Camera (W2C) Matrix:")
        print(w2c_matrix)
        
        # 4. Convert to Camera-to-World (C2W) matrix
        # C2W = inverse(W2C) = [R^T | -R^T @ t; 0 0 0 | 1]
        R_w2c = w2c_matrix[:3, :3]
        t_w2c = w2c_matrix[:3, 3]
        
        # C2W rotation is transpose of W2C rotation
        R_c2w = R_w2c.T
        
        # C2W translation is -R^T @ t
        t_c2w = -R_c2w @ t_w2c
        
        # Create C2W matrix
        c2w_matrix = np.eye(4, dtype=np.float32)
        c2w_matrix[:3, :3] = R_c2w
        c2w_matrix[:3, 3] = t_c2w
        
        print("\n  4x4 Camera-to-World (C2W) Matrix (for inference.py):")
        print(c2w_matrix)
        
        # 5. Camera center in world coordinates (for verification)
        # Camera center = translation part of C2W matrix
        camera_center_world = c2w_matrix[:3, 3]
        print(f"\n  Camera Center Position (World Coords): {camera_center_world}")
        
        # 6. Verify: camera center should also be -R_w2c^T @ t_w2c
        camera_center_verify = -R_w2c.T @ t_w2c
        print(f"  Camera Center (verified): {camera_center_verify}")
        
        # 7. Store for EXTRINSICS_HARDCODED
        extrinsics_list.append(c2w_matrix)

# 8. Print in format ready for EXTRINSICS_HARDCODED
print("\n" + "="*70)
print("EXTRINSICS_HARDCODED format (copy to inference.py):")
print("="*70)
print("EXTRINSICS_HARDCODED = [")
for i, c2w in enumerate(extrinsics_list):
    print(f"    # View {i}: {image_names_to_print[i] if i < len(image_names_to_print) else ''}")
    print(f"    np.array([")
    for row in c2w:
        print(f"        [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}, {row[3]:.4f}],")
    print(f"    ], dtype=np.float32),")
print("]")
