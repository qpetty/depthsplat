import numpy as np
# Assuming read_write_model.py is in the same directory
#from read_write_model import read_cameras_binary, read_images_binary, read_points3d_binary
from read_write_model import read_cameras_binary, read_images_binary, qvec2rotmat

# Define the path to your sparse model directory
model_path = "/Users/quinton/Desktop/colmap_output/sparse/0/"
#image_names_to_print = ["frame_0002.png", "frame_0035.png", "frame_0070.png", "frame_0105.png", "frame_0140.png"]
image_names_to_print = ["frame_0002.png", "frame_0070.png", "frame_0140.png"]

# Read the binary files
cameras = read_cameras_binary(model_path + "cameras.bin")
images = read_images_binary(model_path + "images.bin")
#points3D = read_points3d_binary(model_path + "points3D.bin")

# Store intrinsics for INTRINSICS_HARDCODED output
intrinsics_fx = None
intrinsics_fy = None
intrinsics_cx = None
intrinsics_cy = None
camera_used = None

# First, try to get camera from the images we're processing
# Find which camera_id is used by the images we're processing
camera_ids_used = set()
for img_id, img_data in images.items():
    if img_data.name in image_names_to_print:
        camera_ids_used.add(img_data.camera_id)

# Try camera_id_to_print first, then any camera used by our images, then any camera
camera_id_to_print = 1  # Default camera ID to try first
camera_id_to_try = camera_id_to_print

if camera_id_to_print not in cameras and len(camera_ids_used) > 0:
    camera_id_to_try = list(camera_ids_used)[0]
    print(f"Camera ID {camera_id_to_print} not found. Trying camera ID {camera_id_to_try} (used by images).")

if camera_id_to_try not in cameras and len(cameras) > 0:
    camera_id_to_try = list(cameras.keys())[0]
    print(f"Camera ID not found. Trying first available camera ID {camera_id_to_try}.")

# Print all available cameras for diagnostics
print("\n" + "="*70)
print("Available Cameras in Reconstruction:")
print("="*70)
for cam_id, cam in cameras.items():
    print(f"  Camera ID {cam_id}: {cam.model}, {cam.width}x{cam.height}, {len(cam.params)} params")

# Extract intrinsics from the camera
if camera_id_to_try in cameras:
    camera = cameras[camera_id_to_try]
    camera_used = camera
    
    print(f"\n--- Using Camera ID {camera.id} Intrinsics ---")
    print(f"  Model:  {camera.model}")
    print(f"  Width:  {camera.width} pixels")
    print(f"  Height: {camera.height} pixels")
    print(f"  Params: {camera.params}") # This is a numpy array
    print(f"  Num params: {len(camera.params)}")

    # Extract fx, fy, cx, cy based on camera model
    # COLMAP camera model parameter orders:
    params = camera.params
    
    if camera.model == 'SIMPLE_PINHOLE':
        # f, cx, cy (3 params)
        if len(params) >= 3:
            f, cx, cy = params[0], params[1], params[2]
            intrinsics_fx = f
            intrinsics_fy = f
            intrinsics_cx = cx
            intrinsics_cy = cy
            print(f"\nDetailed Parameters (SIMPLE_PINHOLE model):")
            print(f"  Focal Length (f): {f:.4f} (fx = fy = {f:.4f})")
            print(f"  Principal Point X (cx): {cx:.4f}")
            print(f"  Principal Point Y (cy): {cy:.4f}")
    
    elif camera.model == 'PINHOLE':
        # fx, fy, cx, cy (4 params)
        if len(params) >= 4:
            intrinsics_fx, intrinsics_fy, intrinsics_cx, intrinsics_cy = params[0], params[1], params[2], params[3]
            print(f"\nDetailed Parameters (PINHOLE model):")
            print(f"  Focal Length X (fx): {intrinsics_fx:.4f}")
            print(f"  Focal Length Y (fy): {intrinsics_fy:.4f}")
            print(f"  Principal Point X (cx): {intrinsics_cx:.4f}")
            print(f"  Principal Point Y (cy): {intrinsics_cy:.4f}")
    
    elif camera.model == 'SIMPLE_RADIAL':
        # f, cx, cy, k (4 params) - ignore k (distortion)
        if len(params) >= 3:
            f, cx, cy = params[0], params[1], params[2]
            intrinsics_fx = f
            intrinsics_fy = f
            intrinsics_cx = cx
            intrinsics_cy = cy
            print(f"\nDetailed Parameters (SIMPLE_RADIAL model):")
            print(f"  Focal Length (f): {f:.4f} (fx = fy = {f:.4f})")
            print(f"  Principal Point X (cx): {cx:.4f}")
            print(f"  Principal Point Y (cy): {cy:.4f}")
            print(f"  Distortion k: {params[3]:.4f} (ignored for pinhole intrinsics)")
    
    elif camera.model == 'RADIAL':
        # f, cx, cy, k1, k2 (5 params) - ignore k1, k2 (distortion)
        if len(params) >= 3:
            f, cx, cy = params[0], params[1], params[2]
            intrinsics_fx = f
            intrinsics_fy = f
            intrinsics_cx = cx
            intrinsics_cy = cy
            print(f"\nDetailed Parameters (RADIAL model):")
            print(f"  Focal Length (f): {f:.4f} (fx = fy = {f:.4f})")
            print(f"  Principal Point X (cx): {cx:.4f}")
            print(f"  Principal Point Y (cy): {cy:.4f}")
            print(f"  Distortion k1: {params[3]:.4f}, k2: {params[4]:.4f} (ignored for pinhole intrinsics)")
    
    elif camera.model in ['OPENCV', 'OPENCV_FISHEYE']:
        # fx, fy, cx, cy, ... (distortion params) - ignore distortion
        if len(params) >= 4:
            intrinsics_fx, intrinsics_fy, intrinsics_cx, intrinsics_cy = params[0], params[1], params[2], params[3]
            print(f"\nDetailed Parameters ({camera.model} model):")
            print(f"  Focal Length X (fx): {intrinsics_fx:.4f}")
            print(f"  Focal Length Y (fy): {intrinsics_fy:.4f}")
            print(f"  Principal Point X (cx): {intrinsics_cx:.4f}")
            print(f"  Principal Point Y (cy): {intrinsics_cy:.4f}")
            print(f"  Distortion params: {params[4:]} (ignored for pinhole intrinsics)")
    
    elif camera.model == 'FULL_OPENCV':
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 (12 params)
        if len(params) >= 4:
            intrinsics_fx, intrinsics_fy, intrinsics_cx, intrinsics_cy = params[0], params[1], params[2], params[3]
            print(f"\nDetailed Parameters (FULL_OPENCV model):")
            print(f"  Focal Length X (fx): {intrinsics_fx:.4f}")
            print(f"  Focal Length Y (fy): {intrinsics_fy:.4f}")
            print(f"  Principal Point X (cx): {intrinsics_cx:.4f}")
            print(f"  Principal Point Y (cy): {intrinsics_cy:.4f}")
            print(f"  Distortion params: {params[4:]} (ignored for pinhole intrinsics)")
    
    else:
        print(f"\nWARNING: Camera model '{camera.model}' is not directly supported.")
        print(f"  Params: {params}")
        print(f"  Attempting to extract fx, fy, cx, cy from first 4 params...")
        if len(params) >= 4:
            intrinsics_fx, intrinsics_fy, intrinsics_cx, intrinsics_cy = params[0], params[1], params[2], params[3]
            print(f"  Extracted: fx={intrinsics_fx:.4f}, fy={intrinsics_fy:.4f}, cx={intrinsics_cx:.4f}, cy={intrinsics_cy:.4f}")
        elif len(params) >= 3:
            # Assume SIMPLE_PINHOLE format: f, cx, cy
            f, cx, cy = params[0], params[1], params[2]
            intrinsics_fx = f
            intrinsics_fy = f
            intrinsics_cx = cx
            intrinsics_cy = cy
            print(f"  Extracted (assuming f, cx, cy): fx=fy={f:.4f}, cx={cx:.4f}, cy={cy:.4f}")

    # Print intrinsic matrix if we successfully extracted values
    if intrinsics_fx is not None:
        K = np.array([
            [intrinsics_fx, 0, intrinsics_cx],
            [0, intrinsics_fy, intrinsics_cy],
            [0, 0, 1]
        ])
        print("\nIntrinsic Matrix (K):")
        np.set_printoptions(precision=4, suppress=True)
        print(K)
    else:
        print("\nERROR: Could not extract intrinsics from camera parameters.")

else:
    print(f"\nERROR: No cameras found in the reconstruction.")
    print(f"  Tried camera IDs: {camera_id_to_print}, {list(camera_ids_used) if camera_ids_used else 'none'}")
    print(f"  Available camera IDs: {list(cameras.keys()) if cameras else 'none'}")

print(f"\n--- Extrinsics for Specified Images ---")
np.set_printoptions(precision=4, suppress=True) # For legible numpy output

# Iterate through all registered images to find the specified ones
extrinsics_list = []  # Store C2W matrices for EXTRINSICS_HARDCODED
extrinsics_image_names = []  # Store corresponding image names

for img_id, img_data in images.items():
    if img_data.name in image_names_to_print:
        print(f"\nImage Name: {img_data.name} (Image ID: {img_id})")
        
        # 1. Access the quaternion and translation vector
        qvec = img_data.qvec  # COLMAP format: [QW, QX, QY, QZ]
        tvec = img_data.tvec  # Translation vector [TX, TY, TZ]
        
        print(f"  Quaternion (QW, QX, QY, QZ): {qvec}")
        print(f"  Translation Vector (TX, TY, TZ): {tvec}")
        
        # 2. Convert quaternion to rotation matrix using COLMAP's official function
        # CRITICAL: Use COLMAP's qvec2rotmat instead of scipy to ensure correct conversion
        # This ensures we use the exact same conversion as COLMAP's internal implementation
        rotation_matrix = qvec2rotmat(qvec)
        
        # 3. Create World-to-Camera (W2C) matrix
        # COLMAP provides: X_camera = R @ X_world + t
        w2c_matrix = np.eye(4, dtype=np.float32)
        w2c_matrix[:3, :3] = rotation_matrix
        w2c_matrix[:3, 3] = tvec
        
        print("\n  4x4 World-to-Camera (W2C) Matrix:")
        print(w2c_matrix)
        
        # 4. Convert to Camera-to-World (C2W) matrix
        # Use numpy's inverse for numerical stability and correctness
        c2w_matrix = np.linalg.inv(w2c_matrix).astype(np.float32)
        
        print("\n  4x4 Camera-to-World (C2W) Matrix (for inference.py):")
        print(c2w_matrix)
        
        # 5. Camera center in world coordinates (for verification)
        # Camera center = translation part of C2W matrix
        camera_center_world = c2w_matrix[:3, 3]
        print(f"\n  Camera Center Position (World Coords): {camera_center_world}")
        
        # 6. Verify camera center computation
        R_w2c = w2c_matrix[:3, :3]
        t_w2c = w2c_matrix[:3, 3]
        camera_center_verify = -R_w2c.T @ t_w2c
        print(f"  Camera Center (verified): {camera_center_verify}")
        
        # Verify they match (should be very close)
        if np.allclose(camera_center_world, camera_center_verify, atol=1e-5):
            print(f"  ✓ Camera center verification passed!")
        else:
            diff = np.abs(camera_center_world - camera_center_verify).max()
            print(f"  ⚠️  WARNING: Camera center mismatch! Max difference: {diff:.6f}")
        
        # 7. Store for EXTRINSICS_HARDCODED (maintain order from image_names_to_print)
        extrinsics_list.append(c2w_matrix)
        extrinsics_image_names.append(img_data.name)

# Ensure extrinsics are in the same order as image_names_to_print
ordered_extrinsics = []
ordered_names = []
for img_name in image_names_to_print:
    if img_name in extrinsics_image_names:
        idx = extrinsics_image_names.index(img_name)
        ordered_extrinsics.append(extrinsics_list[idx])
        ordered_names.append(img_name)

# Compute coordinate system scale diagnostics
if len(ordered_extrinsics) >= 2:
    # Compute camera baselines
    camera_centers_for_scale = [c2w[:3, 3] for c2w in ordered_extrinsics]
    baselines = []
    for i in range(len(camera_centers_for_scale)):
        for j in range(i + 1, len(camera_centers_for_scale)):
            baseline = np.linalg.norm(camera_centers_for_scale[i] - camera_centers_for_scale[j])
            baselines.append(baseline)
    
    if baselines:
        avg_baseline = np.mean(baselines)
        min_baseline = np.min(baselines)
        max_baseline = np.max(baselines)
        
        print("\n" + "="*70)
        print("COORDINATE SYSTEM SCALE ANALYSIS")
        print("="*70)
        print(f"Camera baselines (distances between cameras):")
        print(f"  Minimum baseline: {min_baseline:.3f} units")
        print(f"  Maximum baseline: {max_baseline:.3f} units")
        print(f"  Average baseline: {avg_baseline:.3f} units")
        print(f"\nNOTE: COLMAP uses arbitrary scale units.")
        print(f"  - The scale depends on your reconstruction")
        print(f"  - These values are in COLMAP's coordinate system units")
        print(f"  - Near/far planes should be computed dynamically based on baselines")
        print(f"  - inference.py will automatically compute near/far from baselines")
        print(f"\n  Typical baseline scales for reference:")
        print(f"    * Small scene (indoors, close-up): 0.1-2.0 units")
        print(f"    * Medium scene (room, person): 1.0-10.0 units")
        print(f"    * Large scene (outdoor, building): 5.0-100.0 units")
        print(f"\n  Your scene appears to be: ", end="")
        if avg_baseline < 1.0:
            print("SMALL scale (indoor/close-up)")
        elif avg_baseline < 5.0:
            print("MEDIUM scale (room/person)")
        else:
            print("LARGE scale (outdoor/building)")
        print("="*70)

# Print in format ready for inference.py
print("\n" + "="*70)
print("COPY THE FOLLOWING TO inference.py:")
print("="*70)

# Print IMAGE DIMENSIONS - CRITICAL!
if camera_used is not None:
    print("\n" + "="*70)
    print("IMPORTANT: IMAGE DIMENSIONS")
    print("="*70)
    print(f"COLMAP reconstruction used: width={camera_used.width}, height={camera_used.height}")
    print(f"\nIn inference.py, set:")
    print(f"  height, width = {camera_used.height}, {camera_used.width}  # Match COLMAP dimensions")
    print("="*70)

# Print INTRINSICS_HARDCODED
if intrinsics_fx is not None and intrinsics_fy is not None and intrinsics_cx is not None and intrinsics_cy is not None:
    print("\n# Camera intrinsics (before normalization)")
    print("# Set to None to use computed values, or provide [fx, fy, cx, cy] in pixels")
    print("# If provided, will be normalized by image dimensions automatically")
    print("# IMPORTANT: Make sure image dimensions in inference.py match COLMAP dimensions above!")
    print("INTRINSICS_HARDCODED = None")
    print("# Example (uncomment to use):")
    print(f"INTRINSICS_HARDCODED = [{intrinsics_fx:.5f}, {intrinsics_fy:.5f}, {intrinsics_cx:.1f}, {intrinsics_cy:.1f}]  # [fx, fy, cx, cy] in pixels")
else:
    print("\n# WARNING: Could not extract intrinsics.")
    if camera_used is not None:
        print(f"#   Camera model found: {camera_used.model}")
        print(f"#   Camera params: {camera_used.params}")
        print(f"#   Number of params: {len(camera_used.params)}")
        print("#   Please check the diagnostic output above for details.")
    else:
        print("#   No camera was found. Check that:")
        print("#   1. The model_path points to the correct COLMAP sparse reconstruction")
        print("#   2. The cameras.bin file exists and is readable")
        print("#   3. The reconstruction contains at least one camera")

# Print EXTRINSICS_HARDCODED
print("\n# Camera extrinsics (4x4 camera-to-world matrices)")
print("# Set to None to use computed poses, or provide list of 3 numpy arrays or torch tensors")
print("# Each matrix should be 4x4 in shape")
print("# If provided, must have exactly 3 matrices (one per view)")
print("EXTRINSICS_HARDCODED = None")
print("# Example (uncomment to use - note: numpy is already imported as np):")
print("EXTRINSICS_HARDCODED = [")
for i, c2w in enumerate(ordered_extrinsics):
    img_name = ordered_names[i] if i < len(ordered_names) else f"View {i}"
    print(f"     # View {i}: {img_name}")
    print(f"     np.array([")
    for row in c2w:
        print(f"        [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}, {row[3]:.4f}],")
    print(f"     ], dtype=np.float32),")
print("]")

print("\n" + "="*70)
