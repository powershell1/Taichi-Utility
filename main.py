import numpy as np
import json

# 1. Load the numpy array
# Assuming shape is (N_frames, 33_landmarks, 3_coordinates)
data = np.load('/Users/mac/Desktop/Taichi-C3D-Transform/C3D_PROCESSED/P01T01C01.npy') 

# 2. Reformat data for Unity
frames_list = []
for frame in data:
    frame_data = []
    for landmark in frame:
        # Note: We extract X, Y, Z. 
        # MediaPipe's Y axis is typically inverted compared to Unity.
        # We will handle the exact coordinate flip in Unity C#.
        frame_data.append({
            "x": float(landmark[0]),
            "y": float(landmark[1]),
            "z": float(landmark[2])
        })
    frames_list.append({"landmarks": frame_data})

# 3. Save to JSON
output = {"fps": 30, "frames": frames_list}
with open('mediapipe_animation.json', 'w') as f:
    json.dump(output, f, indent=4)
    
print("Successfully exported to mediapipe_animation.json")