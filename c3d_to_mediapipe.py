import os
import glob
import numpy as np
import c3d
from scipy.interpolate import interp1d
from tqdm import tqdm

def c3d_to_mediapipe_33(c3d_path, out_fps=30.0):
    with open(c3d_path, 'rb') as f:
        reader = c3d.Reader(f)
        labels = [label.strip() for label in reader.point_labels]
        original_fps = reader.header.frame_rate
        
        # Extract points
        frames = []
        for i, points, analog in reader.read_frames():
            # points is of shape (N, 5): x, y, z, residual, camera
            frames.append(points[:, :3])
        
    frames = np.array(frames) # (num_frames, num_markers, 3)
    num_frames, num_markers, _ = frames.shape
    
    # Mapping to get 33 markers, handle missing
    def get_marker(name):
        return frames[:, labels.index(name), :] if name in labels else np.zeros((num_frames, 3))
    
    # Extract needed markers
    LFHD = get_marker('LFHD')
    RFHD = get_marker('RFHD')
    LBHD = get_marker('LBHD')
    RBHD = get_marker('RBHD')
    
    LAC = get_marker('LAC')
    RAC = get_marker('RAC')
    
    L_HLE = get_marker('L_HLE')
    R_HLE = get_marker('R_HLE')
    
    L_RSP = get_marker('L_RSP')
    L_USP = get_marker('L_USP')
    R_RSP = get_marker('R_RSP')
    R_USP = get_marker('R_USP')
    
    L_HM1 = get_marker('L_HM1')
    R_HM1 = get_marker('R_HM1')
    
    L_IAS = get_marker('L_IAS')
    R_IAS = get_marker('R_IAS')
    
    L_FLE = get_marker('L_FLE')
    R_FLE = get_marker('R_FLE')
    
    L_FAL = get_marker('L_FAL')
    R_FAL = get_marker('R_FAL')
    
    L_FCC = get_marker('L_FCC')
    R_FCC = get_marker('R_FCC')
    
    L_FM2 = get_marker('L_FM2')
    R_FM2 = get_marker('R_FM2')
    
    head_center = (LFHD + RFHD + LBHD + RBHD) / 4.0
    eye_l = (LFHD + head_center) / 2.0
    eye_r = (RFHD + head_center) / 2.0
    ear_l = LBHD
    ear_r = RBHD
    mouth_l = (LFHD + 2 * head_center) / 3.0
    mouth_r = (RFHD + 2 * head_center) / 3.0

    mp_pose = np.zeros((num_frames, 33, 3))
    
    # 0 Nose
    mp_pose[:, 0] = (LFHD + RFHD) / 2.0
    # 1-3 Left eye related
    mp_pose[:, 1:4] = eye_l[:, np.newaxis, :]
    # 4-6 Right eye related
    mp_pose[:, 4:7] = eye_r[:, np.newaxis, :]
    # 7-8 Ears
    mp_pose[:, 7] = ear_l
    mp_pose[:, 8] = ear_r
    # 9-10 Mouth
    mp_pose[:, 9] = mouth_l
    mp_pose[:, 10] = mouth_r
    
    # 11-12 Shoulders
    mp_pose[:, 11] = LAC
    mp_pose[:, 12] = RAC
    
    # 13-14 Elbows
    mp_pose[:, 13] = L_HLE
    mp_pose[:, 14] = R_HLE
    
    # 15-16 Wrists
    mp_pose[:, 15] = (L_RSP + L_USP) / 2.0
    mp_pose[:, 16] = (R_RSP + R_USP) / 2.0
    
    # 17-22 Fingers/Hands
    mp_pose[:, 17] = L_HM1
    mp_pose[:, 18] = R_HM1
    mp_pose[:, 19] = L_HM1
    mp_pose[:, 20] = R_HM1
    mp_pose[:, 21] = L_HM1
    mp_pose[:, 22] = R_HM1
    
    # 23-24 Hips
    mp_pose[:, 23] = L_IAS
    mp_pose[:, 24] = R_IAS
    
    # 25-26 Knees
    mp_pose[:, 25] = L_FLE
    mp_pose[:, 26] = R_FLE
    
    # 27-28 Ankles
    mp_pose[:, 27] = L_FAL
    mp_pose[:, 28] = R_FAL
    
    # 29-30 Heels
    mp_pose[:, 29] = L_FCC
    mp_pose[:, 30] = R_FCC
    
    # 31-32 Foot index
    mp_pose[:, 31] = L_FM2
    mp_pose[:, 32] = R_FM2
    
    # Resample to out_fps
    duration = num_frames / original_fps
    num_out_frames = int(duration * out_fps)
    
    t_original = np.linspace(0, duration, num_frames)
    t_new = np.linspace(0, duration, num_out_frames)
    
    interpolator = interp1d(t_original, mp_pose, axis=0, kind='linear', fill_value='extrapolate')
    mp_pose_resampled = interpolator(t_new)
    
    return mp_pose_resampled

def process_all_c3d(c3d_folder, out_folder, target_fps=30):
    os.makedirs(out_folder, exist_ok=True)
    c3d_files = glob.glob(os.path.join(c3d_folder, "*.c3d"))
    
    print(f"Found {len(c3d_files)} C3D files. Processing to {target_fps} fps MediaPipe format...")
    for f in tqdm(c3d_files):
        try:
            mp_data = c3d_to_mediapipe_33(f, out_fps=target_fps)
            basename = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(out_folder, f"{basename}.npy")
            np.save(out_path, mp_data)
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == '__main__':
    c3d_dir = "./Segmented_C3D"
    out_dir = "./C3D_PROCESSED"
    process_all_c3d(c3d_dir, out_dir, target_fps=30)
