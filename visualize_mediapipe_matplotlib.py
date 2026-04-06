import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

# Mediapipe POSE connections (33 landmarks)
MEDIAPIPE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]

def visualize_mediapipe(npy_path, fps=30):
    try:
        # Load data. Expected shape: (frames, 33, 3) or similar where 3 is (x, y, z)
        # If shape is like (3, 33, frames), we'll transpose it
        data = np.load(npy_path)
        
        # Heuristics to get (frames, num_joints, 3)
        if len(data.shape) == 3:
            if data.shape[0] == 3:  # (3, joints, frames) -> (frames, joints, 3)
                data = data.transpose((2, 1, 0))
            elif data.shape[-1] != 3 and data.shape[1] == 3: # (frames, 3, joints) -> (frames, joints, 3)
                data = data.transpose((0, 2, 1))

        frames, num_joints, dims = data.shape
        if num_joints != 33:
            print(f"Warning: Expected 33 joints for Mediapipe, got {num_joints}.")
            # We'll just draw the points if connections don't match

        # Initialize figure and 3D axis
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Calculate limits based on data min/max to keep axis fixed
        x_min, x_max = np.min(data[..., 0]), np.max(data[..., 0])
        y_min, y_max = np.min(data[..., 1]), np.max(data[..., 1])
        z_min, z_max = np.min(data[..., 2]), np.max(data[..., 2])
        
        val_range = max(x_max-x_min, y_max-y_min, z_max-z_min) / 2
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2

        ax.set_xlim(mid_x - val_range, mid_x + val_range)
        ax.set_ylim(mid_y - val_range, mid_y + val_range)
        ax.set_zlim(mid_z - val_range, mid_z + val_range)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Mediapipe Animation ({fps} FPS)")

        # Create line objects for connections
        lines = []
        if num_joints == 33:
            for _ in MEDIAPIPE_CONNECTIONS:
                line, = ax.plot([], [], [], color='blue', linewidth=2)
                lines.append(line)
                
        # Create scatter object for joints
        scat = ax.scatter([], [], [], color='red', s=20)

        def update(frame_idx):
            frame_data = data[frame_idx]
            
            # Update scatter (Requires returning a Path3DCollection, matplotlib handles it but easiest to just clear/redraw or use slightly complex internal apis, actually replacing array in scatter is possible)
            scat._offsets3d = (frame_data[:, 0], frame_data[:, 1], frame_data[:, 2])

            # Update lines
            if num_joints == 33:
                for idx, (i, j) in enumerate(MEDIAPIPE_CONNECTIONS):
                    x_line = [frame_data[i, 0], frame_data[j, 0]]
                    y_line = [frame_data[i, 1], frame_data[j, 1]]
                    z_line = [frame_data[i, 2], frame_data[j, 2]]
                    lines[idx].set_data(x_line, y_line)
                    lines[idx].set_3d_properties(z_line)
                return [scat] + lines
            return [scat]

        interval = 1000 / fps
        anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)

        plt.show()

    except Exception as e:
        print(f"Error initializing visualization: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Mediapipe 30fps NPY file in 3D.")
    parser.add_argument("npy_path", type=str, help="Path to the .npy file containing Mediapipe keypoints.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for playback.")
    args = parser.parse_args()

    visualize_mediapipe(args.npy_path, args.fps)