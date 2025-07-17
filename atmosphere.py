import os
import cv2
import h5py
import csv
import time
import numpy as np
import argparse
import traceback
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

def select_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a video or HDF5 file",
        filetypes=[
            ("Supported files", "*.h5 *.hdf5 *.mp4 *.avi"),
            ("HDF5 files", "*.h5 *.hdf5"),
            ("Video files", "*.mp4 *.avi"),
            ("All files", "*.*")
        ]
    )
    return file_path

def load_frames(file_path):
    frames = []
    capture_fps = 120.0  # Default fallback

    if file_path.lower().endswith(('.h5', '.hdf5')):
        with h5py.File(file_path, 'r') as f:
            if 'frames' not in f:
                raise ValueError("No 'frames' dataset found in HDF5 file.")
            frames = f['frames'][:]
            capture_fps = float(f.attrs.get('actual_fps', 120))
    elif file_path.lower().endswith(('.mp4', '.avi')):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError("Failed to open video file.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()
    else:
        raise ValueError("Unsupported file format.")

    return frames, capture_fps

def compute_centroids_and_motion(frames):
    centroids = []
    movements = []
    dx_list, dy_list = [], []
    last_centroid = None

    for frame in frames:
        ret, thresh = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        M = cv2.moments(thresh)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroid = (cX, cY)
        else:
            centroid = (None, None)

        centroids.append(centroid)

        if centroid != (None, None) and last_centroid:
            dx = centroid[0] - last_centroid[0]
            dy = centroid[1] - last_centroid[1]
            dist = np.sqrt(dx**2 + dy**2)
        else:
            dx = dy = dist = 0

        dx_list.append(dx)
        dy_list.append(dy)
        movements.append(dist)

        if centroid != (None, None):
            last_centroid = centroid

    return centroids, dx_list, dy_list, movements

def analyze_centroid_statistics(dx_list, dy_list, movements, capture_fps):
    dx_array = np.array(dx_list)
    dy_array = np.array(dy_list)

    rms_dx = np.sqrt(np.mean(dx_array**2))
    rms_dy = np.sqrt(np.mean(dy_array**2))
    rms_total = np.sqrt(rms_dx**2 + rms_dy**2)

    std_dx = np.std(dx_array)
    std_dy = np.std(dy_array)
    std_total = np.std(movements)

    print(f"[STATS] RMS Displacement X: {rms_dx:.3f} px")
    print(f"[STATS] RMS Displacement Y: {rms_dy:.3f} px")
    print(f"[STATS] RMS Total Displacement: {rms_total:.3f} px")
    print(f"[STATS] Std Dev X: {std_dx:.3f} px")
    print(f"[STATS] Std Dev Y: {std_dy:.3f} px")
    print(f"[STATS] Std Dev Total Motion: {std_total:.3f} px")

    return rms_dx, rms_dy, rms_total, std_dx, std_dy, std_total

def save_to_csv(output_path, centroids, dx, dy, movements, capture_fps):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Centroid_X', 'Centroid_Y', 'dX', 'dY', 'Movement_Pixels', 'FPS'])
        for i, ((x, y), dx_val, dy_val, move) in enumerate(zip(centroids, dx, dy, movements)):
            writer.writerow([i, x if x is not None else 'None', y if y is not None else 'None',
                             dx_val, dy_val, move, capture_fps])
    print(f"[INFO] CSV saved to: {output_path}")

def plot_motion(centroids):
    coords = [(x, y) for x, y in centroids if x is not None]
    if not coords:
        print("[WARN] No valid centroid data to plot.")
        return
    xs, ys = zip(*coords)
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, '-o', markersize=2, linewidth=1)
    plt.title("Centroid Motion (Atmospheric Jitter)")
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels)")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def show_preview(frames, centroids, wait_ms=30):
    print("[INFO] Showing centroid overlay preview...")
    for i, frame in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if centroids[i] != (None, None):
            cv2.circle(frame_bgr, centroids[i], 4, (0, 0, 255), -1)
        cv2.putText(frame_bgr, f"Frame {i+1}/{len(frames)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow("Centroid Preview", frame_bgr)

        key = cv2.waitKey(wait_ms)
        if key == 27:  # ESC
            print("[INFO] Preview interrupted by user.")
            break
        elif key == ord(' '):  # Pause on space
            print("[INFO] Paused. Press space again to continue.")
            while cv2.waitKey(0) != ord(' '):
                pass

    cv2.destroyAllWindows()
    print("[INFO] Preview complete.")

def analyze_file(file_path, preview=False, plot=False):
    try:
        frames, capture_fps = load_frames(file_path)
        print(f"[INFO] Loaded {len(frames)} frames from {file_path}")

        start_time = time.time()
        centroids, dx, dy, movements = compute_centroids_and_motion(frames)
        total_time = time.time() - start_time

        valid_centroids = sum(1 for c in centroids if c != (None, None))
        avg_movement = np.mean([m for m in movements if m > 0])
        max_movement = np.max([m for m in movements if m > 0])

        print("\n========= SUMMARY =========")
        print(f"Total Frames: {len(frames)}")
        print(f"Valid Centroids: {valid_centroids}")
        print(f"Average Movement: {avg_movement:.2f} px")
        print(f"Max Movement: {max_movement:.2f} px")
        print(f"Total Processing Time: {total_time:.2f} s")
        print(f"Avg Per Frame: {total_time / len(frames):.4f} s")
        print("===========================\n")

        analyze_centroid_statistics(dx, dy, movements, capture_fps)

        if plot:
            plot_motion(centroids)

        save_prompt = input("Save data to CSV? (y/n): ").strip().lower()
        if save_prompt == 'y':
            csv_path = os.path.splitext(file_path)[0] + '_centroids.csv'
            save_to_csv(csv_path, centroids, dx, dy, movements, capture_fps)

        # Show preview automatically or prompt user
        if preview:
            show_preview(frames, centroids)
        else:
            preview_prompt = input("Show preview now? (y/n): ").strip().lower()
            if preview_prompt == 'y':
                show_preview(frames, centroids)

    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Analyze atmospheric perturbations from Polaris video.")
    parser.add_argument("--preview", action="store_true", help="Show video preview with centroid overlay")
    parser.add_argument("--plot", action="store_true", help="Show centroid motion plot")
    args = parser.parse_args()

    file_path = select_file()
    if not file_path:
        print("[INFO] No file selected. Exiting.")
        return

    analyze_file(file_path, preview=args.preview, plot=args.plot)

if __name__ == "__main__":
    main()
