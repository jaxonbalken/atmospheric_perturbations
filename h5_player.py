import h5py
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

def normalize_frame(frame):
    """Normalize frame to uint8 for display."""
    if frame.dtype != np.uint8:
        fmin, fmax = frame.min(), frame.max()
        if fmax > fmin:
            frame = ((frame - fmin) / (fmax - fmin) * 255).astype(np.uint8)
        else:
            frame = np.zeros_like(frame, dtype=np.uint8)
    return frame

def main():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select HDF5 File for Playback",
        filetypes=[("HDF5 files", "*.h5 *.hdf5")]
    )
    if not file_path or not os.path.exists(file_path):
        print("[INFO] No file selected or file not found.")
        return

    print(f"[INFO] Opening file: {file_path}")
    with h5py.File(file_path, 'r') as f:
        if 'images' not in f:
            print("[ERROR] 'images' dataset not found in HDF5 file.")
            return
        images = f['images'][:]
        print(f"[INFO] Loaded 'images' dataset: {images.shape}")

        has_images2 = 'images2' in f
        images2 = None
        if has_images2:
            images2 = f['images2'][:]
            print(f"[INFO] Loaded 'images2' dataset: {images2.shape}")
            if images2.shape[0] != images.shape[0]:
                print("[WARNING] 'images2' has different frame count than 'images'. Truncating to minimum length.")
                min_len = min(images.shape[0], images2.shape[0])
                images = images[:min_len]
                images2 = images2[:min_len]

        fps = f.attrs.get('actual_fps', 20.0)
        print(f"[INFO] Using playback FPS: {fps}")

    delay_ms = int(1000 / fps)

    cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
    if has_images2:
        cv2.namedWindow("Images2", cv2.WINDOW_NORMAL)

    for i in range(images.shape[0]):
        frame1 = normalize_frame(images[i])
        disp1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR) if len(frame1.shape) == 2 else frame1
        cv2.imshow("Images", disp1)

        if has_images2:
            frame2 = normalize_frame(images2[i])
            disp2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR) if len(frame2.shape) == 2 else frame2
            cv2.imshow("Images2", disp2)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == 27:  # ESC key
            print("[INFO] Playback interrupted by user.")
            break

    cv2.destroyAllWindows()
    print("[INFO] Playback finished.")

if __name__ == "__main__":
    main()
