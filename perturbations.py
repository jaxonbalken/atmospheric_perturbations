import h5py
import cv2
import numpy as np
import os
import csv
import json
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

def normalize_frame(frame):
    if frame.dtype != np.uint8:
        fmin, fmax = frame.min(), frame.max()
        if fmax > fmin:
            frame = ((frame - fmin) / (fmax - fmin) * 255).astype(np.uint8)
        else:
            frame = np.zeros_like(frame, dtype=np.uint8)
    return frame

def find_centroid(frame, threshold=50):
    ret, thresh = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    else:
        return None, None

def extract_metadata(h5file):
    metadata = {}
    # File-level attributes
    for key, value in h5file.attrs.items():
        metadata[f"attr_{key}"] = value.tolist() if hasattr(value, 'tolist') else value

    # Dataset info
    datasets = {}
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            datasets[name] = {
                'shape': node.shape,
                'dtype': str(node.dtype),
                'attrs': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in node.attrs.items()}
            }
    h5file.visititems(visitor_func)
    metadata['datasets'] = datasets
    return metadata

def main():
    root = tk.Tk()
    root.withdraw()

    # Select HDF5 file
    file_path = filedialog.askopenfilename(
        title="Select HDF5 File for Playback and Centroid Detection",
        filetypes=[("HDF5 files", "*.h5 *.hdf5")]
    )
    if not file_path or not os.path.exists(file_path):
        print("[INFO] No file selected or file not found.")
        return

    print(f"[INFO] Opening file: {file_path}")
    with h5py.File(file_path, 'r') as f:
        metadata = extract_metadata(f)
        print("[INFO] Extracted file metadata and dataset info.")

        has_images = 'images' in f
        has_images2 = 'images2' in f

        if not has_images and not has_images2:
            messagebox.showerror("Error", "No 'images' or 'images2' datasets found in the file.")
            return

        # Ask user which datasets to use if both exist
        datasets_to_use = []
        if has_images and has_images2:
            choice = messagebox.askquestion(
                "Choose Dataset",
                "Both 'images' and 'images2' found.\n\nTrack both?\n\n(Yes = both, No = only 'images')"
            )
            if choice == 'yes':
                datasets_to_use = ['images', 'images2']
            else:
                # Further ask if 'images' or 'images2'
                choice2 = messagebox.askquestion(
                    "Choose Dataset",
                    "Track only 'images'?\n\n(Yes = 'images', No = 'images2')"
                )
                datasets_to_use = ['images'] if choice2 == 'yes' else ['images2']
        elif has_images:
            datasets_to_use = ['images']
        else:
            datasets_to_use = ['images2']

        # Load selected datasets
        images = f['images'][:] if 'images' in datasets_to_use else None
        images2 = f['images2'][:] if 'images2' in datasets_to_use else None

        fps = f.attrs.get('actual_fps', 20.0)
        print(f"[INFO] Using playback FPS: {fps}")

    delay_ms = int(1000 / fps)

    # Setup OpenCV windows
    if images is not None:
        cv2.namedWindow("Images", cv2.WINDOW_NORMAL)
    if images2 is not None:
        cv2.namedWindow("Images2", cv2.WINDOW_NORMAL)

    last_centroid_1 = None
    last_centroid_2 = None
    centroid_data = []

    frame_count = len(images) if images is not None else 0
    frame_count_2 = len(images2) if images2 is not None else 0
    total_frames = max(frame_count, frame_count_2)

    for i in range(total_frames):
        frame1 = images[i] if images is not None and i < frame_count else None
        frame2 = images2[i] if images2 is not None and i < frame_count_2 else None

        cX1 = cY1 = None
        cX2 = cY2 = None
        delta_x1 = delta_y1 = dist1 = None
        delta_x2 = delta_y2 = dist2 = None

        if frame1 is not None:
            norm_frame1 = normalize_frame(frame1)
            cX1, cY1 = find_centroid(norm_frame1)
            if cX1 is not None and last_centroid_1 is not None:
                delta_x1 = cX1 - last_centroid_1[0]
                delta_y1 = cY1 - last_centroid_1[1]
                dist1 = np.sqrt(delta_x1**2 + delta_y1**2)
            last_centroid_1 = (cX1, cY1) if cX1 is not None else last_centroid_1

            disp_frame1 = cv2.cvtColor(norm_frame1, cv2.COLOR_GRAY2BGR)
            if cX1 is not None:
                cv2.circle(disp_frame1, (cX1, cY1), 5, (0, 0, 255), -1)
            cv2.imshow("Images", disp_frame1)

        if frame2 is not None:
            norm_frame2 = normalize_frame(frame2)
            cX2, cY2 = find_centroid(norm_frame2)
            if cX2 is not None and last_centroid_2 is not None:
                delta_x2 = cX2 - last_centroid_2[0]
                delta_y2 = cY2 - last_centroid_2[1]
                dist2 = np.sqrt(delta_x2**2 + delta_y2**2)
            last_centroid_2 = (cX2, cY2) if cX2 is not None else last_centroid_2

            disp_frame2 = cv2.cvtColor(norm_frame2, cv2.COLOR_GRAY2BGR)
            if cX2 is not None:
                cv2.circle(disp_frame2, (cX2, cY2), 5, (0, 255, 0), -1)
            cv2.imshow("Images2", disp_frame2)

        centroid_data.append({
            'frame': i,
            'images_centroid_x': cX1,
            'images_centroid_y': cY1,
            'images_delta_x': delta_x1,
            'images_delta_y': delta_y1,
            'images_distance': dist1,
            'images2_centroid_x': cX2,
            'images2_centroid_y': cY2,
            'images2_delta_x': delta_x2,
            'images2_delta_y': delta_y2,
            'images2_distance': dist2,
            'timestamp_s': i / fps
        })

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == 27:  # ESC to quit
            print("[INFO] Playback interrupted by user.")
            break

    cv2.destroyAllWindows()

    # Ask where to save centroid CSV and metadata JSON
    save_csv_path = filedialog.asksaveasfilename(
        title="Save Centroid Data CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if not save_csv_path:
        print("[INFO] No save location selected. Data will not be saved.")
        return

    # Save centroid data CSV
    with open(save_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'frame',
            'images_centroid_x', 'images_centroid_y', 'images_delta_x', 'images_delta_y', 'images_distance',
            'images2_centroid_x', 'images2_centroid_y', 'images2_delta_x', 'images2_delta_y', 'images2_distance',
            'timestamp_s'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in centroid_data:
            # Drop columns if dataset not tracked
            if images is None:
                for key_drop in ['images_centroid_x', 'images_centroid_y', 'images_delta_x', 'images_delta_y', 'images_distance']:
                    row.pop(key_drop, None)
            if images2 is None:
                for key_drop in ['images2_centroid_x', 'images2_centroid_y', 'images2_delta_x', 'images2_delta_y', 'images2_distance']:
                    row.pop(key_drop, None)
            writer.writerow(row)
    print(f"[INFO] Saved centroid data CSV to: {save_csv_path}")

    # Save metadata JSON next to CSV, same base name
    json_path = os.path.splitext(save_csv_path)[0] + "_metadata.json"
    with open(json_path, 'w') as jf:
        json.dump(metadata, jf, indent=4)
    print(f"[INFO] Saved metadata JSON to: {json_path}")

if __name__ == "__main__":
    main()