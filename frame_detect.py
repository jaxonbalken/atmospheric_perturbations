import cv2
import numpy as np
import h5py
import csv
import os
import time
from tkinter import Tk, filedialog

# === Parameters for detection ===
EXPECTED_W, EXPECTED_H = 126, 126
AREA_TOLERANCE = 0.4
AR_TOLERANCE = 0.4

def select_file():
    Tk().withdraw()
    return filedialog.askopenfilename(filetypes=[
        ("MP4 and HDF5 files", "*.mp4 *.h5 *.hdf5"),
        ("All files", "*.*")
    ])

def load_frames(file_path):
    frames = []
    if file_path.lower().endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()
    elif file_path.lower().endswith(('.h5', '.hdf5')):
        with h5py.File(file_path, 'r') as f:
            for key in ['images', 'frames']:
                if key in f:
                    data = f[key][:]
                    for i in range(data.shape[0]):
                        frame = data[i]
                        if frame.ndim == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frames.append(frame)
                    break
    return frames

def detect_rois_from_frame(frame):
    blurred = cv2.GaussianBlur(frame, (5,5), 0)
    inv = cv2.bitwise_not(blurred)
    _, thresh = cv2.threshold(inv, 50, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean)
    rois = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        exp_area = EXPECTED_W * EXPECTED_H
        if abs(area - exp_area) / exp_area <= AREA_TOLERANCE:
            ar = w / h
            if abs(ar - (EXPECTED_W / EXPECTED_H)) / (EXPECTED_W / EXPECTED_H) <= AR_TOLERANCE:
                rois.append((x, y, w, h))
    return sorted(rois, key=lambda b: (b[1], b[0]))

def analyze_roi(roi_frames, roi_index, output_prefix):
    centroids = []
    movements = []
    last_centroid = None

    for i, frame in enumerate(roi_frames):
        _, thresh = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)
        M = cv2.moments(thresh)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
            if last_centroid:
                dx = cX - last_centroid[0]
                dy = cY - last_centroid[1]
                dist = np.sqrt(dx**2 + dy**2)
                movements.append(dist)
            else:
                movements.append(0)
            last_centroid = (cX, cY)
        else:
            centroids.append((None, None))
            movements.append(0)

    avg_movement = np.mean(movements)
    max_movement = np.max(movements)
    print(f"\n[ROI {roi_index}] Avg movement: {avg_movement:.2f}px, Max: {max_movement:.2f}px")

    # Save CSV
    csv_path = f"{output_prefix}_ROI{roi_index}_centroids.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Centroid_X", "Centroid_Y", "Movement"])
        for i, ((x, y), move) in enumerate(zip(centroids, movements)):
            writer.writerow([i, x, y, move])
    print(f"[INFO] ROI {roi_index} data saved to {csv_path}")

def main():
    file_path = select_file()
    if not file_path:
        print("No file selected.")
        return

    frames = load_frames(file_path)
    if not frames:
        print("No frames loaded.")
        return

    first_frame = frames[0]
    if len(first_frame.shape) > 2:
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    print("[INFO] Detecting embedded video ROIs...")
    rois = detect_rois_from_frame(first_frame)

    if not rois:
        print("[ERROR] No ROIs detected.")
        return

    print(f"[INFO] Detected {len(rois)} ROI(s).")

    for idx, (x, y, w, h) in enumerate(rois):
        roi_frames = [frame[y:y+h, x:x+w] for frame in frames]
        analyze_roi(roi_frames, idx+1, os.path.splitext(file_path)[0])

if __name__ == "__main__":
    main()
