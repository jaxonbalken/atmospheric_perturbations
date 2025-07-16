# show_metadata.py

import os
import h5py
import cv2
import argparse
from tkinter import Tk, filedialog

def select_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select HDF5 or video file",
        filetypes=[
            ("Supported files", "*.h5 *.hdf5 *.mp4 *.avi"),
            ("HDF5 files", "*.h5 *.hdf5"),
            ("Video files", "*.mp4 *.avi"),
            ("All files", "*.*")
        ]
    )
    return file_path

def show_hdf5_metadata(file_path, export=False):
    output_lines = []
    output_lines.append(f"[HDF5] Metadata for: {file_path}")

    with h5py.File(file_path, 'r') as f:
        if f.attrs:
            output_lines.append("\n[ROOT ATTRIBUTES]")
            for key, val in f.attrs.items():
                output_lines.append(f"{key}: {val}  <-- (Header: '{key}')")

        def recurse(name, obj):
            if isinstance(obj, h5py.Group):
                output_lines.append(f"\n[GROUP] {name}")
                for k, v in obj.attrs.items():
                    output_lines.append(f"  {k}: {v}  <-- (Header: '{k}')")
            elif isinstance(obj, h5py.Dataset):
                output_lines.append(f"\n[DATASET] {name}")
                output_lines.append(f"  Shape: {obj.shape}")
                output_lines.append(f"  Dtype: {obj.dtype}")
                for k, v in obj.attrs.items():
                    output_lines.append(f"  {k}: {v}  <-- (Header: '{k}')")

        f.visititems(recurse)

    for line in output_lines:
        print(line)

    if export:
        out_file = os.path.splitext(file_path)[0] + "_metadata.txt"
        with open(out_file, 'w') as f_out:
            f_out.write("\n".join(output_lines))
        print(f"\n[INFO] Metadata exported to: {out_file}")

def show_video_metadata(file_path, export=False):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("[ERROR] Could not open video file.")
        return

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).strip()

    output_lines = [
        f"[VIDEO] Metadata for: {file_path}",
        f"Frame Width: {width}  <-- (Header: 'CAP_PROP_FRAME_WIDTH')",
        f"Frame Height: {height}  <-- (Header: 'CAP_PROP_FRAME_HEIGHT')",
        f"FPS: {fps}  <-- (Header: 'CAP_PROP_FPS')",
        f"Total Frames: {frame_count}  <-- (Header: 'CAP_PROP_FRAME_COUNT')",
        f"Duration: {duration:.2f} seconds (calculated)",
        f"Codec: {codec}  <-- (Header: 'CAP_PROP_FOURCC')"
    ]

    for line in output_lines:
        print(line)

    if export:
        out_file = os.path.splitext(file_path)[0] + "_video_metadata.txt"
        with open(out_file, 'w') as f_out:
            f_out.write("\n".join(output_lines))
        print(f"\n[INFO] Metadata exported to: {out_file}")

    cap.release()

def main():
    parser = argparse.ArgumentParser(description="Show raw metadata from HDF5 or video file.")
    parser.add_argument("--export", action="store_true", help="Export metadata to a .txt file")
    args = parser.parse_args()

    file_path = select_file()
    if not file_path:
        print("[INFO] No file selected.")
        return

    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext in [".h5", ".hdf5"]:
            show_hdf5_metadata(file_path, export=args.export)
        elif ext in [".mp4", ".avi"]:
            show_video_metadata(file_path, export=args.export)
        else:
            print("[ERROR] Unsupported file type.")
    except Exception as e:
        print(f"[ERROR] Failed to read metadata: {e}")

if __name__ == "__main__":
    main()
