import h5py
import cv2
import numpy as np
from tkinter import filedialog, Tk

def select_h5_file():
    Tk().withdraw()
    return filedialog.askopenfilename(filetypes=[("HDF5 Files", "*.h5 *.hdf5")])

def normalize_centroid(centroid_arcsec, frame_size_arcsec=10, image_dim=128):
    """Convert arcseconds to pixels, assuming centered field of view."""
    scale = image_dim / frame_size_arcsec  # pixels per arcsec
    return (centroid_arcsec * scale) + image_dim / 2

def load_h5_frames_and_centroids(filepath):
    with h5py.File(filepath, 'r') as f:
        # Video frames
        images = f['images'][:]
        images2 = f['images2'][:] if 'images2' in f else None

        # Centroids
        xcen = f.get('xcen_arcsec', None)
        ycen = f.get('ycen_arcsec', None)
        xcen2 = f.get('xcen2_arcsec', None)
        ycen2 = f.get('ycen2_arcsec', None)

        centroids = {
            "xcen": xcen[:] if xcen else None,
            "ycen": ycen[:] if ycen else None,
            "xcen2": xcen2[:] if xcen2 else None,
            "ycen2": ycen2[:] if ycen2 else None,
        }

    return images, images2, centroids

def draw_centroid(frame, x_arcsec, y_arcsec, color=(0, 255, 0), label=None):
    if x_arcsec is None or y_arcsec is None:
        return frame
    x_px = int(normalize_centroid(x_arcsec))
    y_px = int(normalize_centroid(y_arcsec))
    frame = cv2.circle(frame.copy(), (x_px, y_px), 4, color, -1)
    if label:
        cv2.putText(frame, label, (x_px + 5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame

def play_h5_video(images1, images2=None, centroids=None, fps=20):
    index = 0
    total_frames = images1.shape[0]
    paused = False

    while True:
        frame1 = images1[index]
        frame2 = images2[index] if images2 is not None else None

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR) if frame2 is not None else None

        # Draw centroids
        if centroids:
            frame1 = draw_centroid(frame1,
                                   centroids['xcen'][index] if centroids['xcen'] is not None else None,
                                   centroids['ycen'][index] if centroids['ycen'] is not None else None,
                                   color=(0, 255, 0), label="1")
            if frame2 is not None:
                frame2 = draw_centroid(frame2,
                                       centroids['xcen2'][index] if centroids['xcen2'] is not None else None,
                                       centroids['ycen2'][index] if centroids['ycen2'] is not None else None,
                                       color=(255, 0, 0), label="2")

        if frame2 is not None:
            combined = np.hstack((frame1, frame2))
        else:
            combined = frame1

        cv2.putText(combined, f"Frame: {index+1}/{total_frames}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("HDF5 Video Player", combined)
        key = cv2.waitKey(0 if paused else int(1000 / fps))

        if key == 27:  # ESC
            break
        elif key == ord(' '):  # Toggle pause
            paused = not paused
        elif key in [ord('d'), 83]:  # Next frame
            index = min(index + 1, total_frames - 1)
        elif key in [ord('a'), 81]:  # Previous frame
            index = max(index - 1, 0)

    cv2.destroyAllWindows()

def main():
    filepath = select_h5_file()
    if not filepath:
        print("[INFO] No file selected.")
        return

    print(f"[INFO] Opening HDF5 file: {filepath}")
    images1, images2, centroids = load_h5_frames_and_centroids(filepath)
    print(f"[INFO] Loaded {images1.shape[0]} frames from 'images'")
    if images2 is not None:
        print(f"[INFO] Loaded {images2.shape[0]} frames from 'images2'")

    play_h5_video(images1, images2, centroids)

if __name__ == "__main__":
    main()
