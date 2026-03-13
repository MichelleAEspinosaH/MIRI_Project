import time
import os
import cv2
import numpy as np
from pyorbbecsdk import *

ESC_KEY = 27
MIN_DEPTH = 20
MAX_DEPTH = 10000
RECORD_DURATION = 3  # seconds


# -----------------------------
# TEMPORAL FILTER
# -----------------------------
class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


# -----------------------------
# CREATE NEW FOLDER IF EXISTS
# -----------------------------
def create_new_folder(base_name):

    i = 1

    while True:
        folder_name = f"{base_name}_{i}"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            return folder_name

        i += 1


# -----------------------------
# MAIN
# -----------------------------
def main():

    # -----------------------------
    # RGB CAMERA
    # -----------------------------
    rgb_cap = cv2.VideoCapture(0)

    if not rgb_cap.isOpened():
        print("Could not open RGB camera")
        return

    print("RGB camera opened")


    # -----------------------------
    # DEPTH CAMERA
    # -----------------------------
    config = Config()
    pipeline = Pipeline()
    temporal_filter = TemporalFilter(alpha=0.5)

    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()

    config.enable_stream(depth_profile)

    pipeline.start(config)


    # -----------------------------
    # CREATE SAVE FOLDERS
    # -----------------------------
    rgb_dir = create_new_folder("recorded_rgb")
    depth_dir = create_new_folder("recorded_depth")

    print("Saving RGB frames to:", rgb_dir)
    print("Saving Depth frames to:", depth_dir)


    # -----------------------------
    # RECORDING STATE
    # -----------------------------
    recording = False
    record_start_time = 0
    frame_counter = 0

    print("Press 'r' to record 3 seconds. Press 'q' to quit.")


    while True:

        # -----------------------------
        # RGB FRAME
        # -----------------------------
        ret, rgb_frame = rgb_cap.read()

        if not ret:
            print("RGB frame grab failed")
            break


        # -----------------------------
        # DEPTH FRAME
        # -----------------------------
        frames = pipeline.wait_for_frames(100)

        if frames is None:
            continue

        depth_frame = frames.get_depth_frame()

        if depth_frame is None:
            continue

        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = depth_frame.get_depth_scale()

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_data = depth_data.reshape((height, width))

        depth_data = depth_data.astype(np.float32) * scale
        depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
        depth_data = depth_data.astype(np.uint16)

        depth_data = temporal_filter.process(depth_data)


        # -----------------------------
        # VISUALIZATION
        # -----------------------------
        depth_vis = cv2.normalize(
            depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)


        # -----------------------------
        # RECORDING
        # -----------------------------
        current_time = time.time()

        if recording:

            if current_time - record_start_time <= RECORD_DURATION:

                rgb_filename = os.path.join(rgb_dir, f"rgb_{frame_counter:04d}.png")
                depth_filename = os.path.join(depth_dir, f"depth_{frame_counter:04d}.png")

                cv2.imwrite(rgb_filename, rgb_frame)
                cv2.imwrite(depth_filename, depth_data)

                frame_counter += 1

            else:
                recording = False
                print(f"Recording finished. Saved {frame_counter} frames.")


        # -----------------------------
        # DISPLAY WINDOWS
        # -----------------------------
        cv2.imshow("RGB Camera", rgb_frame)
        cv2.imshow("Depth Viewer", depth_vis)


        key = cv2.waitKey(1)

        if key == ord('r') and not recording:

            recording = True
            record_start_time = time.time()
            frame_counter = 0

            print("Recording started (3 seconds)")

        if key == ord('q') or key == ESC_KEY:
            break


    # -----------------------------
    # CLEANUP
    # -----------------------------
    rgb_cap.release()
    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()