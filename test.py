import cv2
import numpy as np
import time
from ultralytics import YOLO
from pyorbbecsdk import *

ESC_KEY = 27

MIN_DEPTH = 20
MAX_DEPTH = 10000

FONT = cv2.FONT_HERSHEY_SIMPLEX

# --------------------------------
# LOAD YOLO SEGMENTATION MODEL
# --------------------------------

model = YOLO("yolov8n-seg.pt")


# --------------------------------
# DEPTH PROCESSING
# --------------------------------

def process_depth(depth_frame):

    depth = np.frombuffer(
        depth_frame.get_data(),
        dtype=np.uint16
    ).reshape(
        (depth_frame.get_height(), depth_frame.get_width())
    )

    depth = depth.astype(np.float32) * depth_frame.get_depth_scale()

    depth = np.where(
        (depth > MIN_DEPTH) & (depth < MAX_DEPTH),
        depth,
        0
    )

    return depth


# --------------------------------
# EXTRACT 3D POINTS FROM MASK
# --------------------------------

def mask_to_pointcloud(mask, depth):

    ys, xs = np.where(mask > 0)

    zs = depth[ys, xs]

    valid = zs > 0

    xs = xs[valid]
    ys = ys[valid]
    zs = zs[valid]

    points = np.stack([xs, ys, zs], axis=1)

    return points


# --------------------------------
# ESTIMATE POSE
# --------------------------------

def estimate_pose(points):

    if len(points) < 50:
        return None

    centroid = np.mean(points, axis=0)

    cov = np.cov(points.T)

    eigvals, eigvecs = np.linalg.eig(cov)

    orientation = eigvecs[:, np.argmax(eigvals)]

    return centroid, orientation


# --------------------------------
# DRAW RESULTS
# --------------------------------

def draw_pose(frame, centroid, orientation):

    x, y, z = centroid

    cv2.putText(
        frame,
        f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f}",
        (20, 40),
        FONT,
        0.7,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Dir:{orientation}",
        (20,70),
        FONT,
        0.6,
        (0,255,0),
        2
    )


# --------------------------------
# MAIN
# --------------------------------

def main():

    # RGB
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("RGB camera failed")
        return


    # Depth
    pipeline = Pipeline()
    config = Config()

    depth_profiles = pipeline.get_stream_profile_list(
        OBSensorType.DEPTH_SENSOR
    )

    depth_profile = depth_profiles.get_default_video_stream_profile()

    config.enable_stream(depth_profile)

    pipeline.start(config)


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        frames = pipeline.wait_for_frames(1000)

        if not frames:
            continue

        frames = frames.as_frame_set()

        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue


        depth = process_depth(depth_frame)

        depth = cv2.resize(
            depth,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )


        start = time.time()

        results = model(frame)

        inference_ms = (time.time()-start)*1000


        for r in results:

            if r.masks is None:
                continue

            masks = r.masks.data.cpu().numpy()

            for mask in masks:

                mask = cv2.resize(
                    mask,
                    (frame.shape[1],frame.shape[0])
                )

                mask = (mask>0.5).astype(np.uint8)

                points = mask_to_pointcloud(mask, depth)

                pose = estimate_pose(points)

                if pose is None:
                    continue

                centroid, orientation = pose

                draw_pose(frame, centroid, orientation)

                colored = np.zeros_like(frame)
                colored[:,:,2] = mask*255

                frame = cv2.addWeighted(frame,0.7,colored,0.3,0)


        cv2.putText(
            frame,
            f"Inference:{inference_ms:.2f}ms",
            (10,20),
            FONT,
            0.5,
            (0,0,255),
            1
        )


        cv2.imshow("Tool Pose", frame)

        if cv2.waitKey(1) in (ESC_KEY, ord("q")):
            break


    cap.release()
    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()