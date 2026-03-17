import cv2
import numpy as np
import time
from ultralytics import YOLO
from pyorbbecsdk import *

ESC_KEY = 27

MIN_DEPTH = 20
MAX_DEPTH = 2500
FONT = cv2.FONT_HERSHEY_SIMPLEX

SMOOTHING = 0.8
MIN_MASK_AREA = 2000
TRACKING_DIST_THRESHOLD = 120

#model = YOLO("yolov8n-seg.pt")
model = YOLO("yolo26n-seg.pt")


# ---------------------------
# TRACKED OBJECT CLASS
# ---------------------------
class TrackedObject:

    def __init__(self, centroid, axes):
        self.centroid = centroid
        self.axes = axes
        self.missed = 0

    def update(self, centroid, axes):

        self.centroid = (
            SMOOTHING*self.centroid +
            (1-SMOOTHING)*centroid
        )

        self.axes = (
            SMOOTHING*self.axes +
            (1-SMOOTHING)*axes
        )

        self.missed = 0


tracked_objects = {}


# ---------------------------
# DEPTH PROCESSING
# ---------------------------
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


# ---------------------------
# MASK → POINT CLOUD
# ---------------------------
def mask_to_pointcloud(mask, depth):

    ys, xs = np.where(mask > 0)

    zs = depth[ys, xs]

    valid = zs > 0

    xs = xs[valid]
    ys = ys[valid]
    zs = zs[valid]

    if len(zs) < 50:
        return None

    points = np.stack([xs, ys, zs], axis=1)

    return points


# ---------------------------
# PCA POSE
# ---------------------------
def estimate_pose(points):

    centroid = np.mean(points, axis=0)

    cov = np.cov(points.T)

    eigvals, eigvecs = np.linalg.eig(cov)

    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    for i in range(3):
        if eigvecs[2, i] < 0:
            eigvecs[:, i] *= -1

    return centroid, eigvecs


# ---------------------------
# DRAW AXES
# ---------------------------
def draw_axes(frame, centroid, eigvecs, scale=80):

    cx, cy, _ = centroid
    origin = (int(cx), int(cy))

    x_axis = eigvecs[:,0]*scale
    y_axis = eigvecs[:,1]*scale
    z_axis = eigvecs[:,2]*scale

    pt_x = (int(cx+x_axis[0]), int(cy+x_axis[1]))
    pt_y = (int(cx+y_axis[0]), int(cy+y_axis[1]))
    pt_z = (int(cx+z_axis[0]), int(cy+z_axis[1]))

    cv2.line(frame, origin, pt_x, (0,0,255),3)
    cv2.line(frame, origin, pt_y, (0,255,0),3)
    cv2.line(frame, origin, pt_z, (255,0,0),3)


# ---------------------------
# UPDATE TRACKER
# ---------------------------
def update_tracker(detections):

    global tracked_objects

    for obj in tracked_objects.values():
        obj.missed += 1

    for centroid, axes in detections:

        matched = False

        for obj in tracked_objects.values():

            dist = np.linalg.norm(obj.centroid[:2] - centroid[:2])

            if dist < TRACKING_DIST_THRESHOLD:

                obj.update(centroid, axes)
                matched = True
                break

        if not matched:

            tracked_objects[len(tracked_objects)] = TrackedObject(
                centroid, axes
            )

    tracked_objects = {
        k:v for k,v in tracked_objects.items()
        if v.missed < 15
    }


# ---------------------------
# MAIN
# ---------------------------
def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open RGB camera")
        return

    pipeline = Pipeline()
    config = Config()

    depth_profiles = pipeline.get_stream_profile_list(
        OBSensorType.DEPTH_SENSOR
    )

    depth_profile = depth_profiles.get_default_video_stream_profile()

    config.enable_stream(depth_profile)

    pipeline.start(config)

    print("RGB + depth running")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames = pipeline.wait_for_frames(100)

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

        results = model(frame, imgsz=960)

        inference_ms = (time.time()-start)*1000

        detections = []

        for r in results:

            if r.masks is None:
                continue

            masks = r.masks.data.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)

            for mask, cls in zip(masks, classes):

                label = model.names[cls]

                if label == "person":
                    continue

                mask = cv2.resize(
                    mask,
                    (frame.shape[1], frame.shape[0])
                )

                mask = (mask > 0.5).astype(np.uint8)

                kernel = np.ones((7,7),np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                if np.sum(mask) < MIN_MASK_AREA:
                    continue

                points = mask_to_pointcloud(mask, depth)

                if points is None:
                    continue

                centroid, eigvecs = estimate_pose(points)

                detections.append((centroid, eigvecs))

                colored = np.zeros_like(frame)
                colored[:,:,2] = mask*255
                frame = cv2.addWeighted(frame,0.7,colored,0.3,0)

        update_tracker(detections)

        for obj in tracked_objects.values():

            draw_axes(frame, obj.centroid, obj.axes)

        cv2.putText(
            frame,
            f"Inference: {inference_ms:.1f} ms",
            (20,30),
            FONT,
            0.7,
            (0,255,0),
            2
        )

        cv2.imshow("Tracked Tool Pose", frame)

        if cv2.waitKey(1) in (ESC_KEY, ord("q")):
            break

    cap.release()
    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()