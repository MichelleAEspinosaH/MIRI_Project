import cv2
import numpy as np
import torch
from pyorbbecsdk import *

# -----------------------------
# RGB CAMERA (OpenCV)
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("RGB camera not found")
    exit()

# -----------------------------
# DEPTH CAMERA (Orbbec)
# -----------------------------
pipeline = Pipeline()
config = Config()

depth_profiles = pipeline.get_stream_profile_list(
    OBSensorType.DEPTH_SENSOR
)

depth_profile = depth_profiles.get_default_video_stream_profile()

config.enable_stream(depth_profile)

pipeline.start(config)

# -----------------------------
# YOLO MODEL
# -----------------------------
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# -----------------------------
# TRACKER
# -----------------------------
tracker = cv2.TrackerCSRT_create()
tracking_initialized = False

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    # RGB frame
    ret, frame = cap.read()
    if not ret:
        break

    # DEPTH frame
    frames = pipeline.wait_for_frames(100)
    depth_frame = frames.get_depth_frame()

    depth_image = None
    if depth_frame:
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth_image = depth_data.reshape(
            depth_frame.get_height(),
            depth_frame.get_width()
        )

    # -----------------------------
    # DETECTION
    # -----------------------------
    if not tracking_initialized:

        results = model(frame)

        detections = results.xyxy[0]

        if len(detections) > 0:

            x1, y1, x2, y2 = detections[0][:4].cpu().numpy()

            bbox = (
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1)
            )

            tracker.init(frame, bbox)
            tracking_initialized = True

    # -----------------------------
    # TRACKING
    # -----------------------------
    else:

        success, box = tracker.update(frame)

        if success:

            x, y, w, h = [int(v) for v in box]

            cx = int(x + w / 2)
            cy = int(y + h / 2)

            depth_value = None

            if depth_image is not None:
                if cy < depth_image.shape[0] and cx < depth_image.shape[1]:
                    depth_value = depth_image[cy, cx]

            # draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

            # draw center
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

            if depth_value is not None:
                cv2.putText(
                    frame,
                    f"Depth: {depth_value} mm",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

    # -----------------------------
    # DISPLAY
    # -----------------------------
    cv2.imshow("RGB Tracking + Depth", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
pipeline.stop()
cv2.destroyAllWindows()