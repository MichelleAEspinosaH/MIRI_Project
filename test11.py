# from ultralytics import YOLO

# VIDEO = "/Users/michelleespinosa/Desktop/MIRAI_project/surgery.mp4"

# # 1) Detection (people, knife, scissors)
# det_model = YOLO("yolo26n.pt")
# det_model.predict(VIDEO, classes=[0, 43, 44], show=True)

# # 2) Segmentation (people, knife, scissors)
# seg_model = YOLO("yolo26n-seg.pt")
# seg_model.predict(VIDEO, classes=[0, 43, 44], show=True)

# # 3) Tracking (people, knife, scissors)
# trk_model = YOLO("yolo26n.pt")
# trk_model.track(VIDEO, classes=[0, 43, 44], show=True)

# # 4) Pose estimation (people)
# pose_model = YOLO("yolo26n-pose.pt")
# pose_model.predict(VIDEO, show=True)

import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# -----------------------------
# Configuration
# -----------------------------
VIDEO_PATH = "surgery.mp4"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load SAM model
# -----------------------------
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=100
)

# -----------------------------
# Video
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video")
    exit()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Generate masks
    masks = mask_generator.generate(rgb)

    object_count = len(masks)

    # Draw masks
    for mask in masks:

        segmentation = mask["segmentation"]

        color = np.random.randint(0,255,3)

        frame[segmentation] = frame[segmentation] * 0.5 + color * 0.5

    # Display object count
    cv2.putText(
        frame,
        f"Objects detected: {object_count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("SAM Object Segmentation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()