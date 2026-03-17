import os
from ultralytics import YOLO
import cv2

# -----------------------------
# Config
# -----------------------------
VIDEO_PATH = "surgery4.mp4"          # your input video
OUTPUT_DIR = "runs/surgery_all_coco"
MODEL_PATH = "yolov8n-seg.pt"      # COCO-pretrained segmentation model
TRACKER_CFG = "bytetrack.yaml"
CONF_THRES = 0.4
IOU_THRES = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
model = YOLO(MODEL_PATH)

print("Model classes (id: name):")
for cid, cname in model.names.items():
    print(f"{cid}: {cname}")

# -----------------------------
# Read video metadata
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

output_video_path = os.path.join(OUTPUT_DIR, "surgery_all_coco_tracked4.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# -----------------------------
# Detect + segment + track (all classes)
# -----------------------------
#TARGET_CLASSES = [0, 43, 76]  # person, knife, scissors
TARGET_CLASSES = [43, 76]  # person, knife, scissors

results_generator = model.track(
    source=VIDEO_PATH,
    stream=True,
    tracker=TRACKER_CFG,
    conf=0.25,        # lower threshold to catch small tools
    iou=IOU_THRES,
    classes=TARGET_CLASSES,
    #tracker=TRACKER_CFG,
    save=False,
    verbose=False,
)

frame_index = 0
for results in results_generator:
    frame = results.orig_img

    boxes = results.boxes
    names = results.names

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        print(f"\nFrame {frame_index}:")
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box
            cls_id = cls_ids[i]
            conf = confidences[i]
            cls_name = names.get(cls_id, str(cls_id))
            tid = track_ids[i] if track_ids is not None and i < len(track_ids) else -1

            print(
                f"  TrackID={tid:>3}  "
                f"Class={cls_name} (id={cls_id})  "
                f"Conf={conf:.2f}  "
                f"BBox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
            )

    # Draw all detections + masks + track IDs
    annotated_frame = results.plot()
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_index}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    writer.write(annotated_frame)
    frame_index += 1

writer.release()
print(f"\nDone. Annotated video saved to: {output_video_path}")