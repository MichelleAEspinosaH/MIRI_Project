import cv2
import time
import argparse
import numpy as np
import onnxruntime as ort
from pyorbbecsdk import *

ESC_KEY = 27
INPUT_WIDTH, INPUT_HEIGHT = 640, 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.5
MAX_DISPLAY_BOXES = 10

MIN_DEPTH = 20
MAX_DEPTH = 10000

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1
BLACK, RED = (0,0,0), (0,0,255)

PALETTE = [
    (255,255,255),(0,255,0),(0,0,255),(255,255,0),
    (255,0,255),(0,255,255),(128,128,0),
    (128,0,128),(0,128,128),(128,128,128)
]

# -------------------------
# Draw labels
# -------------------------
def draw_label(img, label, x, y, color, extra_line=None):

    lines = [label] if extra_line is None else [label, extra_line]
    y_offset = 0

    for text in lines:
        ts, bs = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, THICKNESS)
        w, h = ts

        cv2.rectangle(img,(x,y+y_offset),(x+w,y+y_offset+h+bs),BLACK,cv2.FILLED)

        cv2.putText(
            img,
            text,
            (x,y+y_offset+h),
            FONT_FACE,
            FONT_SCALE,
            color,
            THICKNESS,
            cv2.LINE_AA
        )

        y_offset += h+bs


# -------------------------
# YOLO Preprocess
# -------------------------
def pre_process(img):

    blob = cv2.resize(img,(INPUT_WIDTH,INPUT_HEIGHT))
    blob = cv2.cvtColor(blob,cv2.COLOR_BGR2RGB)
    blob = blob.astype(np.float32)/255.0
    blob = np.transpose(blob,(2,0,1))[np.newaxis,...]

    return blob


# -------------------------
# Depth filtering
# -------------------------
def filter_depth_outliers(depth_values,threshold=0.2):

    if depth_values.size==0:
        return depth_values

    median=np.median(depth_values)

    lower=median*(1-threshold)
    upper=median*(1+threshold)

    return depth_values[(depth_values>=lower)&(depth_values<=upper)]


# -------------------------
# Postprocess detections
# -------------------------
def post_process(img, depth_frame, outs):

    predictions=np.squeeze(outs[0])

    boxes=[]
    confidences=[]
    class_ids=[]

    img_h,img_w=img.shape[:2]

    x_factor=img_w/INPUT_WIDTH
    y_factor=img_h/INPUT_HEIGHT


    depth_data=np.frombuffer(
        depth_frame.get_data(),
        dtype=np.uint16
    ).reshape(
        (depth_frame.get_height(),depth_frame.get_width())
    )

    depth_data=depth_data.astype(np.float32)*depth_frame.get_depth_scale()

    depth_data=np.where(
        (depth_data>MIN_DEPTH)&(depth_data<MAX_DEPTH),
        depth_data,
        0
    )

    depth_data=depth_data.astype(np.uint16)


    for row in predictions:

        conf=row[4]

        if conf<CONFIDENCE_THRESHOLD:
            continue

        cls_scores=row[5:]
        class_id=np.argmax(cls_scores)

        if cls_scores[class_id]*conf<SCORE_THRESHOLD:
            continue

        cx,cy,w,h=row[0:4]

        left=int((cx-w/2)*x_factor)
        top=int((cy-h/2)*y_factor)
        width=int(w*x_factor)
        height=int(h*y_factor)

        boxes.append([left,top,width,height])
        confidences.append(float(conf))
        class_ids.append(class_id)


    indices=cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD
    )

    if len(indices)==0:
        return img


    for i in indices.flatten()[:MAX_DISPLAY_BOXES]:

        left,top,width,height=boxes[i]

        right=min(left+width,depth_data.shape[1])
        bottom=min(top+height,depth_data.shape[0])

        depth_roi=depth_data[top:bottom,left:right]

        depth_values=depth_roi.flatten()
        valid_depths=depth_values[depth_values>0]

        filtered_depths=filter_depth_outliers(valid_depths)

        if filtered_depths.size>0:
            depth_at_center=int(np.median(filtered_depths))
            depth_label=f"depth:{depth_at_center}mm"
        else:
            depth_label="depth:N/A"


        box_color=PALETTE[class_ids[i]%len(PALETTE)]

        cv2.rectangle(
            img,
            (left,top),
            (left+width,top+height),
            box_color,
            2
        )

        label=f"{classes[class_ids[i]]}:{confidences[i]:.2f}"

        draw_label(
            img,
            label,
            left,
            top-20,
            box_color,
            depth_label
        )

    return img


# -------------------------
# MAIN
# -------------------------
if __name__=="__main__":

    parser=argparse.ArgumentParser()
    args=parser.parse_args()

    with open("pyorbbecsdk/examples/object_detection/coco.names","rt") as f:
        classes=f.read().strip().split("\n")

    ort_session=ort.InferenceSession("pyorbbecsdk/examples/object_detection/models/yolov5s.onnx")
    input_name=ort_session.get_inputs()[0].name


    # -------------------------
    # RGB via OpenCV
    # -------------------------
    cap=cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open RGB camera")
        exit()

    print("RGB camera opened")


    # -------------------------
    # Depth via Orbbec SDK
    # -------------------------
    pipeline=Pipeline()

    config=Config()

    depth_profiles=pipeline.get_stream_profile_list(
        OBSensorType.DEPTH_SENSOR
    )

    depth_profile=depth_profiles.get_default_video_stream_profile()

    config.enable_stream(depth_profile)

    pipeline.start(config)


    prev_time=time.time()


    while True:

        # RGB frame
        ret,img_bgr=cap.read()

        if not ret:
            print("Frame grab failed")
            break


        # Depth frame
        frames=pipeline.wait_for_frames(1000)

        if not frames:
            continue

        frames=frames.as_frame_set()

        depth_frame=frames.get_depth_frame()

        if not depth_frame:
            continue


        # YOLO inference
        input_tensor=pre_process(img_bgr)

        start_infer=time.time()

        outputs=ort_session.run(None,{input_name:input_tensor})

        inference_time_ms=(time.time()-start_infer)*1000


        result=post_process(img_bgr.copy(),depth_frame,outputs)


        cv2.putText(
            result,
            f"Inference:{inference_time_ms:.2f}ms",
            (10,20),
            FONT_FACE,
            FONT_SCALE,
            RED,
            THICKNESS,
            cv2.LINE_AA
        )


        cv2.imshow("Output",result)

        if cv2.waitKey(1) in (ESC_KEY,ord("q")):
            break


    cap.release()
    pipeline.stop()
    cv2.destroyAllWindows()