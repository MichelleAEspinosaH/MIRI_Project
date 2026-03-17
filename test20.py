import cv2
import numpy as np
from ultralytics import YOLO
from pyorbbecsdk import *

# -----------------------------
# Load YOLO segmentation model
# -----------------------------
model = YOLO("yolov8n-seg.pt")

# -----------------------------
# Start RGB camera
# -----------------------------
rgb_cap = cv2.VideoCapture(0)

if not rgb_cap.isOpened():
    print("Could not open RGB camera")
    exit()

# -----------------------------
# Start depth pipeline
# -----------------------------
pipeline = Pipeline()
config = Config()

profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
depth_profile = profile_list.get_default_video_stream_profile()

config.enable_stream(depth_profile)
pipeline.start(config)

print("Depth stream started")

# -----------------------------
# Camera intrinsics (approx)
# Replace with real ones later
# -----------------------------
fx, fy = 600, 600
cx, cy = 320, 240


# -----------------------------
# Convert mask depth → 3D points
# -----------------------------
def depth_to_points(depth, mask):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    z = depth[ys, xs] / 1000.0
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    points = np.stack((x, y, z), axis=1)

    return points


# -----------------------------
# PCA pose estimation
# -----------------------------
def compute_axes(points):

    centroid = np.mean(points, axis=0)

    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    order = np.argsort(eigvals)

    x_axis = eigvecs[:, order[0]]
    y_axis = eigvecs[:, order[1]]
    z_axis = eigvecs[:, order[2]]

    return centroid, x_axis, y_axis, z_axis


# -----------------------------
# Project 3D → image
# -----------------------------
def project_point(point):

    X, Y, Z = point

    if Z == 0:
        return None

    u = (X * fx / Z) + cx
    v = (Y * fy / Z) + cy

    return int(u), int(v)


# -----------------------------
# Draw axes at COM
# -----------------------------
def draw_axes(img, center, axes, scale=0.05):

    cx_img, cy_img = center

    colors = [(0,0,255),(0,255,0),(255,0,0)]

    for axis, color in zip(axes, colors):

        endpoint_3d = centroid + axis * scale
        endpoint_2d = project_point(endpoint_3d)

        if endpoint_2d is None:
            continue

        cv2.arrowedLine(img,(cx_img,cy_img),endpoint_2d,color,3)


# -----------------------------
# Main loop
# -----------------------------
try:

    while True:

        ret, rgb = rgb_cap.read()
        if not ret:
            break

        frames = pipeline.wait_for_frames(100)
        depth_frame = frames.get_depth_frame()

        if depth_frame is None:
            continue

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)

        h = depth_frame.get_height()
        w = depth_frame.get_width()

        depth = depth_data.reshape((h, w))

        # -----------------------------
        # YOLO segmentation
        # -----------------------------
        results = model(rgb)[0]

        if results.masks is not None:

            masks = results.masks.data.cpu().numpy()

            for mask in masks:

                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

                points = depth_to_points(depth, mask)

                if points is None or len(points) < 100:
                    continue

                centroid, x_axis, y_axis, z_axis = compute_axes(points)

                center_img = project_point(centroid)

                if center_img is None:
                    continue

                draw_axes(
                    rgb,
                    center_img,
                    [x_axis, y_axis, z_axis]
                )

        cv2.imshow("Object COM Pose", rgb)

        if cv2.waitKey(1) == 27:
            break

finally:

    print("Stopping pipeline")

    pipeline.stop()
    rgb_cap.release()
    cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from pyorbbecsdk import *

# # ----------------------------
# # Load model
# # ----------------------------
# model = YOLO("yolov8n-seg.pt")

# # ----------------------------
# # RGB Camera (OpenCV)
# # ----------------------------
# rgb_cap = cv2.VideoCapture(0)

# # ----------------------------
# # Depth pipeline
# # ----------------------------
# pipeline = Pipeline()
# config = Config()

# profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
# depth_profile = profile_list.get_default_video_stream_profile()

# config.enable_stream(depth_profile)
# pipeline.start(config)

# # ----------------------------
# # Utility functions
# # ----------------------------

# def depth_to_points(depth, mask, fx=600, fy=600, cx=320, cy=240):
#     """Convert masked depth to 3D points"""
#     ys, xs = np.where(mask)

#     zs = depth[ys, xs] / 1000.0

#     xs3 = (xs - cx) * zs / fx
#     ys3 = (ys - cy) * zs / fy

#     points = np.stack([xs3, ys3, zs], axis=1)
#     return points


# def compute_axes(points):
#     """Compute object coordinate frame"""
#     centroid = np.mean(points, axis=0)

#     cov = np.cov(points.T)
#     eigvals, eigvecs = np.linalg.eig(cov)

#     # smallest eigenvector → shortest dimension
#     x_axis = eigvecs[:, np.argmin(eigvals)]

#     # vertical axis
#     z_axis = np.array([0,0,1])

#     # orthogonal axis
#     y_axis = np.cross(z_axis, x_axis)

#     return centroid, x_axis, y_axis, z_axis


# def draw_axes(img, center, axes, scale=100):
#     cx, cy = int(center[0]), int(center[1])

#     colors = [(0,0,255),(0,255,0),(255,0,0)]

#     for axis,color in zip(axes,colors):
#         end = (int(cx + axis[0]*scale), int(cy + axis[1]*scale))
#         cv2.arrowedLine(img,(cx,cy),end,color,3)

# # ----------------------------
# # Initial surface reference
# # ----------------------------

# surface_depth = None

# # ----------------------------
# # Main loop
# # ----------------------------

# while True:

#     ret, rgb = rgb_cap.read()
#     if not ret:
#         break

#     frames = pipeline.wait_for_frames(100)
#     depth_frame = frames.get_depth_frame()

#     depth = np.asanyarray(depth_frame.get_data())

#     if surface_depth is None:
#         surface_depth = np.median(depth)
#         print("Surface depth initialized")

#     # ----------------------------
#     # Object detection
#     # ----------------------------
#     results = model(rgb)[0]

#     if results.masks is not None:

#         masks = results.masks.data.cpu().numpy()

#         for mask in masks:

#             mask = cv2.resize(mask,(rgb.shape[1],rgb.shape[0]))
#             mask = mask > 0.5

#             pts = depth_to_points(depth,mask)

#             if len(pts) < 100:
#                 continue

#             center, x_axis, y_axis, z_axis = compute_axes(pts)

#             # project center to image
#             ys,xs = np.where(mask)
#             cx = np.mean(xs)
#             cy = np.mean(ys)

#             draw_axes(rgb,(cx,cy),[x_axis,y_axis,z_axis])

#     cv2.imshow("Pose Estimation",rgb)

#     if cv2.waitKey(1)==27:
#         break

# pipeline.stop()
# rgb_cap.release()
# cv2.destroyAllWindows()