import os
import time
from dataclasses import dataclass

import argparse
import cv2
import numpy as np

from pyorbbecsdk import *
from utils import frame_to_bgr_image


ESC_KEY = 27
MIN_DEPTH_MM = 20
MAX_DEPTH_MM = 10000


@dataclass
class SaveState:
    out_dir: str
    idx: int = 0


def _depth_frame_to_mm(depth_frame) -> np.ndarray:
    depth_u16 = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
        (depth_frame.get_height(), depth_frame.get_width())
    )
    depth_mm = depth_u16.astype(np.float32) * depth_frame.get_depth_scale()
    depth_mm = np.where((depth_mm > MIN_DEPTH_MM) & (depth_mm < MAX_DEPTH_MM), depth_mm, 0)
    return depth_mm.astype(np.uint16)


def _depth_mm_to_colormap(depth_mm: np.ndarray) -> np.ndarray:
    depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _profile_desc(p) -> str:
    try:
        fps = p.get_fps()
    except Exception:
        fps = "?"
    try:
        fmt = p.get_format()
    except Exception:
        fmt = "?"
    try:
        w = p.get_width()
        h = p.get_height()
    except Exception:
        w, h = "?", "?"
    return f"{w}x{h} fmt={fmt} fps={fps}"


def _list_profiles(pipeline: Pipeline) -> None:
    try:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        print("\nColor profiles:")
        for i in range(len(color_profiles)):
            p = color_profiles[i]
            print(f"- {_profile_desc(p)}")
    except Exception as e:
        print(f"\nColor profiles: unavailable ({e})")

    try:
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        print("\nDepth profiles:")
        for i in range(len(depth_profiles)):
            p = depth_profiles[i]
            print(f"- {_profile_desc(p)}")
    except Exception as e:
        print(f"\nDepth profiles: unavailable ({e})")


def _pick_color_profile(pipeline: Pipeline, req_w: int | None, req_h: int | None) -> VideoStreamProfile:
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

    # On USB2.0, MJPG is often the only workable color format at decent fps.
    preferred_formats = [OBFormat.MJPG, OBFormat.RGB, OBFormat.YUYV]
    if req_w and req_h:
        for fmt in preferred_formats:
            for p in profiles:
                try:
                    if p.get_format() == fmt and p.get_width() == req_w and p.get_height() == req_h:
                        return p
                except Exception:
                    continue

    # Fall back to default if no exact match.
    return profiles.get_default_video_stream_profile()


def _pick_depth_profile(pipeline: Pipeline, req_w: int | None, req_h: int | None):
    profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    if req_w and req_h:
        for p in profiles:
            try:
                if p.get_width() == req_w and p.get_height() == req_h:
                    return p
            except Exception:
                continue
    return profiles.get_default_video_stream_profile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color_width", type=int, default=640)
    parser.add_argument("--color_height", type=int, default=480)
    parser.add_argument("--depth_width", type=int, default=640)
    parser.add_argument("--depth_height", type=int, default=480)
    parser.add_argument("--list_profiles", action="store_true", help="Print available stream profiles and exit")
    parser.add_argument("--depth_only", action="store_true", help="Start only depth stream (debug UVC busy issues)")
    args = parser.parse_args()

    window = "RGB + Aligned Depth (D2C)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)

    pipeline = Pipeline()
    config = Config()

    if args.list_profiles:
        _list_profiles(pipeline)
        return

    enable_sync = False

    try:
        if not args.depth_only:
            color_profile = _pick_color_profile(pipeline, args.color_width, args.color_height)
            print(f"Selected color profile: {_profile_desc(color_profile)}")
            config.enable_stream(color_profile)

        depth_profile = _pick_depth_profile(pipeline, args.depth_width, args.depth_height)
        print(f"Selected depth profile: {_profile_desc(depth_profile)}")
        config.enable_stream(depth_profile)

        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    except Exception as e:
        msg = str(e)
        if "uvc_open" in msg and "already opened" in msg:
            print("Stream configuration error: color camera is already opened by another process/app.")
            print("Close any app using the camera (Zoom/Teams/Chrome tab/Photo Booth/other scripts), then unplug/replug the device and retry.")
            print(f"Details: {e}")
        else:
            print(f"Stream configuration error: {e}")
        return

    if enable_sync:
        try:
            pipeline.enable_frame_sync()
        except Exception as e:
            print(f"Sync error: {e}")

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Pipeline start error: {e}")
        return

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)  # D2C

    print("\nControls:")
    print("- 's': save RGB + aligned depth (png + npy)")
    print("- 'f': toggle hardware frame sync (if supported)")
    print("- 'q' or ESC: quit\n")

    save = SaveState(out_dir=os.path.join(os.getcwd(), "captures"))
    last_status = 0.0

    while True:
        try:
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue

            frames = align_filter.process(frames)
            if not frames:
                continue
            frames = frames.as_frame_set()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_bgr = frame_to_bgr_image(color_frame)
            if color_bgr is None:
                continue

            if depth_frame.get_format() != OBFormat.Y16:
                continue

            depth_mm = _depth_frame_to_mm(depth_frame)
            depth_cmap = _depth_mm_to_colormap(depth_mm)

            # Depth-to-color alignment should already match color resolution, but keep this safe.
            if depth_cmap.shape[:2] != color_bgr.shape[:2]:
                depth_cmap = cv2.resize(
                    depth_cmap,
                    (color_bgr.shape[1], color_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                depth_mm = cv2.resize(
                    depth_mm,
                    (color_bgr.shape[1], color_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            overlay = cv2.addWeighted(color_bgr, 0.6, depth_cmap, 0.4, 0)

            now = time.time()
            if now - last_status > 1.0:
                cy, cx = depth_mm.shape[0] // 2, depth_mm.shape[1] // 2
                print(f"Center depth (mm): {int(depth_mm[cy, cx])}")
                last_status = now

            cv2.imshow(window, overlay)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ESC_KEY):
                break
            if key in (ord("f"), ord("F")):
                enable_sync = not enable_sync
                try:
                    if enable_sync:
                        pipeline.enable_frame_sync()
                        print("Sync: Enabled")
                    else:
                        pipeline.disable_frame_sync()
                        print("Sync: Disabled")
                except Exception as e:
                    print(f"Sync toggle error: {e}")
            if key in (ord("s"), ord("S")):
                _ensure_dir(save.out_dir)
                ts = int(time.time() * 1000)
                base = f"{ts}_{save.idx:06d}"
                rgb_path = os.path.join(save.out_dir, f"{base}_rgb.png")
                depth_vis_path = os.path.join(save.out_dir, f"{base}_depth_colormap.png")
                depth_npy_path = os.path.join(save.out_dir, f"{base}_depth_mm.npy")

                cv2.imwrite(rgb_path, color_bgr)
                cv2.imwrite(depth_vis_path, depth_cmap)
                np.save(depth_npy_path, depth_mm)
                print(f"Saved: {rgb_path}")
                print(f"Saved: {depth_npy_path}")
                save.idx += 1

        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    main()

