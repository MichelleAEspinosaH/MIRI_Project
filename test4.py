import cv2
import numpy as np

from pyorbbecsdk import *
from utils import frame_to_bgr_image

# --- Configuration Constants ---
ESC_KEY = 27
MIN_DEPTH = 20    # Minimum valid depth distance in mm
MAX_DEPTH = 10000 # Maximum valid depth distance in mm

def main():
    window_name = "SyncAlignViewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Initialize the pipeline and configuration objects
    pipeline = Pipeline()
    config = Config()

    # Default settings for synchronization and alignment mode
    enable_sync = False
    align_mode = 0 # 0: Depth to Color (D2C), 1: Color to Depth (C2D)
    
    try:
        # 1. Setup Color Stream Profile
        #profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        stream = pipeline.get_stream_type(OBStreamType.DEPTH_STREAM)
    except KeyboardInterrupt:
        return
    
    # Clean up resources
    cv2.destroyAllWindows()
    pipeline.stop()
    

if __name__ == "__main__":
    main()