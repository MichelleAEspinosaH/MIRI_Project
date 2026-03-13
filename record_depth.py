import time
import os
import cv2
import numpy as np
from pyorbbecsdk import *


ESC_KEY = 27
PRINT_INTERVAL = 1
MIN_DEPTH = 20
MAX_DEPTH = 10000


RECORD_DURATION = 5  # seconds




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




def main():


   config = Config()
   pipeline = Pipeline()
   temporal_filter = TemporalFilter(alpha=0.5)


   recording = False
   record_start_time = None
   frame_counter = 0


   save_dir = "test_clear_recorded_depth"
   os.makedirs(save_dir, exist_ok=True)


   try:
       profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
       depth_profile = profile_list.get_default_video_stream_profile()
       print("depth profile:", depth_profile)
       config.enable_stream(depth_profile)


   except Exception as e:
       print(e)
       return


   pipeline.start(config)


   last_print_time = time.time()


   while True:
       try:


           frames = pipeline.wait_for_frames(1000)
           if frames is None:
               continue


           depth_frame = frames.get_depth_frame()
           if depth_frame is None:
               continue


           if depth_frame.get_format() != OBFormat.Y16:
               print("depth format is not Y16")
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


           center_y = height // 2
           center_x = width // 2
           center_distance = depth_data[center_y, center_x]


           current_time = time.time()


           if current_time - last_print_time >= PRINT_INTERVAL:
               print("center distance:", center_distance)
               last_print_time = current_time


           # -------------------------
           # RECORDING LOGIC
           # -------------------------
           if recording:


               if current_time - record_start_time <= RECORD_DURATION:


                   filename = os.path.join(save_dir, f"depth_{frame_counter:04d}.png")
                   cv2.imwrite(filename, depth_data)
                   frame_counter += 1


               else:
                   recording = False
                   print("Recording finished")


           # -------------------------


           depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
           depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)


           cv2.imshow("Depth Viewer", depth_image)


           key = cv2.waitKey(1)


           if key == ord('r') and not recording:


               recording = True
               record_start_time = time.time()
               frame_counter = 0
               print("Recording started (5 seconds)")


           if key == ord('q') or key == ESC_KEY:
               break


       except KeyboardInterrupt:
           break


   cv2.destroyAllWindows()
   pipeline.stop()




if __name__ == "__main__":
   main()