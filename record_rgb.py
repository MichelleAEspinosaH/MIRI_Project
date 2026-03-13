import cv2
import time

# Try different indexes if needed (0,1,2,3)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

print("RGB camera opened")

# Get camera properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

recording = False
start_time = None
out = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame grab failed")
        break

    cv2.imshow("Orbbec RGB Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'r' to start recording
    if key == ord('r') and not recording:
        print("Recording for 3 seconds...")
        recording = True
        start_time = time.time()

        out = cv2.VideoWriter(
            "recorded_rgb.avi",
            cv2.VideoWriter_fourcc(*'XVID'),
            fps if fps > 0 else 30,
            (width, height)
        )

    # Record frames
    if recording:
        out.write(frame)

        # Stop after 3 seconds
        if time.time() - start_time >= 3:
            print("Recording finished")
            recording = False
            out.release()

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()