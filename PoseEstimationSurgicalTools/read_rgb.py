import cv2


# Try different indexes if needed (0,1,2,3)
cap = cv2.VideoCapture(0)


if not cap.isOpened():
   print("Could not open camera")
   exit()


print("RGB camera opened")


while True:
   ret, frame = cap.read()


   if not ret:
       print("Frame grab failed")
       break


   cv2.imshow("Orbbec RGB Camera", frame)


   if cv2.waitKey(1) & 0xFF == ord('q'):
       break


cap.release()
cv2.destroyAllWindows()
