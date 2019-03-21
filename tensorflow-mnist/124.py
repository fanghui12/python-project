import cv2

cap = cv2.VideoCapture(0)

while(1):
   ret,frame = cap.read()
   cv2.rectangle(frame, (50, 50), (600, 400), (0, 0, 255), 2)
   cv2.imshow("capture", frame)
   roiImg = frame[50:400, 50:600]
   img = cv2.resize(roiImg, (275, 175))
   img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   cv2.imshow("roi",img)
   cv2.imshow("canny",cv2.Canny(img,100,200))
   if cv2.waitKey(1) & 0xFF == ord('q'):
     break
cap.release()
cv2.destroyAllWindows()