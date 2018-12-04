import cv2


detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
i = 0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        image = img[y:y+h, x:x+w]  # 将当前帧含人脸部分保存为图片，注意这里存的还是彩色图片，前面检测时灰度化是为了降低计算量；这里访问的是从y位开始到y+h-1位

        fileName = 'testData/face' + str(i) + '.jpg'
        cv2.imwrite(fileName, image)
        i = i+1

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

