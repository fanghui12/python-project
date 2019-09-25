import cv2 as cv
import numpy as np

kernel = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="float32")

kernel2 = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]), dtype="float32")

kernel3 = np.array((
    [-2, -2, -2],
    [0, 0, 0],
    [2, 2, 2]), dtype="float32")

kernel4 = np.array((
    [-2, 0, 2],
    [-2, 0, 2],
    [-2, 0, 2]), dtype="float32")


# video = cv.VideoCapture(0)
# while(1):
#    ret,image = video.read()
#    b, g, r = cv.split(image)
#
#    imgs = np.hstack([b,g,r])
#    cv.imshow("r,g,b",imgs)
#
#    b = cv.filter2D(b, -1, kernel2)
#    g = cv.filter2D(g, -1, kernel3)
#    r = cv.filter2D(r, -1, kernel4)
#
#    img2 = cv.merge([r, g, b])
#
#    newImage = cv.resize(img2,(400,400))
#    #src3 = cv.medianBlur(newImage, 7)
#    cv.imshow("vedio",newImage)
#    if cv.waitKey(1) & 0xFF == ord('q'):
#        break

img1 = cv.imread("D:\\image\\1.jpg")
img2 = cv.imread("D:\\image\\2.jpg")
img3 = cv.imread("D:\\image\\3.jpg")
img4 = cv.imread("D:\\image\\4.jpg")
img5 = cv.imread("D:\\image\\5.jpg")

stitcher = cv.createStitcher(False)
(_result, pano) = stitcher.stitch((img1,img2,img3,img4,img5))
# pano = cv.resize(pano,(800,800))
# cv.imshow("hebing",pano)
#cv.waitKey()
cv.imwrite("D:\\image\\save.jpg",pano)




