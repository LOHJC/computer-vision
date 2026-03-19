
import cv2 as cv

cap1 = cv.VideoCapture(0)
cap2 = cv.VideoCapture(1)

while(1):
    ret, frame1 = cap1.read()
    frame1 = cv.flip(frame1, 1)
    ret, frame2 = cap2.read()
    frame2 = cv.flip(frame2, 1)
    
    cv.imshow("camera1", frame1)
    cv.imshow("camera2", frame2)
    
    key = cv.waitKey(1)
    if (key == ord("q")):
        break

    if (key == ord("s")):
        cv.imwrite("frame1.png", frame1)
        cv.imwrite("frame2.png", frame2)