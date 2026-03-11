

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json
import os

IMG_PATH = "imgs" 
OUTPUT_PATH = "undistort-imgs"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# load calibration data saved in JSON format
with open('camera-calib.json', 'r') as jf:
    calib = json.load(jf)

mtx = np.array(calib['camera_matrix'])
dist = np.array(calib['dist_coeff'])

# read input image
for imgname in os.listdir(IMG_PATH):
    imgpath = os.path.join(IMG_PATH, imgname)
    img = cv.imread(imgpath)
    if img is None:
        raise FileNotFoundError(f"could not open image {imgpath}")

    h, w = img.shape[:2]

    # undistort using OpenCV convenience function
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv.undistort(img, mtx, dist, None, newcameramtx)

    # alternative (often higher quality) method using remapping
    mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None,
                                            newcameramtx, (w, h), cv.CV_32FC1)
    undistorted2 = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop both results to roi
    x, y, w2, h2 = roi
    undistorted = undistorted[y:y+h2, x:x+w2]
    undistorted2 = undistorted2[y:y+h2, x:x+w2]

    cv.imwrite(os.path.join(OUTPUT_PATH, imgname), undistorted2)

    # display side-by-side using matplotlib
    show = False
    if (show):
        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1)
        plt.title('distorted')
        plt.axis('off')
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.subplot(1,3,2)
        plt.title('undistorted')
        plt.axis('off')
        plt.imshow(cv.cvtColor(undistorted, cv.COLOR_BGR2RGB))
        plt.subplot(1,3,3)
        plt.title('undistorted2')
        plt.axis('off')
        plt.imshow(cv.cvtColor(undistorted2, cv.COLOR_BGR2RGB))
        plt.show()
