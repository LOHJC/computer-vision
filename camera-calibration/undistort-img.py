

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json

IMG = "imgs/keyboard (8).jpeg" 

# load calibration data saved in JSON format
with open('camera_calib.json', 'r') as jf:
    calib = json.load(jf)

mtx = np.array(calib['camera_matrix'])
dist = np.array(calib['dist_coeff'])

# read input image
img = cv.imread(IMG)
if img is None:
    raise FileNotFoundError(f"could not open image {IMG}")

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

# display side-by-side using matplotlib
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
