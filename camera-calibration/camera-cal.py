
import numpy as np
import cv2 as cv
import os
import json

CAL_IMG_PATH = "cal-images"
PATTERN_SIZE = (7, 6)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:PATTERN_SIZE[0],0:PATTERN_SIZE[1]].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in os.listdir(CAL_IMG_PATH):
    img = cv.imread(os.path.join(CAL_IMG_PATH, fname))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv.findCirclesGrid(gray, PATTERN_SIZE, None)
 
    # If found, add object points and refine the corner locations
    if ret == True:
        objpoints.append(objp)

        # refine to subpixel accuracy using the termination criteria defined above
        corners_refined = cv.cornerSubPix(gray, corners,
                                         winSize=(3,3),
                                         zeroZone=(-1,-1),
                                         criteria=criteria)
        imgpoints.append(corners_refined)
 
        # Draw and display the corners (use refined positions)
        cv.drawChessboardCorners(img, (7,6), corners_refined, ret)
        print(f"done {fname}")
        cv.imshow('img', img)
        cv.waitKey(500)
 
cv.destroyAllWindows()

# calibrate the camera and save the params to json
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    # compute reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)
    print(f"calibration RMS error: {ret}")
    print(f"mean reprojection error: {mean_error}")
    
    # also write out a JSON-friendly version if desired
    def to_list(x):
        return np.array(x).tolist()

    json_data = {
        'camera_matrix': to_list(mtx),
        'dist_coeff': to_list(dist),
        'rvecs': [to_list(r) for r in rvecs],
        'tvecs': [to_list(t) for t in tvecs],
        'reproj_error': float(mean_error)
    }
    with open('camera_calib.json', 'w') as jf:
        json.dump(json_data, jf, indent=2)
    print('saved calibration to camera_calib.json')
else:
    print('no valid patterns detected; cannot calibrate')

