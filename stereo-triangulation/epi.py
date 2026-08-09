import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

IMG_LEFT_PATH = r"ambient-artroom/im0.png"
IMG_RIGHT_PATH = r"ambient-artroom/im1.png"
RESIZE_FACTOR = 1.0
MATCHING_THRESHOLD = 0.5  # 0.7


def load_image(path, resize_factor=1.0):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if resize_factor != 1.0:
        img = cv.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
    return img


def match(img_left, img_right):
    # Initialize SIFT detector
    sift = cv.SIFT_create()

    # Detect keypoints and compute descriptors
    kp_left, des_left = sift.detectAndCompute(img_left, None)
    kp_right, des_right = sift.detectAndCompute(img_right, None)

    # Use FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)  # Using KDTree for SIFT
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_left, des_right, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < MATCHING_THRESHOLD * n.distance:
            good_matches.append(m)

    return kp_left, kp_right, good_matches


if __name__ == "__main__":
    img_left = load_image(IMG_LEFT_PATH, resize_factor=RESIZE_FACTOR)
    img_right = load_image(IMG_RIGHT_PATH, resize_factor=RESIZE_FACTOR)

    # match the images
    kp_left, kp_right, good_matches = match(img_left, img_right)

    # Extract the coordinates of the matched points
    points_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
    points_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches])
    print(f"points_left: {points_left}")
    print(f"points_right: {points_right}")

    # draw matches
    draw_match = True
    if draw_match:
        img_matches = cv.drawMatches(
            img_left,
            kp_left,
            img_right,
            kp_right,
            good_matches,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv.namedWindow("Matches", cv.WINDOW_NORMAL)
        cv.namedWindow("Left Image", cv.WINDOW_NORMAL)
        cv.namedWindow("Right Image", cv.WINDOW_NORMAL)

        cv.imshow("Matches", img_matches)
        cv.imshow("Left Image", img_left)
        cv.imshow("Right Image", img_right)
        cv.waitKey(0)
        cv.destroyAllWindows()
