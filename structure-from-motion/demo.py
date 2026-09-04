import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

IMG_PATH = "./viking"
K = np.matrix("523.81 0.00 252.00; 0.00 523.81 336.00; 0.00 0.00 1.00")


MATCHING_THRESHOLD = 0.5  # 0.7


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
    imgs = []
    for imgname in sorted(os.listdir(IMG_PATH)):
        imgpath = os.path.join(IMG_PATH, imgname)
        img = cv.imread(imgpath)
        imgs.append(img)

    for i in range(len(imgs) - 1):
        img1 = imgs[i].copy()
        img2 = imgs[i + 1].copy()

        kp_img1 = img1.copy()
        kp_img2 = img2.copy()

        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        kp1, kp2, good_matches = match(gray1, gray2)

        break  # debug use: temp using 1st 2 images

    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    print(f"points1: {len(points1)}")
    print(f"points2: {len(points2)}")

    # draw matches
    draw_match = True
    if draw_match:
        img_matches = cv.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            good_matches,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv.namedWindow("Matches", cv.WINDOW_NORMAL)
        cv.imshow("Matches", img_matches)
        cv.waitKey(0)
        cv.destroyAllWindows()
