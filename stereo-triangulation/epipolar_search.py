import cv2 as cv
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

IMG_ROOT_PATH = "bandsaw1"
IMG_LEFT_PATH = f"{IMG_ROOT_PATH}/im0.png"
IMG_RIGHT_PATH = f"{IMG_ROOT_PATH}/im1.png"
CALIB_PATH = f"{IMG_ROOT_PATH}/calib.txt"
RESIZE_FACTOR = 1.0
MATCHING_THRESHOLD = 0.5  # 0.7


# read the calib.txt
def load_K():
    with open(CALIB_PATH, "r") as f:
        lines = f.readlines()
        cam0_K = None
        cam1_K = None
        for line in lines:
            if "cam0" in line:
                inner_text = re.search(r"\[(.*?)\]", line).group(1)
                rows = inner_text.split(";")
                cam0_K = np.array(
                    [[float(val) for val in row.split()] for row in rows]
                ).reshape(3, 3)
            if "cam1" in line:
                inner_text = re.search(r"\[(.*?)\]", line).group(1)
                rows = inner_text.split(";")
                cam1_K = np.array(
                    [[float(val) for val in row.split()] for row in rows]
                ).reshape(3, 3)

            if cam0_K is not None and cam1_K is not None:
                break
        return cam0_K, cam1_K


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


def draw_epilines(img1, img2, pts1, pts2, lines):
    img_1_with_point = img1.copy()
    img_2_with_line = img2.copy()

    cv.namedWindow("Image 1 (Source Point)", cv.WINDOW_NORMAL)
    cv.namedWindow("Image 2 (Line + Target Point)", cv.WINDOW_NORMAL)

    if len(img_1_with_point.shape) == 2:
        img_1_with_point = cv.cvtColor(img_1_with_point, cv.COLOR_GRAY2BGR)
        img_2_with_line = cv.cvtColor(img_2_with_line, cv.COLOR_GRAY2BGR)

    w = img_right.shape[1]  # width of the image
    MAX_IDX = 20
    for idx, line in enumerate(lines):
        if idx >= MAX_IDX:
            break
        # 1. Select the first point to visualize as an example
        a, b, c = line[0]
        if idx < len(pts1):
            pt1 = pts1[idx]  # The raw 2D pixel coordinate in the left image
        else:
            pt1 = None
        if idx < len(pts2):
            pt2 = pts2[idx]  # The raw 2D pixel coordinate in the right image
        else:
            pt2 = None

        x0, y0 = 0, int(-c / b)
        x1, y1 = w, int(-(a * w + c) / b)

        # random color on point
        pt_color = tuple(np.random.randint(0, 255, 3).tolist())
        line_color = tuple(np.random.randint(0, 255, 3).tolist())
        if pt1 is not None:
            cv.circle(img_1_with_point, (int(pt1[0]), int(pt1[1])), 10, pt_color, -1)
        # 5. Draw the epipolar line on the RIGHT image (Green line)
        cv.line(img_2_with_line, (x0, y0), (x1, y1), line_color, 2)
        if pt2 is not None:
            cv.circle(img_2_with_line, (int(pt2[0]), int(pt2[1])), 10, pt_color, -1)

    cv.imshow("Image 1 (Source Point)", img_1_with_point)
    cv.imshow("Image 2 (Line + Target Point)", img_2_with_line)
    cv.waitKey(0)
    cv.destroyAllWindows()


def epipolar_search(img_left, img_right, fund_mat, left_seed_points):
    img_left_copy = img_left.copy()
    img_right_copy = img_right.copy()

    if len(img_left_copy.shape) == 2:
        img_left_copy = cv.cvtColor(img_left_copy, cv.COLOR_GRAY2BGR)
        img_right_copy = cv.cvtColor(img_right_copy, cv.COLOR_GRAY2BGR)

    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]
    patch_size = 10
    matches = {}  # pt_left, pt_right
    draw_epiline = False
    for pt_left in left_seed_points:
        x, y = pt_left
        left_pos = (x, y)
        half = patch_size // 2
        if x - half < 0 or x + half >= w_left or y - half < 0 or y + half >= h_left:
            continue
        x0, y0 = (
            max(0, x - half),
            max(0, y - half),
        )
        x1, y1 = (
            min(img_right.shape[1], x + half),
            min(img_right.shape[0], y + half),
        )
        # form the initial left patch
        patch_left = img_left[y0:y1, x0:x1]

        # form the potential candidates right patches
        lines_right = cv.computeCorrespondEpilines(np.array([pt_left]), 1, fund_mat)
        if draw_epiline:
            draw_epilines(img_left, img_right, [left_pos], [], lines_right)
        candidates = []
        for idx, line in enumerate(lines_right):
            a, b, c = line[0]

            for x in range(0, w_right):
                y = int(-(a * x + c) / b)
                if 0 <= y < img_right.shape[0]:
                    candidates.append((x, y))

        # find the best match among right patches
        score_thres = 0.7
        best_score, best_pt_right = float("-inf"), None
        for x, y in candidates:
            half = patch_size // 2
            if (
                x - half < 0
                or x + half >= w_right
                or y - half < 0
                or y + half >= h_right
            ):
                continue
            # Extract patch from img_right
            x0, y0 = (
                max(0, x - half),
                max(0, y - half),
            )
            x1, y1 = (
                min(img_right.shape[1], x + half),
                min(img_right.shape[0], y + half),
            )
            cand_patch = img_right[y0:y1, x0:x1]
            score = np.corrcoef(patch_left.flatten(), cand_patch.flatten())[0, 1]
            if score > score_thres and score > best_score:
                best_score = score
                best_pt_right = (x, y)

        if best_pt_right is not None:
            matches[left_pos] = best_pt_right

            pt_color = tuple(np.random.randint(0, 255, 3).tolist())
            cv.circle(img_left_copy, pt_left, 10, pt_color, -1)
            cv.circle(img_right_copy, best_pt_right, 10, pt_color, -1)

    cv.namedWindow("Left Image with Matches", cv.WINDOW_NORMAL)
    cv.namedWindow("Right Image with Matches", cv.WINDOW_NORMAL)

    cv.imshow("Left Image with Matches", img_left_copy)
    cv.imshow("Right Image with Matches", img_right_copy)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print(f"Number of matches found: {len(matches)}")


if __name__ == "__main__":
    img_left = load_image(IMG_LEFT_PATH, resize_factor=RESIZE_FACTOR)
    img_right = load_image(IMG_RIGHT_PATH, resize_factor=RESIZE_FACTOR)
    img_left_K, img_right_K = load_K()
    print(f"img_left_K: {img_left_K}")
    print(f"img_right_K: {img_right_K}")

    # match the images
    kp_left, kp_right, good_matches = match(img_left, img_right)

    # Extract the coordinates of the matched points
    points_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
    points_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches])
    print(f"points_left: {len(points_left)}")
    print(f"points_right: {len(points_right)}")

    # draw matches
    draw_match = False
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

    # find fundamental matrix
    fund_matrix, mask = cv.findFundamentalMat(
        points_left, points_right, cv.FM_RANSAC, 3, 0.99
    )
    print(f"Fundamental Matrix:\n{fund_matrix}")
    points_left = points_left[mask.ravel() == 1]
    points_right = points_right[mask.ravel() == 1]
    print(f"Filtered points - Left: {len(points_left)}, Right: {len(points_right)}")

    left_seed_points = []
    left_seed_points = cv.goodFeaturesToTrack(img_left, 50, 0.01, 10)
    left_seed_points = np.int_(left_seed_points).reshape(-1, 2)
    print(f"Left seed points: {len(left_seed_points)}")
    print(f"Left seed points: {left_seed_points.shape}")
    epipolar_search(img_left, img_right, fund_matrix, left_seed_points)
