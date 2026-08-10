import cv2 as cv
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

IMG_ROOT_PATH = "artroom1"
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


def load_baseline():
    baseline = None
    with open(CALIB_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "baseline" in line:
                baseline = float(line.replace("baseline=", ""))
                break

    return baseline


def load_disp_min_max():
    vmin, vmax = None, None
    with open(CALIB_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "vmin" in line:
                vmin = float(line.replace("vmin=", ""))
            if "vmax" in line:
                vmax = float(line.replace("vmax=", ""))
    return vmin, vmax


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


def block_matching(img_left, img_right, window_size=101, max_disp=64):
    h, w = img_left.shape
    half = window_size // 2
    disparity = np.zeros((h, w), dtype=np.float32)

    for y in range(half, h - half):
        for x in range(half, w - half):
            # left patch
            patch_left = img_left[y - half : y + half + 1, x - half : x + half + 1]

            best_score = float("inf")
            best_disp = 0

            # search disparity range
            for d in range(max_disp):
                xr = x - d
                if xr - half < 0:
                    break
                patch_right = img_right[
                    y - half : y + half + 1, xr - half : xr + half + 1
                ]

                # SSD cost
                diff = patch_left.astype(np.float32) - patch_right.astype(np.float32)
                score = np.sum(diff * diff)

                if score < best_score:
                    best_score = score
                    best_disp = d

            disparity[y, x] = best_disp

    return disparity


def block_matching_vectorized(
    img_left, img_right, window_size=15, min_disp=0, max_disp=64
):
    h, w = img_left.shape
    disparity = np.zeros((h, w), dtype=np.float32)
    min_ssd = np.full((h, w), float("inf"), dtype=np.float32)

    # 1. Convert to float32 once to avoid overhead inside the loop
    img_l_f = img_left.astype(np.float32)
    img_r_f = img_right.astype(np.float32)

    # 2. Only loop over the disparities (e.g., 64 iterations instead of millions)
    for d in range(min_disp, max_disp + 1):
        # Shift the right image to the right by 'd' pixels
        # Fill empty padding on the left edge with zeros
        shifted_right = np.zeros_like(img_r_f)
        shifted_right[:, d:] = img_r_f[:, : w - d]

        # Calculate raw squared pixel differences
        pixel_diff_sq = (img_l_f - shifted_right) ** 2

        # 3. Use a Box Filter to aggregate patch costs instantly!
        # This replaces the nested loops over the window_size patch
        ssd_map = cv.boxFilter(
            pixel_diff_sq, -1, (window_size, window_size), normalize=False
        )

        # 4. Update pixel coordinates where this disparity yields a better score
        better_mask = ssd_map < min_ssd
        min_ssd[better_mask] = ssd_map[better_mask]
        disparity[better_mask] = d

    # 5. Clean up boundary margins where windows fell off the image frame
    half = window_size // 2
    disparity[:half, :] = 0
    disparity[-half:, :] = 0
    disparity[:, :half] = 0
    disparity[:, -half:] = 0

    return disparity


def compute_depth_map(disparity, focal_length, baseline, disp_min, disp_max):
    # Initialize a blank depth map with zeros
    depth_map = np.zeros_like(disparity, dtype=np.float32)

    # Identify valid disparity pixels (avoid division by zero or negative noise)
    valid_mask = (disparity >= disp_min) & (disparity <= disp_max)

    # Apply the fundamental stereo equation: Z = (B * f) / d
    depth_map[valid_mask] = (baseline * focal_length) / disparity[valid_mask]

    depth_min = (
        baseline * focal_length
    ) / disp_max  # (536.62 * 1733.74) / 142 = ~6552 mm
    depth_max = (
        baseline * focal_length
    ) / disp_min  # (536.62 * 1733.74) / 55  = ~16916 mm

    return depth_map, depth_min, depth_max


def triangulate_points(points_left, points_right, K_left, K_right, R, T):
    P1 = K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K_right @ np.hstack((R, T))

    points_4d = cv.triangulatePoints(P1, P2, points_left.T, points_right.T)

    # Convert 4D homogeneous coordinates back to 3D Cartesian (x, y, z)
    points_3d = points_4d[:3, :] / points_4d[3, :]

    # Return shape (N, 3) for clean downstream use
    return points_3d.T


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

    # rectify images
    dist_left = dist_right = np.zeros(5)  # chcked the calib.txt is 0
    R = np.eye(3)
    T = np.array([[-load_baseline()], [0], [0]], dtype=np.float64)
    R0, R1, P0, P1, Q, _, _ = cv.stereoRectify(
        img_left_K,
        dist_left,
        img_right_K,
        dist_right,
        (img_left.shape[1], img_left.shape[0]),
        R,
        T,
        flags=cv.CALIB_ZERO_DISPARITY,
        alpha=-1,
    )
    # Left camera map
    map0x, map0y = cv.initUndistortRectifyMap(
        img_left_K,
        dist_left,
        R0,
        P0,
        (img_left.shape[1], img_left.shape[0]),
        cv.CV_32FC1,
    )

    # Right camera map
    map1x, map1y = cv.initUndistortRectifyMap(
        img_right_K,
        dist_right,
        R1,
        P1,
        (img_right.shape[1], img_right.shape[0]),
        cv.CV_32FC1,
    )

    img_left_rect = cv.remap(img_left, map0x, map0y, cv.INTER_LINEAR)
    img_right_rect = cv.remap(img_right, map1x, map1y, cv.INTER_LINEAR)

    show_rectified = False
    if show_rectified:
        cv.namedWindow("Left Rectified", cv.WINDOW_NORMAL)
        cv.namedWindow("Right Rectified", cv.WINDOW_NORMAL)
        cv.imshow("Left Rectified", img_left_rect)
        cv.imshow("Right Rectified", img_right_rect)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # block matching
    disp_min, disp_max = load_disp_min_max()
    disparity = block_matching_vectorized(
        img_left_rect,
        img_right_rect,
        window_size=21,
        min_disp=int(disp_min),
        max_disp=int(disp_max),
    )
    plt.imshow(disparity)
    plt.colorbar()
    plt.show()

    # form 3d depth map
    focal_length = img_left_K[0, 0]
    baseline = load_baseline()
    # if baseline > 10:
    #     baseline /= 1000
    print(f"Focal Length: {focal_length}")
    print(f"Baseline: {baseline}")

    # Generate your depth grid
    depth, depth_min, depth_max = compute_depth_map(
        disparity, focal_length, baseline, disp_min, disp_max
    )
    plt.imshow(depth, cmap="jet_r", vmin=depth_min, vmax=depth_max)
    plt.colorbar(label="Depth (mm)")
    plt.show()
