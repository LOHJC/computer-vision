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
        pt1 = pts1[idx]  # The raw 2D pixel coordinate in the left image
        pt2 = pts2[idx]  # The raw 2D pixel coordinate in the right image

        x0, y0 = 0, int(-c / b)
        x1, y1 = w, int(-(a * w + c) / b)

        # random color on point
        pt_color = tuple(np.random.randint(0, 255, 3).tolist())
        line_color = tuple(np.random.randint(0, 255, 3).tolist())
        cv.circle(img_1_with_point, (int(pt1[0]), int(pt1[1])), 5, pt_color, -1)
        # 5. Draw the epipolar line on the RIGHT image (Green line)
        cv.line(img_2_with_line, (x0, y0), (x1, y1), line_color, 2)
        cv.circle(img_2_with_line, (int(pt2[0]), int(pt2[1])), 5, pt_color, -1)

    cv.imshow("Image 1 (Source Point)", img_1_with_point)
    cv.imshow("Image 2 (Line + Target Point)", img_2_with_line)
    cv.waitKey(0)
    cv.destroyAllWindows()


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

    draw_epiline = True
    if draw_epiline:
        # Compute epipolar lines for the right image
        lines_right = cv.computeCorrespondEpilines(points_left, 1, fund_matrix)
        draw_epilines(img_left, img_right, points_left, points_right, lines_right)

        # do the epipolar search

    # find essential matrix
    essen_matrix = img_right_K.T @ fund_matrix @ img_left_K
    print(f"Essential Matrix:\n{essen_matrix}")
    # find rotation and transation matrix
    _, R, t, pose_mask = cv.recoverPose(
        essen_matrix, points_left, points_right, cameraMatrix=np.eye(3)
    )

    use_essen_m2 = False
    if use_essen_m2:
        pts_left_norm = cv.undistortPoints(
            np.expand_dims(points_left, axis=1),
            cameraMatrix=img_left_K,
            distCoeffs=None,
        )
        pts_right_norm = cv.undistortPoints(
            np.expand_dims(points_right, axis=1),
            cameraMatrix=img_right_K,
            distCoeffs=None,
        )
        # Compute using normalized coordinates and identity matrix
        essen_matrix2, E_mask = cv.findEssentialMat(
            points1=pts_left_norm,
            points2=pts_right_norm,
            cameraMatrix=np.eye(3),
            method=cv.RANSAC,
            prob=0.99,
            threshold=0.001,  # Small threshold because coordinates are normalized
        )
        print(f"Essential Matrix (cv.findEssentialMat):\n{essen_matrix2}")
        points_left = points_left[E_mask.ravel() == 1]
        points_right = points_right[E_mask.ravel() == 1]
        print(f"Filtered points - Left: {len(points_left)}, Right: {len(points_right)}")

        # find rotation and transation matrix
        _, R, t, pose_mask = cv.recoverPose(
            essen_matrix2, pts_left_norm, pts_right_norm, cameraMatrix=np.eye(3)
        )

    print(f"Rotation Matrix (R):\n{R}")
    print(f"Translation Vector (t):\n{t}")

    # run 3d triangulation
    # Camera 1 (Left) is our world origin reference frame
    P1 = img_left_K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Camera 2 (Right) is transformed by R and t relative to Camera 1
    P2 = img_right_K @ np.hstack((R, t))

    # Triangulate using original filtered pixel points (not the normalized ones, since K is inside P1/P2)
    # We reshape points to (2, N) as required by cv.triangulatePoints
    points_4D = cv.triangulatePoints(P1, P2, points_left.T, points_right.T)

    # Convert Homogeneous coordinates (X, Y, Z, W) to standard 3D Cartesian coordinates (X, Y, Z)
    points_3D = points_4D[:3, :] / points_4D[3, :]
    points_3D = points_3D.T  # Final shape: (N, 3)

    print(f"Successfully triangulated {len(points_3D)} points in 3D Space.")
    print(f"Sample 3D Point Coordinates:\n{points_3D[:5]}")

    filtered_points_3D = points_3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot configuration
    # X = Column 0, Y = Column 1, Z = Column 2
    ax.scatter(
        filtered_points_3D[:, 0],
        filtered_points_3D[:, 1],
        filtered_points_3D[:, 2],
        c=filtered_points_3D[:, 2],
    )

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Matplotlib 3D Points Visualization")

    # Set initial viewing angle perspective
    ax.view_init(elev=-90, azim=-90)

    plt.show()
