import cv2 as cv
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

IMG_ROOT_PATH = "ladder1"
IMG_LEFT_PATH = f"{IMG_ROOT_PATH}/im0.png"
IMG_RIGHT_PATH = f"{IMG_ROOT_PATH}/im1.png"
CALIB_PATH = f"{IMG_ROOT_PATH}/calib.txt"
RESIZE_FACTOR = 1.0
MATCHING_THRESHOLD = 0.5  # 0.7


def auto_select_pose_and_points(
    points_left, points_right, pts1_norm, pts2_norm, img_left_K, img_right_K
):
    """
    Evaluates both uncalibrated (Method 1) and calibrated (Method 2) pipelines.
    Returns the winning (R, t, points_3D, selected_method) based on structural validity.
    """
    results = {}

    # ==========================================
    # PIPELINE 1: Method 1 (Fundamental -> Essential)
    # ==========================================
    try:
        fund_matrix, mask1 = cv.findFundamentalMat(
            points_left, points_right, cv.FM_RANSAC, 3, 0.99
        )
        E_mat1 = img_left_K.T @ fund_matrix @ img_right_K

        valid1 = mask1.ravel() == 1
        _, R1, t1, pose_mask1 = cv.recoverPose(
            E_mat1, pts1_norm[valid1], pts2_norm[valid1], cameraMatrix=np.eye(3)
        )

        # Triangulate
        valid_pose1 = pose_mask1.ravel() == 255
        p3D_1 = triangulate_points(
            points_left[valid1][valid_pose1],
            points_right[valid1][valid_pose1],
            img_left_K,
            img_right_K,
            R1,
            t1,
        )

        results[1] = {
            "R": R1,
            "t": t1,
            "p3D": p3D_1,
            "inliers": len(p3D_1),
            "E": E_mat1,
        }
    except Exception:
        results[1] = {"inliers": 0, "E": None}  # Failed completely

    # ==========================================
    # PIPELINE 2: Method 2 (Direct Essential Mat)
    # ==========================================
    try:
        E_mat2, mask2 = cv.findEssentialMat(
            points1=pts1_norm,
            points2=pts2_norm,
            cameraMatrix=np.eye(3),
            method=cv.RANSAC,
            prob=0.99,
            threshold=0.001,
        )

        valid2 = mask2.ravel() == 1
        _, R2, t2, pose_mask2 = cv.recoverPose(
            E_mat2, pts1_norm[valid2], pts2_norm[valid2], cameraMatrix=np.eye(3)
        )

        # Triangulate
        valid_pose2 = pose_mask2.ravel() == 255
        p3D_2 = triangulate_points(
            points_left[valid2][valid_pose2],
            points_right[valid2][valid_pose2],
            img_left_K,
            img_right_K,
            R2,
            t2,
        )

        results[2] = {
            "R": R2,
            "t": t2,
            "p3D": p3D_2,
            "inliers": len(p3D_2),
            "E": E_mat2,
        }
    except Exception:
        results[2] = {"inliers": 0, "E": None}

    # ==========================================
    # SCORING & SELECTION ENGINE
    # ==========================================
    scores = {1: -100, 2: -100}  # Initialize low scores

    for method_id in [1, 2]:
        data = results[method_id]
        if data["inliers"] < 8:  # Absolute minimum points required for stable 3D shape
            continue

        z_depths = data["p3D"][:, 2]

        # Metric A: Positive Depth Check (Chirality)
        positive_depth_ratio = np.sum(z_depths > 0) / len(z_depths)
        if (
            positive_depth_ratio < 0.85
        ):  # If more than 15% of points are behind the camera, disqualify
            continue

        # Metric B: Depth Spread (Detects the collapsed clump/smudge bug)
        # We look at the Standard Deviation of depth relative to its Median value
        median_z = np.median(z_depths)
        std_z = np.std(z_depths)
        relative_spread = std_z / (median_z + 1e-6)

        # Compute final score
        # A good reconstruction has high inlier count AND a natural structural distribution (relative_spread > 0.05)
        if (
            0.01 < relative_spread < 5.0
        ):  # Disqualify collapsed clusters (<0.01) or exploding inf values (>5.0)
            scores[method_id] = data[
                "inliers"
            ]  # Base score is number of valid structured points
        else:
            scores[method_id] = (
                data["inliers"] * 0.01
            )  # Heavily penalize structural failures

    # Pick the winner
    best_method = max(scores, key=scores.get)

    # Fallback to prevent crash if both methods scored terribly
    if scores[best_method] < 0:
        print(
            "⚠️ WARNING: Both initialization methods failed geometric verification. Defaulting to Method 1."
        )
        best_method = 1

    print(
        f"Decision Engine chosen: METHOD {best_method} (Score M1: {scores[1]:.1f}, Score M2: {scores[2]:.1f})"
    )
    return (
        results[best_method]["E"],
        results[best_method]["R"],
        results[best_method]["t"],
        results[best_method]["p3D"],
        best_method,
    )


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


def triangulate_points(points_left, points_right, K_left, K_right, R, T):
    P1 = K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K_right @ np.hstack((R, T))

    points_4d = cv.triangulatePoints(P1, P2, points_left.T, points_right.T)

    # Convert 4D homogeneous coordinates back to 3D Cartesian (x, y, z)
    points_3d = points_4d[:3, :] / points_4d[3, :]

    # Return shape (N, 3) for clean downstream use
    return points_3d.T


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

    # find fundamental matrix and essential matrix
    fund_matrix, mask = cv.findFundamentalMat(
        points_left, points_right, cv.FM_RANSAC, 3, 0.99
    )
    # Pre-undistort / normalize points using the actual K matrices
    pts1_norm = cv.undistortPoints(
        np.expand_dims(points_left, axis=1),
        cameraMatrix=img_left_K,
        distCoeffs=None,
    ).squeeze(axis=1)

    pts2_norm = cv.undistortPoints(
        np.expand_dims(points_right, axis=1),
        cameraMatrix=img_right_K,
        distCoeffs=None,
    ).squeeze(axis=1)

    # find matrices and triangulate
    E_mat, R, t, points_3D, chosen_method = auto_select_pose_and_points(
        points_left, points_right, pts1_norm, pts2_norm, img_left_K, img_right_K
    )

    draw_epiline = False
    if draw_epiline:
        # Compute epipolar lines for the right image
        lines_right = cv.computeCorrespondEpilines(points_left, 1, fund_matrix)
        draw_epilines(img_left, img_right, points_left, points_right, lines_right)

    print(f"Rotation Matrix (R):\n{R}")
    print(f"Translation Vector (t):\n{t}")

    # show  3d result
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot configuration
    # X = Column 0, Y = Column 1, Z = Column 2
    ax.scatter(
        points_3D[:, 0],
        points_3D[:, 1],
        points_3D[:, 2],
        c=points_3D[:, 2],
    )

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Matplotlib 3D Points Visualization")

    # Set initial viewing angle perspective
    ax.view_init(elev=-90, azim=-90)

    plt.show()
