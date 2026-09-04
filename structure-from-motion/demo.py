import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

IMG_PATH = "./viking"
K = np.matrix("523.81 0.00 252.00; 0.00 523.81 336.00; 0.00 0.00 1.00")


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
            pts_1=pts1_norm,
            pts_2=pts2_norm,
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


def triangulate_points(points_left, points_right, K_left, K_right, R, T):
    P1 = K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K_right @ np.hstack((R, T))

    points_4d = cv.triangulatePoints(P1, P2, points_left.T, points_right.T)

    # Convert 4D homogeneous coordinates back to 3D Cartesian (x, y, z)
    points_3d = points_4d[:3, :] / points_4d[3, :]

    # Return shape (N, 3) for clean downstream use
    return points_3d.T


def match(img_left, img_right):
    # Initialize SIFT detector
    sift = cv.SIFT_create()

    # Detect keypoints and compute descriptors
    kp_left, desc_left = sift.detectAndCompute(img_left, None)
    kp_right, desc_right = sift.detectAndCompute(img_right, None)

    # Use FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)  # Using KDTree for SIFT
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_left, desc_right, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < MATCHING_THRESHOLD * n.distance:
            good_matches.append(m)

    return kp_left, kp_right, desc_left, desc_right, good_matches


# this is use to track 3d points based on im0 (img with idx 0)
# key im0  -> value = x,y,z coord
point_cloud_tracker = {}


def init_3d_points(img1, img2):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    kp_1, kp_2, desc_1, desc_2, good_matches = match(gray1, gray2)

    pts_1 = np.float32([kp_1[m.queryIdx].pt for m in good_matches])
    pts_2 = np.float32([kp_2[m.trainIdx].pt for m in good_matches])
    print(f"pts_1: {len(pts_1)}")
    print(f"pts_2: {len(pts_2)}")

    # draw matches
    draw_match = True
    if draw_match:
        img_matches = cv.drawMatches(
            img1,
            kp_1,
            img2,
            kp_2,
            good_matches,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv.namedWindow("Matches", cv.WINDOW_NORMAL)
        cv.imshow("Matches", img_matches)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # find matrices
    pts1_norm = cv.undistortPoints(
        np.expand_dims(pts_1, axis=1),
        cameraMatrix=K,
        distCoeffs=None,
    )
    pts2_norm = cv.undistortPoints(
        np.expand_dims(pts_2, axis=1),
        cameraMatrix=K,
        distCoeffs=None,
    )
    E_mat, R, t, points_3d, chosen_method = auto_select_pose_and_points(
        pts_1, pts_2, pts1_norm, pts2_norm, K, K
    )

    print(f"Essential Matrix:\n{E_mat}")
    print(f"Rotation Matrix (R):\n{R}")
    print(f"Translation Vector (t):\n{t}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot configuration
    # X = Column 0, Y = Column 1, Z = Column 2
    ax.scatter(
        points_3d[:, 0],
        points_3d[:, 1],
        points_3d[:, 2],
        c=points_3d[:, 2],
    )

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Matplotlib 3D Points Visualization")

    # Set initial viewing angle perspective
    ax.view_init(elev=-90, azim=-90)

    plt.show()


if __name__ == "__main__":
    imgs = []
    for imgname in sorted(os.listdir(IMG_PATH)):
        imgpath = os.path.join(IMG_PATH, imgname)
        img = cv.imread(imgpath)
        imgs.append(img)

    for i in range(len(imgs)):
        if i <= 1:
            if i == 0:
                img1 = imgs[i].copy()
                img2 = imgs[i + 1].copy()
                init_3d_points(img1, img2)
        else:
            # solve pnp
            pass
