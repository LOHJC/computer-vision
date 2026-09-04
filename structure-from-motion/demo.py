import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

IMG_PATH = "./viking"
K = np.matrix("523.81 0.00 252.00; 0.00 523.81 336.00; 0.00 0.00 1.00")


MATCHING_THRESHOLD = 0.5  # 0.7


def auto_select_pose_and_points(
    pts_1,
    pts_2,
    pts1_norm,
    pts2_norm,
    desc_1,
    desc_2,
    good_matches,
    img_1_K,
    img_2_K,
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
        fund_matrix, mask1 = cv.findFundamentalMat(pts_1, pts_2, cv.FM_RANSAC, 3, 0.99)
        E_mat1 = img_1_K.T @ fund_matrix @ img_2_K

        valid1 = mask1.ravel() == 1
        _, R1, t1, pose_mask1 = cv.recoverPose(
            E_mat1, pts1_norm[valid1], pts2_norm[valid1], cameraMatrix=np.eye(3)
        )

        # Triangulate
        valid_pose1 = pose_mask1.ravel() == 255
        p3D_1 = triangulate_points(
            pts_1[valid1][valid_pose1],
            pts_2[valid1][valid_pose1],
            img_1_K,
            img_2_K,
            R1,
            t1,
        )

        # filter the descriptors
        final_mask1 = valid1.copy()
        final_mask1[valid1] = valid_pose1
        surviving_idx1 = [
            m.queryIdx for idx, m in enumerate(good_matches) if final_mask1[idx]
        ]
        clean_desc1 = desc_1[surviving_idx1]  # Maps 1:1 with p3D_1 rows

        results[1] = {
            "R": R1,
            "t": t1,
            "p3D": p3D_1,
            "inliers": len(p3D_1),
            "E": E_mat1,
            "descriptors": clean_desc1,
        }

    except Exception as e:
        print(f"Error in Method 1: {e}")
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
            pts_1[valid2][valid_pose2],
            pts_2[valid2][valid_pose2],
            img_1_K,
            img_2_K,
            R2,
            t2,
        )

        # filter the descriptors
        final_mask2 = valid2.copy()
        final_mask2[valid2] = valid_pose2
        surviving_idx2 = [
            m.queryIdx for idx, m in enumerate(good_matches) if final_mask2[idx]
        ]
        clean_desc2 = desc_2[surviving_idx2]  # Maps 1:1 with p3D_2 rows

        results[2] = {
            "R": R2,
            "t": t2,
            "p3D": p3D_2,
            "inliers": len(p3D_2),
            "E": E_mat2,
            "descriptors": clean_desc2,
        }
    except Exception as e:
        print(f"Error in Method 2: {e}")
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
        results[best_method]["descriptors"],
        best_method,
    )


def triangulate_points(pts_1, pts_2, K_left, K_right, R, T):
    P1 = K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K_right @ np.hstack((R, T))

    points_4d = cv.triangulatePoints(P1, P2, pts_1.T, pts_2.T)

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
    E_mat, R, t, points_3d, descriptors, chosen_method = auto_select_pose_and_points(
        pts_1, pts_2, pts1_norm, pts2_norm, desc_1, desc_2, good_matches, K, K
    )

    print(f"Essential Matrix:\n{E_mat}")
    print(f"Rotation Matrix (R):\n{R}")
    print(f"Translation Vector (t):\n{t}")
    print(f"3D Points:\n{len(points_3d)}")
    print(f"Descriptors:\n{len(descriptors)}")

    assert len(points_3d) == len(descriptors)

    return points_3d, descriptors, R, t


def get_2d_to_3d_correspondences(
    kp_im2, clean_descriptors, clean_points_3d, des_im2, matching_threshold=0.5
):
    """
    Matches Image 2 descriptors directly against the clean 3D descriptor bank.

    Returns:
        object_points: (N, 3) numpy array of existing 3D points
        image_points:  (N, 2) numpy array of corresponding 2D pixels in Image 2
    """
    # 1. Match Image 2's descriptors directly against our clean 3D descriptor matrix
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Note: clean_descriptors is the query matrix, des_im2 is the train matrix
    matches = flann.knnMatch(clean_descriptors, des_im2, k=2)

    object_points = []  # Will hold 3D coordinates (X, Y, Z)
    image_points = []  # Will hold 2D pixel coordinates in Image 2 (u, v)

    # 2. Extract pairs that pass Lowe's ratio test
    for m, n in matches:
        if m.distance < matching_threshold * n.distance:
            # m.queryIdx corresponds directly to the row index inside clean_descriptors.
            # Because clean_points_3d matches 1:1 by row, we use it to pull the 3D point immediately.
            clean_row_idx = m.queryIdx
            im2_kp_idx = m.trainIdx

            point_3d = clean_points_3d[clean_row_idx]
            pixel_2d = kp_im2[im2_kp_idx].pt

            object_points.append(point_3d)
            image_points.append(pixel_2d)

    return np.array(object_points, dtype=np.float32), np.array(
        image_points, dtype=np.float32
    )


if __name__ == "__main__":
    imgs = []
    for imgname in sorted(os.listdir(IMG_PATH)):
        imgpath = os.path.join(IMG_PATH, imgname)
        img = cv.imread(imgpath)
        imgs.append(img)

    clean_points_3d = None
    clean_descriptors = None
    ref_img = None
    ref_gray = None
    ref_img_next = None
    ref_gray_next = None
    R_ref = None
    t_ref = None

    for i in range(len(imgs)):
        if i <= 1:
            if i == 0:
                img1 = imgs[i].copy()
                img2 = imgs[i + 1].copy()

                ref_img = img1.copy()
                ref_gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)

                ref_img_next = img2.copy()
                ref_gray_next = cv.cvtColor(ref_img_next, cv.COLOR_BGR2GRAY)

                clean_points_3d, clean_descriptors, R_ref, t_ref = init_3d_points(
                    img1, img2
                )
            else:
                continue
        else:
            img_next = imgs[i].copy()
            gray_next = cv.cvtColor(img_next, cv.COLOR_BGR2GRAY)

            # match the new image with the reference image
            kp_ref, kp_next, desc_ref, desc_next, good_matches = match(
                ref_gray, gray_next
            )

            # Find 2D-3D correspondences
            object_points, image_points = get_2d_to_3d_correspondences(
                kp_next, clean_descriptors, clean_points_3d, desc_next
            )
            print(f"Found {len(object_points)} shared points to locate Camera {i}.")

            # solve pnp
            success, rvec, tvec, pnp_inliers = cv.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=K,  # Camera 2's Intrinsics
                distCoeffs=None,
                flags=cv.SOLVEPNP_ITERATIVE,
            )

            if not success or pnp_inliers is None:
                raise RuntimeError(
                    "PnP failed to find Camera 2 pose. Not enough overlapping points."
                )

            # 3. Convert the rotation vector to a standard 3x3 Rotation Matrix
            R2, _ = cv.Rodrigues(rvec)
            t2 = tvec

            print("Camera 2 successfully registered into the world!")
            print(f"R2:\n{R2}\nt2:\n{t2}")

            # TODO: new triangulation
            # match with another ref image
            kp_ref_next, kp_next, desc_ref_next, desc_next, good_matches_new = match(
                ref_gray_next, gray_next
            )

            # TODO: only use the points that are not in the previous descriptors
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv.FlannBasedMatcher(index_params, search_params)

            # Match clean bank (query) against current Image 1 descriptors (train)
            already_built_matches = flann.knnMatch(
                clean_descriptors, desc_ref_next, k=2
            )

            # Extract a set of Image 1 descriptor indices that are ALREADY in the 3D cloud
            already_triangulated_im1_idx = set()
            for m, n in already_built_matches:
                if m.distance < MATCHING_THRESHOLD * n.distance:
                    # m.trainIdx corresponds to the row index inside desc_ref_next
                    already_triangulated_im1_idx.add(m.trainIdx)

            # 3. Filter for completely BRAND NEW features shared between Image 1 and Image 2
            new_pts_1 = []
            new_pts_2 = []
            new_desc_1 = []  # We also save the new descriptors to expand our tracking bank!

            for m in good_matches_new:
                im1_feat_idx = m.queryIdx  # Feature index in Image 1
                im2_feat_idx = m.trainIdx  # Feature index in Image 2

                # If this Image 1 feature does NOT exist in our clean 3D bank, it's brand new!
                if im1_feat_idx not in already_triangulated_im1_idx:
                    new_pts_1.append(kp_ref_next[im1_feat_idx].pt)
                    new_pts_2.append(kp_next[im2_feat_idx].pt)
                    new_desc_1.append(desc_ref_next[im1_feat_idx])

            # Convert to numpy arrays for calculation
            new_pts_1 = np.float32(new_pts_1)
            new_pts_2 = np.float32(new_pts_2)
            new_desc_1 = np.array(new_desc_1, dtype=np.float32)

            print(f"Found {len(new_pts_1)} brand new features to grow the cloud.")

            # 4. Triangulate the new structure using Camera 1 and Camera 2
            # R_ref, t_ref = Absolute global pose matrices of your reference camera (calculated in previous loop)
            # R2, t2       = Absolute global pose matrices of your next camera (from your recent PnP step)
            if len(new_pts_1) > 0:
                # Build complete projection matrices for both cameras explicitly
                P1 = K @ np.hstack((R_ref, t_ref))
                P2 = K @ np.hstack((R2, t2))

                # Perform direct triangulation using absolute global projection grids
                points_4d = cv.triangulatePoints(P1, P2, new_pts_1.T, new_pts_2.T)
                new_points_3D = (points_4d[:3, :] / points_4d[3, :]).T

                # 5. Geometrically filter the new points (Chirality / Depth check)
                new_z_depths = new_points_3D[:, 2]
                valid_new_points_mask = (new_z_depths > 0) & (
                    new_z_depths < np.median(new_z_depths) + 2 * np.std(new_z_depths)
                )

                filtered_new_points_3D = new_points_3D[valid_new_points_mask]
                filtered_new_desc_1 = new_desc_1[valid_new_points_mask]

                # 6. Expand your master tracking arrays dynamically!
                clean_points_3d = np.vstack((clean_points_3d, filtered_new_points_3D))
                clean_descriptors = np.vstack((clean_descriptors, filtered_new_desc_1))

                # update the next ref
                ref_img_next = img_next.copy()
                ref_gray_next = gray_next.copy()
                R_ref = R2.copy()
                t_ref = t2.copy()

            print(
                f"Point cloud expanded! Total tracked 3D points now: {len(clean_points_3d)}"
            )

        print(f"clean_points_3d: {len(clean_points_3d)}")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Scatter plot configuration
        # X = Column 0, Y = Column 1, Z = Column 2
        ax.scatter(
            clean_points_3d[:, 0],
            clean_points_3d[:, 1],
            clean_points_3d[:, 2],
            c=clean_points_3d[:, 2],
        )

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Matplotlib 3D Points Visualization")

        # Set initial viewing angle perspective
        ax.view_init(elev=-90, azim=-90)

        plt.show()
