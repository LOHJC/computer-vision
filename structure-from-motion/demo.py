import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

IMG_PATH = "./viking"
K = np.matrix("523.81 0.00 252.00; 0.00 523.81 336.00; 0.00 0.00 1.00")


MATCHING_THRESHOLD = 0.7  # 0.7


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
        img1 = imgs[i + 0].copy()
        img2 = imgs[i + 1].copy()

        kp_img1 = img1.copy()
        kp_img2 = img2.copy()

        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        kp1, kp2, good_matches = match(gray1, gray2)

        points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        print(f"points1: {len(points1)}")
        print(f"points2: {len(points2)}")

        # draw matches
        draw_match = False
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

        # find matrices
        use_method = 2
        pts1_norm = cv.undistortPoints(
            np.expand_dims(points1, axis=1),
            cameraMatrix=K,
            distCoeffs=None,
        )
        pts2_norm = cv.undistortPoints(
            np.expand_dims(points2, axis=1),
            cameraMatrix=K,
            distCoeffs=None,
        )
        fund_matrix, mask = cv.findFundamentalMat(
            points1, points2, cv.FM_RANSAC, 3, 0.99
        )

        if use_method == 1:
            E_mat = K.T @ fund_matrix @ K
            valid_idx = mask.ravel() == 1

        elif use_method == 2:
            # Compute Essential matrix directly using normalized coordinates
            E_mat, E_mask = cv.findEssentialMat(
                points1=pts1_norm,
                points2=pts2_norm,
                cameraMatrix=np.eye(3),
                method=cv.RANSAC,
                prob=0.99,
                threshold=0.001,
            )
            valid_idx = E_mask.ravel() == 1

        # Synchronize all arrays using the filtering mask
        points1 = points1[valid_idx]
        points2 = points2[valid_idx]
        pts1_norm = pts1_norm[valid_idx]
        pts2_norm = pts2_norm[valid_idx]
        print(f"Filtered clean matches: {len(points1)}")

        # find rotation and transation matrix
        _, R, t, pose_mask = cv.recoverPose(
            E_mat, pts1_norm, pts2_norm, cameraMatrix=np.eye(3)
        )

        print(f"Essential Matrix:\n{E_mat}")
        print(f"Rotation Matrix (R):\n{R}")
        print(f"Translation Vector (t):\n{t}")

        # triangulate 3D points
        points_3d = triangulate_points(points1, points2, K, K, R, t)

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

        break  # debug use: temp using 1st 2 images
