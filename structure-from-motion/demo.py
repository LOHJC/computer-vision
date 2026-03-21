
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import open3d as o3d

IMG_PATH = "./imgs"
TEMPLATE_PATH = "./feature_imgs"

def fill_image(img):
    """Find the biggest contour and create a mask"""
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    
    biggest_contour = max(contours, key=cv.contourArea)
    mask = np.zeros_like(img)
    cv.drawContours(mask, [biggest_contour], 0, 255, -1)
    return mask

def common_match(matcher, template_descriptor, des1, des2, kp1, kp2):
    # 1. Get matches for both images against the template
    matches1 = matcher.knnMatch(template_descriptor, des1, k=2)
    matches2 = matcher.knnMatch(template_descriptor, des2, k=2)

    # 2. Filter for good matches (Ratio Test)
    good1 = {m.queryIdx: m.trainIdx for m, n in matches1 if m.distance < 0.75 * n.distance}
    good2 = {m.queryIdx: m.trainIdx for m, n in matches2 if m.distance < 0.75 * n.distance}

    # 3. Find common template indices
    common_template_indices = list(set(good1.keys()).intersection(set(good2.keys())))

    if len(common_template_indices) < 4:
        print("Not enough common points for RANSAC.")
        return [[], []]

    # 4. Extract points for RANSAC
    pts1 = np.float32([kp1[good1[idx]].pt for idx in common_template_indices]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[good2[idx]].pt for idx in common_template_indices]).reshape(-1, 1, 2)

    # 5. Apply RANSAC to find the "True" geometric matches
    # H is the transformation, mask is 1 for inliers and 0 for outliers
    H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)

    if mask is None:
        return [[], []]

    # 6. Filter points to return ONLY inliers
    inlier_pts1 = [tuple(pts1[i][0]) for i in range(len(mask)) if mask[i]]
    inlier_pts2 = [tuple(pts2[i][0]) for i in range(len(mask)) if mask[i]]

    print(f"RANSAC filtered: {len(inlier_pts1)} inliers out of {len(common_template_indices)} common points.")
    
    return [inlier_pts1, inlier_pts2]

if __name__ == "__main__":
    imgs = []
    for imgname in sorted(os.listdir(IMG_PATH)):
        imgpath = os.path.join(IMG_PATH, imgname)
        img = cv.imread(imgpath)
        imgs.append(img)

    feature_imgs = []
    for imgname in sorted(os.listdir(TEMPLATE_PATH)):
        imgpath = os.path.join(TEMPLATE_PATH, imgname)
        img = cv.imread(imgpath)
        feature_imgs.append(img)
    
    feature_detector = cv.SIFT.create()
    matcher = cv.BFMatcher()
    
    final_pts1 = []
    final_pts2 = []
    for template_img in feature_imgs:
        template_gray = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)
        template_kp, template_descriptor = feature_detector.detectAndCompute(template_gray, None)
        
        for i in range(len(imgs)-1):
            img1 = imgs[i].copy()
            img2 = imgs[i + 1].copy()

            kp_img1 = img1.copy()
            kp_img2 = img2.copy()

            gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

            # rough remove background
            _, bin1 = cv.threshold(gray1, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            bin1 = fill_image(bin1)
            _, bin2 = cv.threshold(gray2, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            bin2 = fill_image(bin2)

            cv.bitwise_and(gray1, bin1, gray1)
            cv.bitwise_and(gray2, bin2, gray2)

            # plt.subplot(1,2,1)
            # plt.imshow(gray1)
            # plt.subplot(1,2,2)
            # plt.imshow(gray2)
            # plt.show()

            kp1, descriptor1 = feature_detector.detectAndCompute(gray1, None)
            kp2, descriptor2 = feature_detector.detectAndCompute(gray2, None)

            cv.drawKeypoints(kp_img1, kp1, kp_img1)
            cv.drawKeypoints(kp_img2, kp2, kp_img2)
            
            kp1, kp2 = common_match(matcher, template_descriptor, descriptor1, descriptor2, kp1, kp2)

            # for img, kp, descriptor in zip([img1, img2], [kp1, kp2], [descriptor1, descriptor2]):
            #     # sift
            #     matches = matcher.knnMatch(template_descriptor, descriptor, k=2)
            #     good_matches = []
            #     for m, n in matches:
            #         if m.distance < 0.75 * n.distance:
            #             good_matches.append([m])
            #     match_res = cv.drawMatchesKnn(template_img, template_kp, img, kp, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            #     if len(good_matches) >= 4:
            #         src_pts = np.float32([ template_kp[m[0].queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            #         dst_pts = np.float32([ kp[m[0].trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
            #         H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            #         inliers = int(mask.sum()) if mask is not None else 0
            #         print(f"inliers {inliers}/{len(good_matches)}")

            #         match_mask = [[m] for m in mask.ravel().tolist()] 

            #         draw_params = dict(matchColor = (0, 255, 0), # draw matches in green
            #                     singlePointColor = None,
            #                     matchesMask = match_mask, # <--- This is the key line
            #                     flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            #         match_res = cv.drawMatchesKnn(template_img, template_kp, img, kp, 
            #                                     good_matches, None, **draw_params)
                

            #     plt.imshow(match_res)
            #     plt.show()
            
            # match the img1 and img2 kp
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
            vis[:h1, :w1] = img1
            vis[:h2, w1:w1+w2] = img2

            for i in range(len(kp1)):
                color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                
                pt1 = [int(kp1[i][0]), int(kp1[i][1])]
                pt2 = [int(kp2[i][0]), int(kp2[i][1])]

                final_pts1.append(pt1)
                final_pts2.append(pt2)

                pt2[0] += w1 # add offset
                
                cv.circle(vis, pt1, 5, color, -1)
                cv.circle(vis, pt2, 5, color, -1)
                cv.line(vis, pt1, pt2, color, 1)
            
            # plt.imshow(vis)
            # plt.show()
        
    # SFM

