
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
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

    # feature_detector = cv.ORB.create()
    # matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    
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

            for img, kp, descriptor in zip([img1, img2], [kp1, kp2], [descriptor1, descriptor2]):
                # sift
                matches = matcher.knnMatch(template_descriptor, descriptor, k=2)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append([m])
                match_res = cv.drawMatchesKnn(template_img, template_kp, img, kp, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                if len(good_matches) >= 4:
                    src_pts = np.float32([ template_kp[m[0].queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp[m[0].trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                    inliers = int(mask.sum()) if mask is not None else 0
                    print(f"inliers {inliers}/{len(good_matches)}")

                    match_mask = [[m] for m in mask.ravel().tolist()] 

                    draw_params = dict(matchColor = (0, 255, 0), # draw matches in green
                                singlePointColor = None,
                                matchesMask = match_mask, # <--- This is the key line
                                flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                    match_res = cv.drawMatchesKnn(template_img, template_kp, img, kp, 
                                                good_matches, None, **draw_params)
                

                
                # orb
                # matches = matcher.match(template_descriptor, descriptor2)
                # matches = sorted(matches, key = lambda x:x.distance)
                # match_res = cv.drawMatches(template_img, template_kp, img, kp, matches, None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

                plt.imshow(match_res)
                plt.show()
            

            break
