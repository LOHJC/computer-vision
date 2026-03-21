
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

IMG_PATH = "./imgs"

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
    
    feature_detector = cv.SIFT.create()
    matcher = cv.BFMatcher()

    # feature_detector = cv.ORB.create()
    # matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    
    for i in range(len(imgs)-1):
        img1 = imgs[i].copy()
        img2 = imgs[i + 1].copy()

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

        cv.drawKeypoints(img1, kp1, img1)
        cv.drawKeypoints(img2, kp2, img2)

        # sift
        # matches = matcher.knnMatch(descriptor1, descriptor2, k=2)
        # good_matches = []
        # for m, n in matches:
        #     if m.distance < 0.7 * n.distance:
        #         good_matches.append([m])
        # match_res = cv.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # orb
        matches = matcher.match(descriptor1, descriptor2)
        matches = sorted(matches, key = lambda x:x.distance)
        match_res = cv.drawMatches(img1, kp1, img2, kp2, matches, None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        plt.subplot(1,3,1)
        plt.imshow(img1)
        plt.subplot(1,3,2)
        plt.imshow(img2)
        plt.subplot(1,3,3)
        plt.imshow(match_res)
        plt.show()

        break
