
import cv2 as cv
import os

IMG_PATH = "imgs"

def stich_images_opencv():
    imgs = []
    for img in os.listdir(IMG_PATH):
        img_path = os.path.join(IMG_PATH, img)
        image = cv.imread(img_path)
        imgs.append(image)
    
    sticher = cv.Stitcher_create()
    status, stiched = sticher.stitch(imgs)
    if status == cv.Stitcher_OK:
        cv.imshow("Stiched Image", stiched)
        cv.waitKey(0)
    else:
        print("Error stiching images")

if __name__ == "__main__":
    stich_images_opencv()