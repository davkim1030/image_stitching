# import the necessary packages
import os
import math
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


def resize(filename):
    img = cv2.imread(filename)
    width, height = img.shape[:2]
    if height * width * 3 <= 2 ** 25:
        return img
    i = 2
    t_height, t_width = height, width
    while t_height * t_width * 3 > 2 ** 25:
        t_height = int(t_height / math.sqrt(i))
        t_width = int(t_width / math.sqrt(i))
        i += 1
    height, width = t_height, t_width
    image = Image.open(filename)
    resize_image = image.resize((height, width))
    filename = filename[:-1 * (len(filename.split(".")[-1]) + 1)] + "_resized." + filename.split(".")[-1]
    resize_image.save(filename)
    img = cv2.imread(filename)
    os.system("del " + filename.replace("/", "\\"))
    return img


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
                help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to the output image")
args = vars(ap.parse_args())

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

# loop over the image paths, load each one, and add them to our
# images to stich list
for imagePath in imagePaths:
    image = resize(imagePath)
    images.append(image)


# initialize OpenCV's image sticher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# if the status is '0', then OpenCV successfully performed image
# stitching
if status == 0:
    # write the output stitched image to disk
    cv2.imwrite(args["output"], stitched)

    # display the output stitched image to our screen
    cv2.imshow("Stitched", stitched)
    cv2.waitKey(0)

# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
    if status == cv2.STITCHER_ERR_NEED_MORE_IMGS:
        print("[INFO] image stitching failed (1: STITCHER_ERR_NEED_MORE_IMGS)")
    elif status == cv2.STITCHER_ERR_HOMOGRAPHY_EST_FAIL:
        print("[INFO] image stitching failed (2: STITCHER_ERR_HOMOGRAPHY_EST_FAIL)")
    else:
        print("[INFO] image stitching failed (3: STITCHER_ERR_CAMERA_PARAMETERS_ADJUSTMENT_FAIL)")
