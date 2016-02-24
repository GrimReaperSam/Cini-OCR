import cv2
import utils
import numpy as np


def crop_and_skew(image):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = cv2.resize(image, (int(image.shape[1]/ratio), 500))

    gray = cv2.bilateralFilter(image, 11, 75, 75)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 75, 200)

    (contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    box = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (_, (w, h), _) = rect
        area = w * h
        if area > max_area:
            max_area = area
            box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)

    return utils.crop_rectangle_warp(orig, box.reshape(4, 2) * ratio)
