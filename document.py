import cv2
import utils
import numpy as np


def crop_cardboard(image):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = cv2.resize(image, (int(image.shape[1]/ratio), 500))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    (contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_box = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (_, (w, h), _) = rect
        area = w * h
        if area > max_area:
            max_area = area
            max_box = cv2.cv.BoxPoints(rect)
    max_box = np.int0(max_box)

    return utils.crop_rectangle_warp(orig, max_box.reshape(4, 2), ratio)
