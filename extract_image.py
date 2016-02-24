import cv2
import utils
import numpy as np


def extract_image(page):
    ratio = page.shape[0] / 500.0
    orig = page.copy()
    page = cv2.resize(page, (int(page.shape[1] / ratio), 500))

    edged = cv2.Canny(page, 50, 200)
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(edged, kernel, iterations=1)

    (contours, _) = cv2.findContours(dil.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    box = None
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (_, (w, h), _) = rect
        area = w * h
        if area > page.shape[1] * 450:
            continue
        if area > max_area:
            max_area = area
            box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)

    return utils.crop_rectangle_warp(orig, box.reshape(4, 2) * ratio)
