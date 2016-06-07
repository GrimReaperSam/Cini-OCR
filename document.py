import cv2
import utils
import numpy as np
import warnings
from shared import *


def validate_width(width):
    if not CARDBOARD_MIN_WIDTH <= width <= CARDBOARD_MAX_WIDTH:
        warnings.warn("The width of the cardboard is not usual")


def validate_height(height):
    if not CARDBOARD_MIN_HEIGHT <= height <= CARDBOARD_MAX_HEIGHT:
        warnings.warn("The height of the cardboard is not usual")


def validate_ratio(ratio):
    if not CARDBOARD_MIN_RATIO <= ratio <= CARDBOARD_MAX_RATIO:
        warnings.warn("The width/height ratio of the cardboard is not usual")


def validate_cardboard(cardboard):
    h, w = cardboard.shape[:2]
    validate_height(cardboard.shape[0])
    validate_width(cardboard.shape[1])
    validate_ratio(float(h)/w)


def crop_cardboard(image):
    """
    :param image: A numpy array representing the full scanned document extracted from the RAW file
    :return: A subsection of the document representing the cardboard only
    """
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = cv2.resize(image, (int(image.shape[1]/ratio), 500))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    (_, contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_box = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (_, (w, h), _) = rect
        area = w * h
        if area > max_area:
            max_area = area
            max_box = cv2.boxPoints(rect)
    max_box = np.int0(max_box)

    cardboard = utils.crop_rectangle_warp(orig, max_box.reshape(4, 2), ratio)

    validate_cardboard(cardboard)

    return cardboard
