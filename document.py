import cv2
import utils
import numpy as np
import warnings
from shared import *


def validate_width(width):
    if not CARDBOARD_MIN_WIDTH <= width <= CARDBOARD_MAX_WIDTH:
        warnings.warn("The width of the cardboard is not usual : {}".format(width))


def validate_height(height):
    if not CARDBOARD_MIN_HEIGHT <= height <= CARDBOARD_MAX_HEIGHT:
        warnings.warn("The height of the cardboard is not usual : {}".format(height))


def validate_ratio(ratio):
    if not CARDBOARD_MIN_RATIO <= ratio <= CARDBOARD_MAX_RATIO:
        warnings.warn("The width/height ratio of the cardboard is not usual : {}".format(ratio))


def validate_cardboard(cardboard):
    h, w = cardboard.shape[:2]
    validate_height(cardboard.shape[0])
    validate_width(cardboard.shape[1])
    validate_ratio(float(h) / w)


def best_lines_combinaison(lines, target_distance):
    """
    From a list of similarly oriented lines, find the pair that best match a distance
    """
    assert len(lines) >= 2
    best_distance = np.inf
    b_v1, b_v2 = None, None
    for i, v1 in enumerate(lines[:-1]):
        for v2 in lines[i + 1:]:
            approx_dist = np.abs(np.abs(np.abs(v1[0])-np.abs(v2[0])) - target_distance)
            if approx_dist < best_distance:
                b_v1 = v1
                b_v2 = v2
                best_distance = approx_dist
    return b_v1, b_v2


def lines_intersection(l1, l2):
    """
    Find the position [[x,y]] of the intersection of two lines [rho_i, theta_i]
    """
    rho1, theta1 = l1
    rho2, theta2 = l2
    a = np.hstack((np.cos([theta1, theta2])[:, np.newaxis], np.sin([theta1, theta2])[:, np.newaxis]))
    b = np.array([[rho1], [rho2]])
    return np.linalg.solve(a, b).T


def find_rectangle(gray, desired_height, desired_width, line_threshold=None, canny_threshold=50):
    """
    Take an grayscale (already cleaned, filtered) image,
    find the HoughLines of the image and find a combinaison of the lines that matches
    the desired height and width of the rectangle.
    Return a 4x2 matrix of the rectangle coordinates
    """
    if line_threshold is None:
        line_threshold = round(0.3 * min(desired_width, desired_height))
    # Extract lines
    lines = cv2.HoughLines(cv2.Canny(gray, canny_threshold, 3 * canny_threshold, apertureSize=3),
                           1, np.pi / 180, line_threshold)
    vertical_lines = []
    horizontal_lines = []
    for l in lines[:, 0, :]:
        if np.pi / 4 < l[1] < np.pi * 3 / 4:
            horizontal_lines.append(l)
        else:
            vertical_lines.append(l)

    # Find the best combinaisons of lines
    v1, v2 = best_lines_combinaison(vertical_lines, desired_width)
    h1, h2 = best_lines_combinaison(horizontal_lines, desired_height)

    # Get the rectangle coordinates
    coords = []
    for h_l in h1, h2:
        for v_l in v1, v2:
            coords.append(lines_intersection(h_l, v_l))
    coords = np.stack(coords)
    return coords


def crop_cardboard(image):
    """
    :param image: A numpy array representing the full scanned document extracted from the RAW file
    :return: A subsection of the document representing the cardboard only
    """
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.medianBlur(gray, 9)

    max_box = find_rectangle(gray_filtered,
              (CARDBOARD_MIN_HEIGHT + CARDBOARD_MAX_HEIGHT)/2/ratio,
              (CARDBOARD_MIN_WIDTH + CARDBOARD_MAX_WIDTH)/2/ratio)

    cardboard = utils.crop_rectangle_warp(orig, max_box.reshape(4, 2), ratio)

    validate_cardboard(cardboard)

    return cardboard


def crop_cardboard_old(image):
    """
    :param image: A numpy array representing the full scanned document extracted from the RAW file
    :return: A subsection of the document representing the cardboard only
    """
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

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
