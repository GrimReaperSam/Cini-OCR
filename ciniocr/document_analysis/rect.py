import cv2
import numpy as np
from ..shared import *
from .. import utils

from skimage.morphology import convex_hull_image


def crop_biggest_rectangle(image, line_threshold=None, mask=None):
    """
    :param image: A numpy array representing the full scanned document extracted from the RAW file
    :return: A subsection of the document representing the cardboard only
    """
    ratio = image.shape[0] / RESIZE_HEIGHT
    orig = image.copy()
    image = cv2.resize(image, (int(image.shape[1] / ratio), int(RESIZE_HEIGHT)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.medianBlur(gray, 9)

    if line_threshold is None:
        line_threshold = round(0.25 * min((CARDBOARD_MIN_HEIGHT + CARDBOARD_MAX_HEIGHT) / 2 / ratio,
                                          (CARDBOARD_MIN_WIDTH + CARDBOARD_MAX_WIDTH) / 2 / ratio))
    max_box = find_biggest_rectangle(gray_filtered, line_threshold, mask=mask)

    cardboard = utils.crop_rectangle_warp(orig, max_box.reshape(4, 2), ratio)
    return cardboard


def find_biggest_rectangle(gray, line_threshold, canny_threshold=50, mask=None):
    """
    Take an grayscale (already cleaned, filtered) image,
    find the HoughLines of the image and find a combinaison of the lines that matches
    the desired height and width of the rectangle.
    Return a 4x2 matrix of the rectangle coordinates
    """
    # if line_threshold is None:
    #    line_threshold = round(0.25 * min(desired_width, desired_height))

    # Extract lines
    edge_map = cv2.Canny(gray, canny_threshold, 3 * canny_threshold, apertureSize=3)
    if mask is not None:
        edge_map = edge_map*mask
    lines = cv2.HoughLines(edge_map, 1, (np.pi / 180)/2, line_threshold)

    vertical_lines = []
    horizontal_lines = []
    for l in lines[:, 0, :]:
        if np.pi / 3 < l[1] < np.pi * 2 / 3:
            horizontal_lines.append(l)
        elif np.pi * 5 / 6 < l[1] or l[1] < np.pi / 6:
            vertical_lines.append(l)

    # print(len(vertical_lines), len(horizontal_lines))

    # Get the rectangle coordinates
    return biggest_rectangle_from_lines(vertical_lines, horizontal_lines, edge_map)


def _best_lines_pair(lines, target_distance):
    """
    From a list of similarly oriented lines, find the pair that best match a distance
    """
    assert len(lines) >= 2
    best_distance = np.inf
    b_v1, b_v2 = None, None
    for i, v1 in enumerate(lines[:-1]):
        for v2 in lines[i + 1:]:
            approx_dist = np.abs(np.abs(np.abs(v1[0]) - np.abs(v2[0])) - target_distance)
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


def get_rect_coords(v_lines, h_lines):
    h1, h2 = h_lines
    v1, v2 = v_lines
    coords = []
    for h_l in h1, h2:
        for v_l in v1, v2:
            coords.append(lines_intersection(h_l, v_l))
    coords = np.vstack(coords)
    return utils.abcd_rect(coords).astype(np.int)


def biggest_rectangle_from_lines(vertical_lines, horizontal_lines, edge_map,
                                 line_width=10, angle_threshold=1.0 * np.pi / 180):
    """
    Find the biggest rectangle from a collection of lines
    :param vertical_lines:
    :param horizontal_lines:
    :param edge_map:
    :param line_width:
    :return: rectangle coords
    """
    def angle_diff(a_1, a_2):
        diff_angle = np.abs(a_1-a_2)
        diff_angle = np.modf(diff_angle/np.pi)[0]*np.pi
        return min(diff_angle, np.pi-diff_angle)

    rectangle_mask = np.zeros(edge_map.shape, dtype=np.uint8)
    best_score = 0
    best_rect = None
    # print('nb lines : {} {}'.format(len(vertical_lines), len(horizontal_lines)))
    for i, v1 in enumerate(vertical_lines[:-1]):
        for v2 in vertical_lines[i + 1:]:
            if not (angle_diff(v1[1], v2[1]) <= angle_threshold):
                continue
            for j, h1 in enumerate(horizontal_lines[:-1]):
                for h2 in horizontal_lines[j + 1:]:
                    if not (angle_diff(h1[1], h2[1]) <= angle_threshold):
                        continue
                    rect_coords = get_rect_coords((v1, v2), (h1, h2))
                    rectangle_mask[:, :] = 0
                    rectangle_mask = cv2.polylines(rectangle_mask,
                                                   rect_coords[np.newaxis], True, 1, line_width)
                    score = np.sum(edge_map * rectangle_mask)
                    if score > best_score:
                        best_score = score
                        best_rect = rect_coords
    return best_rect


def find_convex_hull_rectangle(grey_image, mask=None, canny_threshold=50):
    edge_map = cv2.Canny(grey_image, canny_threshold, 3 * canny_threshold, apertureSize=3)
    if mask is not None:
        edge_map = edge_map*mask
    hull = convex_hull_image(edge_map)
    (_, contours, _) = cv2.findContours(hull.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1
    rect = cv2.minAreaRect(contours[0])
    max_box = cv2.boxPoints(rect)
    return max_box.astype(np.int)


def crop_cardboard_old(image):
    """
    :param image: A numpy array representing the full scanned document extracted from the RAW file
    :return: A subsection of the document representing the cardboard only
    """
    ratio = image.shape[0] / RESIZE_HEIGHT
    orig = image.copy()
    image = cv2.resize(image, (int(image.shape[1] / ratio), int(RESIZE_HEIGHT)))

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
    max_box = max_box.astype(np.int)

    cardboard = utils.crop_rectangle_warp(orig, max_box.reshape(4, 2), ratio)
    return cardboard
