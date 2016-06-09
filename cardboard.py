import cv2
import utils
import numpy as np
import warnings
from shared import *


def get_y(line):
    return line[0][1]


def validate_image_section(x, page_width):
    page_mid_x = page_width / 2
    acceptable_x_min = page_mid_x - 0.05 * page_width
    acceptable_x_max = page_mid_x + 0.05 * page_width
    if acceptable_x_min <= x <= acceptable_x_max:
        return
    warnings.warn("The image crop isn't horizontally centered")


def validate_text_section(y_value):
    valid = False
    for (mini, maxi) in ACCEPTABLE_TEXT_SECTIONS_Y_RANGES:
        if mini <= y_value <= maxi:
            valid = True
            break
    if not valid:
        warnings.warn("The height of this text section is not usual")


def crop_image_and_text(document):
    """
    Given a recto cardboard, detects the painting and the text sections and crops them
    :param document: A recto cardboard
    :return: The painting and the text section as numpy arrays
    """
    width = document.shape[1]
    ratio = document.shape[0] / RESIZE_HEIGHT
    orig = document.copy()
    page = cv2.resize(document, (int(width / ratio), int(RESIZE_HEIGHT)))

    # -----------------------
    # Extracting image section
    # -----------------------
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(page, kernel, iterations=1)
    erokernel = np.ones((11, 11), np.uint8)
    dilated = cv2.erode(dilated, erokernel, iterations=1)

    gray = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
    grad_x = cv2.pow(grad_x, 2)

    grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)
    grad_y = cv2.pow(grad_y, 2)

    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    grad = cv2.pow(grad.astype("float32"), 0.5).astype('uint8')

    blur = cv2.GaussianBlur(grad, (7, 7), 0)
    _, bina = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    (_, contours, _) = cv2.findContours(bina.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_box = None
    max_x = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        ((x, _), (h, w), _) = rect
        area = w * h
        if h > IMAGE_HEIGHT_LIMIT or w > page.shape[1] - IMAGE_WIDTH_DELIMITER:
            continue
        if area > max_area:
            max_area = area
            max_x = x
            max_box = cv2.boxPoints(rect)
    max_box = np.int0(max_box)

    scan = utils.crop_rectangle_warp(orig, max_box.reshape(4, 2), ratio)

    validate_image_section(max_x, page.shape[1])

    # -----------------------
    # Extracting text section
    # -----------------------
    horizontal_k = np.ones((1, 3), np.uint8)
    horizontal = cv2.dilate(page, horizontal_k, iterations=1)
    gray = cv2.cvtColor(horizontal, cv2.COLOR_BGR2GRAY)
    th_gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Remove the image region
    cv2.drawContours(th_gray, [max_box], -1, (255, 255, 255), -1)
    # Remove a bit around the image (Artifacts from dilation)
    cv2.drawContours(th_gray, [max_box], -1, (255, 255, 255), IMAGE_MASK_BORDER_WIDTH)
    # Remove noisy artifacts
    closed = cv2.dilate(th_gray, horizontal_k, iterations=12)

    # Elongate remaining lines to fill the width
    erode_k = np.ones((1, width), np.uint8)
    eroded = cv2.bitwise_not(cv2.erode(closed, erode_k, iterations=2))

    lines = cv2.HoughLinesP(eroded.copy(), 1, np.pi / 180, 100, 5, 10)
    # Filter out the vertical lines
    h_lines = [line for line in lines if line[0][1] == line[0][3]]
    # Filter out lines in the lower half of the page
    h_lines = [line for line in h_lines if line[0][1] < eroded.shape[0] / 2]

    # Get lowest line
    lowest = get_y(sorted(h_lines, key=get_y, reverse=True)[0])

    validate_text_section(lowest)

    text_section = orig[0:int(lowest * ratio), 0:width]

    return scan, text_section
