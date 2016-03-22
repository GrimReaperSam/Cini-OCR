import cv2
import utils
import numpy as np
import warnings

ACCEPTABLE_Y_RANGES = [(60, 80), (130, 140)]


# line is of the form (x1, y1, x2, y2)
def length(line):
    (ox, _, dx, _) = line
    return abs(dx - ox)


def get_y(line):
    return line[1]


def validate_text_section(y_value):
    valid = False
    for (mini, maxi) in ACCEPTABLE_Y_RANGES:
        if mini <= y_value <= maxi:
            valid = True
            break
    if not valid:
        warnings.warn('THIS TEXT SECTION IS WEIRD YO!')


def crop_image_and_text(page):
    width = page.shape[1]
    ratio = page.shape[0] / 500.0
    orig = page.copy()
    page = cv2.resize(page, (int(width / ratio), 500))

    # -----------------------
    # Extracting image section
    # -----------------------
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(page, kernel, iterations=1)
    dilated = cv2.erode(dilated, kernel, iterations=1)

    gray = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    grad_x = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_x = cv2.pow(grad_x, 2)

    grad_y = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=0, dy=1, ksize=-1)
    grad_y = cv2.convertScaleAbs(grad_y)
    grad_y = cv2.pow(grad_y, 2)

    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    grad = cv2.pow(grad.astype("float32"), 0.5).astype("uint8")

    blur = cv2.GaussianBlur(grad, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh_kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.erode(thresh, thresh_kernel, iterations=1)

    (contours, _) = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_box = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (_, (h, w), _) = rect
        area = w * h
        if h > 450 or w > page.shape[1] - 25:
            continue
        if area > max_area:
            max_area = area
            max_box = cv2.cv.BoxPoints(rect)
    max_box = np.int0(max_box)

    scan = utils.crop_rectangle_warp(orig, max_box.reshape(4, 2), ratio)

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
    cv2.drawContours(th_gray, [max_box], -1, (255, 255, 255), 10)
    # Remove noisy artifacts
    closed = cv2.dilate(th_gray, horizontal_k, iterations=12)

    # Elongate remaining lines to fill the width
    erode_k = np.ones((1, width), np.uint8)
    eroded = cv2.bitwise_not(cv2.erode(closed, erode_k, iterations=2))

    lines = cv2.HoughLinesP(eroded.copy(), 1, np.pi / 180, 100, 5, 10)[0]
    # Filter out the vertical lines
    h_lines = [line for line in lines if line[1] == line[3]]
    h_lines = [line for line in h_lines if line[1] < eroded.shape[0] / 2]

    # Get lowest line
    lowest = get_y(sorted(h_lines, key=get_y, reverse=True)[0])

    validate_text_section(lowest)

    text_section = orig[0:int(lowest * ratio), 0:width]

    return scan, text_section
