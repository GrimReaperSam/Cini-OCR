import cv2
import utils
import numpy as np
from itertools import groupby


# line is of the form (x1, y1, x2, y2)
def length(line):
    (ox, _, dx, _) = line
    return abs(dx - ox)


def get_y(line):
    return line[1]


def crop_image_and_text(page):
    width = page.shape[1]
    ratio = page.shape[0] / 500.0
    orig = page.copy()
    page = cv2.resize(page, (int(width / ratio), 500))

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(page, kernel, iterations=1)
    dilated = cv2.erode(dilated, kernel, iterations=1)

    gray = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    grad_x = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=0, dy=1, ksize=-1)

    grad_x = cv2.convertScaleAbs(grad_x)
    grad_x = cv2.pow(grad_x, 2)
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

    # Extracting text section
    edged = cv2.Canny(page, 15, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(close.copy(), 1, np.pi / 180, 100, 100, 10)[0]

    # Filter out the vertical lines
    h_lines = [line for line in lines if line[1] == line[3]]
    # Filter lines below the image we found above
    (a, b, _, _) = utils.abcd_rect(max_box)
    y_min = min(a[1], b[1])
    h_lines = [line for line in h_lines if line[1] < y_min]
    longest = sorted(h_lines, key=length, reverse=True)

    # Group by Y and take the longest line on the same y
    line_groups = groupby(longest, key=length)
    candidates = []
    for (key, data) in line_groups:
        lowest = max(data, key=get_y)
        candidates.append(lowest)

    # Take the longest four horizontal lines
    long_4 = sorted(candidates, key=length, reverse=True)[:4]

    # Take the lowest of these lines
    text_section = orig[0:int(get_y(max(long_4, key=get_y)) * ratio), 0:width]

    return scan, text_section
