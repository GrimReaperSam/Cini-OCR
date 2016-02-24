import cv2
import utils
import numpy as np
from itertools import groupby


# line is of the form (x1, y1, x2, y2)
def length(line):
    (ox, _, dx, _) = line
    return abs(dx - ox)


def y(line):
    return line[1]


def split_image(page):
    width = page.shape[1]
    ratio = page.shape[0] / 500.0
    orig = page.copy()
    page = cv2.resize(page, (int(width / ratio), 500))

    # Extracting image section
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

    scan = utils.crop_rectangle_warp(orig, box.reshape(4, 2) * ratio)

    # Extracting text section
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 15, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(close.copy(), 1, np.pi / 180, 100, 100, 10)[0]

    # Filter out the vertical lines
    h_lines = [line for line in lines if line[1] == line[3]]
    # Filter lines below the image we found above
    (a, b, _, _) = utils.abcd_rect(box)
    y_min = min(a[1], b[1])
    h_lines = [line for line in h_lines if line[1] < y_min]
    longest = sorted(h_lines, key=length, reverse=True)

    # Group by Y and take the longest line on the same y
    line_groups = groupby(longest, key=length)
    candidates = []
    for (key, data) in line_groups:
        lowest = max(data, key=y)
        candidates.append(lowest)

    # Take the longest four horizontal lines
    long_4 = sorted(candidates, key=length, reverse=True)[:4]

    # Take the lowest of these lines
    text_section = orig[0:int(y(max(long_4, key=y)) * ratio), 0:width]

    return scan, text_section