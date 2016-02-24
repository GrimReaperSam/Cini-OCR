import cv2
import numpy as np
from itertools import groupby


# line is of the form (x1, y1, x2, y2)
def length(line):
    (ox, _, dx, _) = line
    return abs(dx - ox)


def y(line):
    return line[1]


def extract_text_section(page):
    width = page.shape[1]
    ratio = page.shape[0] / 500.0
    orig = page.copy()
    page = cv2.resize(page, (int(width / ratio), 500))

    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 15, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(close.copy(), 1, np.pi / 180, 100, 100, 10)[0]

    # Filter out the vertical lines
    h_lines = [line for line in lines if line[1] == line[3]]
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
    return orig[0:int(y(max(long_4, key=y)) * ratio), 0:width]
