import numpy as np
import cv2


# Rectangles are treated in this file following this convention
# Rectangle looks: A B
#                  D C
# Rectangle in array is stored : [A B C D]

def abcd_rect(pts):
    """
    Create array of 4 points and fill it to form this rectangle
    A:[SMALLEST SUM] B:[SMALLEST DIFF]
    D:[LARGEST DIFF] C:[ LARGEST SUM ]
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    d = np.diff(pts)
    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmin(d)]
    rect[2] = pts[np.argmax(s)]
    rect[3] = pts[np.argmax(d)]
    return rect


def crop_rectangle_warp(image, rect_coords):
    orig_rect = abcd_rect(rect_coords)
    (a, b, c, d) = orig_rect

    w = max(int(np.linalg.norm(a - b)), int(np.linalg.norm(c - d)))
    h = max(int(np.linalg.norm(a - d)), int(np.linalg.norm(b - c)))

    # using the form [A B C D]
    dest_rect = np.array([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
    ], dtype="float32")

    warp_matrix = cv2.getPerspectiveTransform(orig_rect, dest_rect)
    return cv2.warpPerspective(image, warp_matrix, (w, h))
