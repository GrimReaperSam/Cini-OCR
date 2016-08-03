from rawkit.raw import Raw
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


def resize(abcd, amount):
    (a, b, c, d) = abcd

    na = [a[0] - amount, a[1] - amount]
    nb = [b[0] + amount, b[1] - amount]
    nc = [c[0] + amount, c[1] + amount]
    nd = [d[0] - amount, d[1] + amount]

    return np.asarray([na, nb, nc, nd], np.float32)


def crop_rectangle_warp(image, rect_coords, ratio, amount=0):
    orig_rect = abcd_rect(rect_coords)
    if amount != 0:
        orig_rect = resize(orig_rect, amount)
    orig_rect *= ratio
    (a, b, c, d) = orig_rect

    w = max(int(np.linalg.norm(a - b)), int(np.linalg.norm(c - d)))
    h = max(int(np.linalg.norm(a - d)), int(np.linalg.norm(b - c)))

    # using the form [A B C D]
    dest_rect = np.array([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
    ], dtype="float32")

    warp_matrix = cv2.getPerspectiveTransform(orig_rect, dest_rect)
    return cv2.warpPerspective(image, warp_matrix, (w, h))


def load_raw_file_to_cv2(filename):
    """
    Load a Raw file
    :param filename:
    :return:
    """
    with Raw(filename=filename) as raw:
        raw.options.rotation = 0
        w = raw.data.contents.sizes.width
        h = raw.data.contents.sizes.height
        na = np.frombuffer(raw.to_buffer(), np.int8)
        na = na.reshape(h, w, 3).astype("uint8")
        return cv2.cvtColor(na, cv2.COLOR_BGR2RGB)


def get_enclosing_contours(binary):
    _, contours, hierarchy = cv2.findContours(binary.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    results = []
    if len(contours)>0:
        for cnt, h in zip(contours, hierarchy[0]):
            if h[3] < 0:
                results.append(cnt)
    return results
