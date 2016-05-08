import numpy as np
import cv2
from kraken import binarization, pageseg
from PIL import Image
import pytesseract
import utils


# Returns PIL Image
def _binarize(cv2image):
    pil_image = Image.fromarray(cv2image)
    return binarization.nlbin(pil_image, zoom=1.0)


# Return OpenCV Box list
def _segment(pil_image):
    bounds = pageseg.segment(pil_image)
    boxes = list()
    for x1, y1, x2, y2 in bounds:
        mid = ((x1 + x2) / 2, (y1 + y2) / 2)
        # Adding 10 to grow borders a bit
        rect = (mid, (x2 - x1 + 10, y2 - y1 + 10), 0)
        boxes.append(np.int0(cv2.boxPoints(rect)))
    return bounds, boxes


def extract(cv2image):
    binary = _binarize(cv2image)
    boxes = _segment(binary)
    texts = list()
    for box in boxes:
        current = utils.crop_rectangle_warp(cv2image, box.reshape(4, 2), 1)
        text = pytesseract.image_to_string(Image.fromarray(current))
        text = text.split('\n', 1)[0]
        texts.append(text)
    return texts


def bound_image(cv2image):
    binary = _binarize(cv2image)
    rbounds, boxes = _segment(binary)
    bounds = list(rbounds)
    count = len(bounds)
    common = []
    for i in range(count):
        for j in range(i + 1, count):
            rect_a = bounds[i]
            rect_b = bounds[j]
            if abs(rect_a[0] - rect_b[0]) < 0.01 * cv2image.shape[1]:
                if abs(rect_a[3] - rect_b[1]) < 0.02 * cv2image.shape[0]:
                    common.append((i, j))

            elif abs(rect_a[1] - rect_b[1]) < 0.01 * cv2image.shape[0]:
                if abs(rect_a[2] - rect_b[0]) < 0.02 * cv2image.shape[1]:
                    common.append((i, j))

            elif rect_a[0] < rect_b[2] and rect_a[2] > rect_b[0] and rect_a[1] < rect_b[3] and rect_a[3] > rect_b[1]:
                common.append((i, j))

    for (f, s) in common:
        b1 = bounds[f]
        b2 = bounds[s]
        new_bound = [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]
        bounds.append(new_bound)

    indices = [e for l in common for e in l]
    for i in sorted(indices, reverse=True):
        del bounds[i]

    for x1, y1, x2, y2 in bounds:
        mid = ((x1 + x2) / 2, (y1 + y2) / 2)
        # Adding 10 to grow borders a bit
        rect = (mid, (x2 - x1 + 10, y2 - y1 + 10), 0)
        boxes.append(np.int0(cv2.boxPoints(rect)))
    return boxes
