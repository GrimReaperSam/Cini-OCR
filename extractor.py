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
    return boxes


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
