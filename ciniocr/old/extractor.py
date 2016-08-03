import cv2
import numpy as np
import pytesseract
from PIL import Image
from kraken import binarization, pageseg

from ciniocr.document_analysis import frangi_filter
from ciniocr import utils


class TextBound(object):
    def __init__(self, text, text_bound, area_bound=None):
        self.text = text
        self.text_bound = text_bound
        self.area_bound = area_bound
        if area_bound is None:
            self.warning = True

    def __repr__(self):
        return 'TextBound({})'.format(self.text)


# Returns PIL Image
def _binarize(cv2image):
    pil_image = Image.fromarray(cv2image)
    return binarization.nlbin(pil_image, zoom=1.0)


def bound_image(cv2image):
    """
    :param cv2image: Numpy array representing the text section of the cardboard
    :return: The detected regions that might contain text according to the kraken page segmenter
    """
    binary = _binarize(cv2image)
    rbounds = pageseg.segment(binary)
    bounds = list(rbounds)
    count = len(bounds)
    common = []
    for i in range(count):
        for j in range(i + 1, count):
            rect_a = bounds[i]
            rect_b = bounds[j]
            # TODO Fix merging algorithm
            if abs(rect_a[0] - rect_b[0]) < 0.03 * cv2image.shape[1] and abs(rect_a[3] - rect_b[1]) < 0.03 * cv2image.shape[0]:
                common.append((i, j))

            elif abs(rect_a[1] - rect_b[1]) < 0.03 * cv2image.shape[0] and abs(rect_a[2] - rect_b[0]) < 0.03 * cv2image.shape[1]:
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

    boxes = []
    for x1, y1, x2, y2 in bounds:
        mid = ((x1 + x2) / 2, (y1 + y2) / 2)
        # Adding 10 to grow borders a bit
        rect = (mid, (x2 - x1 + 10, y2 - y1 + 10), 0)
        boxes.append(np.int0(cv2.boxPoints(rect)))
    return boxes


def text_bounds(cv2image):
    """
    :param cv2image: Numpy array representing the text section of the cardboard
    :return: A list of text bounds containing the extracted text, its location and its surrounding area if possible
    """
    binary = _binarize(cv2image)

    RESIZE_HEIGHT = 500.0
    orig = np.asarray(binary)
    width = orig.shape[1]
    ratio = orig.shape[0] / RESIZE_HEIGHT
    npbin = cv2.resize(orig, (int(width / ratio), int(RESIZE_HEIGHT)))

    kernel = np.ones((5, 5), np.uint8)
    ppbin = cv2.erode(npbin, kernel, iterations=1)

    frf = frangi_filter.frangi_filter_2d(ppbin, FrangiScaleRange=np.array([3, 4]))
    binnn = (frf < 0.01).astype('uint8')

    dilated = cv2.morphologyEx(binnn, cv2.MORPH_OPEN, kernel)

    (_, contours, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = [cv2.minAreaRect(cnt) for cnt in contours]
    boxs = [np.int0(cv2.boxPoints(rt)) for rt in rects if rt[1][0] > 50 and rt[1][1] > 50]

    boxsr = [np.int0(box * ratio) for box in boxs]
    boxsr = [np.int0(utils.abcd_rect(box)) for box in boxsr]

    im_bounds = bound_image(cv2image)

    text_boundaries = []

    for bb in im_bounds:
        inside = False
        my_area = None

        current = utils.crop_rectangle_warp(orig, bb.reshape(4, 2), 1)
        text = pytesseract.image_to_string(Image.fromarray(current))
        text = text.split('\n', 1)[0]
        for area in boxsr:
            if is_inside(bb, area):
                inside = True
                my_area = area
        if inside:
            text_boundaries.append(TextBound(text, bb.tolist(), my_area.tolist()))
        else:
            text_boundaries.append(TextBound(text, bb.tolist()))
    return text_boundaries


def is_inside(box, area):
    if box[0][0] > area[0][0] and box[0][1] > area[0][1]:
        if box[2][0] < area[2][0] and box[2][1] < area[2][1]:
            return True
    return False

