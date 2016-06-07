import numpy as np
import cv2
import zbarlight
from PIL import Image
from kraken import binarization
import utils

HEIGHT_RESIZE = 2000.0


def detect(page):
    """
        Given a verso page detects the existence of a barcode and returns its value,
        otherwise an empty string.
        :param page: A verso cardboard
    """
    width = page.shape[1]
    ratio = page.shape[0] / HEIGHT_RESIZE
    orig = page.copy()
    page = cv2.resize(page, (int(width / ratio), int(HEIGHT_RESIZE)))

    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    grad = cv2.subtract(grad_x, grad_y)
    grad = cv2.convertScaleAbs(grad)

    blurred = cv2.blur(grad, (9, 9))
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)

    eroded = cv2.erode(close, None, iterations=4)
    dilated = cv2.dilate(eroded, None, iterations=4)

    (_, contours, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        rect = cv2.minAreaRect(c)
        barcode_box = np.int0(cv2.boxPoints(rect))

        im = utils.crop_rectangle_warp(orig, barcode_box.reshape(4, 2), ratio, amount=5)
        return _read(im)
    return None


def _read(cv2_image):
    barcode_binary = binarization.nlbin(Image.fromarray(cv2_image), zoom=1.0)
    codes = zbarlight.scan_codes('code39', barcode_binary)
    if codes:
        return codes[0].decode('utf-8')
    return ''
