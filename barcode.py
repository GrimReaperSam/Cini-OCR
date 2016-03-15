import numpy as np
import cv2
import zbar
from PIL import Image
import utils

HEIGHT_RESIZE = 2000.0


def detect(page):
    width = page.shape[1]
    ratio = page.shape[0] / HEIGHT_RESIZE
    orig = page.copy()
    page = cv2.resize(page, (int(width / ratio), int(HEIGHT_RESIZE)))

    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(gray, ddepth=cv2.cv.CV_32F, dx=0, dy=1, ksize=-1)

    grad = cv2.subtract(grad_x, grad_y)
    grad = cv2.convertScaleAbs(grad)

    blurred = cv2.blur(grad, (9, 9))
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)

    eroded = cv2.erode(close, None, iterations=4)
    dilated = cv2.dilate(eroded, None, iterations=4)

    (contours, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        rect = cv2.minAreaRect(c)
        barcode_box = np.int0(cv2.cv.BoxPoints(rect))

        im = utils.crop_rectangle_warp(orig, barcode_box.reshape(4, 2), ratio, amount=5)
        return _read(im)
    return None


def _read(cv2_image):

    scanner = zbar.ImageScanner()
    scanner.parse_config('enable')

    pil = Image.fromarray(cv2_image).convert('L')
    w, h = pil.size
    raw = pil.tobytes()

    zim = zbar.Image(w, h, 'Y800', raw)
    scanner.scan(zim)

    for s in zim:
        return s.data
    return ''
