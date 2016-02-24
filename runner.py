from crop_and_skew import *
from extract_image import *
import os

for filename in os.listdir('samples'):
    image = cv2.imread("samples/%s" % filename)
    scan = crop_and_skew(image)
    cv2.imwrite("results/%s-scan.png" % filename, scan)
    crop = extract_image(scan)
    cv2.imwrite("results/%s-crop.png" % filename, crop)
    print "%s has been processed" % filename
