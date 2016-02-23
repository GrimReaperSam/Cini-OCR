from crop_and_skew import *
import os

for filename in os.listdir('samples'):
    image = cv2.imread("samples/%s" % filename)
    scan = crop_and_skew(image)
    cv2.imwrite("results/%s-scan.png" % filename, scan)
    # crop = crop_and_skew(scan, 300)
    # cv2.imwrite("%s-crop.png" % filename, crop)
    print "%s has been processed" % filename