from crop_and_skew import *
from split_image import *
import os

for filename in os.listdir('samples'):
    print "Begin processing image %s" % filename
    image = cv2.imread("samples/%s" % filename)
    page = crop_and_skew(image)
    cv2.imwrite("results/pages/%s-page.png" % filename, page)
    crop, text_section = split_image(page)
    cv2.imwrite("results/images/%s-crop.png" % filename, crop)
    cv2.imwrite("results/text-sections/%s-text-section.png" % filename, text_section)
    print "End processing image %s" % filename
    print
