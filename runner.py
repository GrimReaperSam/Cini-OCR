from crop_and_skew import *
from extract_image import *
from extract_text_section import *
import os

for filename in os.listdir('samples'):
    print "Begin processing image %s" % filename
    image = cv2.imread("samples/%s" % filename)
    page = crop_and_skew(image)
    cv2.imwrite("results/%s-page.png" % filename, page)
    crop = extract_image(page)
    cv2.imwrite("results/%s-crop.png" % filename, crop)
    text_section = extract_text_section(page)
    cv2.imwrite("results/%s-text-section.png" % filename, text_section)
    print "End processing image %s" % filename
    print "\n"
