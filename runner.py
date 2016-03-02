from crop_and_skew import *
from split_image import *
import re
import os

for filename in sorted(os.listdir('samples')):
    name = re.sub('_recto.cr2', '', filename)
    name = re.sub('_', '', name)
    print "Begin processing image %s" % filename
    image = cv2.imread("samples/%s" % filename)
    page = crop_and_skew(image)
    cv2.imwrite("results/pages/%s-page.png" % name, page)
    crop, text_section = split_image(page)
    cv2.imwrite("results/images/%s-crop.png" % name, crop)
    cv2.imwrite("results/text-sections/%s-text-section.png" % name, text_section)
    print "End processing image %s" % filename
    print
