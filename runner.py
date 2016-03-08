import os
import re
from shared import data
from crop_and_skew import *
from split_image import *

filenames = os.listdir(data['SAMPLES_DIR'])
versos = filter(lambda f: data['VERSO_SUBSTR'] in f, filenames)
rectos = filter(lambda f: data['RECTO_SUBSTR'] in f, filenames)

for filename in sorted(rectos):
    name = re.sub(data['RECTO_SUBSTR'], '', filename)
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


# IMU + Laser Scanners
# 0.6124
# 0.6424 0.25
# All but IMU

# Not IMU+GPS