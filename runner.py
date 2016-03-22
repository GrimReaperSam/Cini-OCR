import os
import re
from shared import data
import cardboard
import document
import cv2

filenames = os.listdir(data['SAMPLES_DIR'])
versos = filter(lambda f: data['VERSO_SUBSTR'] in f, filenames)
rectos = filter(lambda f: data['RECTO_SUBSTR'] in f, filenames)

for filename in sorted(rectos):
    name = re.sub(data['RECTO_SUBSTR'], '', filename)
    name = re.sub('_', '', name)
    print "Begin processing image %s" % filename
    image = cv2.imread("samples/%s" % filename)
    page = document.crop_cardboard(image)
    # cv2.imwrite("results/pages/%s-page.png" % name, page)
    crop, text_section = cardboard.crop_image_and_text(page)
    cv2.imwrite("results/images/%s.png" % name, crop)
    cv2.imwrite("results/text-sections/%s.png" % name, text_section)
    print "End processing image %s" % filename
    print
