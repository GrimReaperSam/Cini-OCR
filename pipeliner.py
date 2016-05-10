import argparse
import json
import jsonpickle
import os
import re

import cv2

import barcode
import cardboard
import document
import raw_converter
from info import Info
from shared import *

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--raws", required=True,
                help="Folder with raw images")
ap.add_argument("-d", "--destination", required=True,
                help="Folder where the results will be saved")
ap.add_argument("-s", "--skip-processed", required=False, default=False, help="Skips already processed images")
args = vars(ap.parse_args())

working_dir = os.getcwd()
skip_processed = args['skip_processed']

raws_path = args['raws']
raws_folder = os.path.join(working_dir, raws_path)
if not os.path.exists(raws_folder):
    raise Exception("Raws folder not found under %s" % raws_folder)

destination_path = args['destination']
destination_folder = os.path.join(working_dir, destination_path)
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

processed_images = []
log_file = os.path.join(destination_folder, VISITED_LOG_FILE_NAME)
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        processed_images = [line.rstrip('\n') for line in f]

for filename in sorted(os.listdir(raws_folder)):
    recto = RECTO_SUBSTRING in filename
    if not recto:
        continue
    name = re.sub(RECTO_SUBSTRING, '', filename)
    name = re.sub('_', '', name)

    if name in processed_images and skip_processed:
        continue

    current_folder = os.path.join(destination_folder, name)
    if not os.path.exists(current_folder):
        os.makedirs(current_folder)

    print("Begin processing image %s" % name)

    # RECTO PROCESSING
    image = raw_converter.to_cv2("samples/%s" % filename)
    page = document.crop_cardboard(image)
    image = None
    cv2.imwrite(os.path.join(current_folder, 'cardboard-re.png'), page)
    crop, text_section = cardboard.crop_image_and_text(page)
    cv2.imwrite(os.path.join(current_folder, 'image.png'), crop)
    cv2.imwrite(os.path.join(current_folder, 'text-section.png'), text_section)

    # VERSO PROCESSING
    verso_name = re.sub(RECTO_SUBSTRING, VERSO_SUBSTRING, filename)
    image = raw_converter.to_cv2("samples/%s" % verso_name)
    page = document.crop_cardboard(image)
    image = None
    cv2.imwrite(os.path.join(current_folder, 'cardboard-ve.png'), page)
    im_info = Info(barcode.detect(page))
    pretty_json = json.dumps(json.loads(jsonpickle.encode(im_info)), indent=4, sort_keys=True)

    with open(os.path.join(current_folder, 'info.json'), 'w') as f:
        f.write(pretty_json)

    if name not in processed_images:
        processed_images.append(name)

        with open(log_file, 'a') as f:
            f.write(name + '\n')

    print("End processing image %s" % name)
    print()
