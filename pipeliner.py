import argparse
import json
import jsonpickle
from pathlib import Path
import re

import cv2

import barcode
import cardboard
import document
import raw_converter
import extractor
from info import Info
from shared import *

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--raws", required=True,
                help="Folder with raw images")
ap.add_argument("-d", "--destination", required=True,
                help="Folder where the results will be saved")
ap.add_argument("-s", "--skip-processed", required=False, default=False, help="Skips already processed images")
args = vars(ap.parse_args())

skip_processed = args['skip_processed']

##################################################
# Getting raws folder and checking for existence #
##################################################
raws_path = args['raws']
raws_folder = Path(raws_path)
if not raws_folder.exists():
    raise Exception("Raws folder not found under %s" % raws_folder)
raws_folder = raws_folder.resolve()

#############################################
# Getting destination folder or creating it #
#############################################
destination_path = args['destination']
destination_folder = Path(destination_path)
if not destination_folder.exists():
    destination_folder.mkdir()
destination_folder = destination_folder.resolve()

####################
# Reading Log File #
####################
processed_images = []
log_file = destination_folder / VISITED_LOG_FILE_NAME
if log_file.exists():
    with log_file.open() as f:
        processed_images = [line.rstrip('\n') for line in f]

##########################
# Looping over raw scans #
##########################
for file in sorted([x for x in raws_folder.iterdir()]):
    filename = file.name
    recto = RECTO_SUBSTRING in filename
    if not recto:
        continue
    name = re.sub(RECTO_SUBSTRING, '', filename)
    name = re.sub('_', '', name)

    if name in processed_images and skip_processed:
        continue

    current_folder = destination_folder / name
    if not current_folder.exists():
        current_folder.mkdir()

    print("Begin processing image %s" % name)

    ####################
    # RECTO PROCESSING #
    ####################
    image = raw_converter.to_cv2(str(file))
    page = document.crop_cardboard(image)
    cv2.imwrite(str(current_folder / 'cardboard-re.png'), page)
    crop, text_section = cardboard.crop_image_and_text(page)
    cv2.imwrite(str(current_folder / 'image.png'), crop)
    cv2.imwrite(str(current_folder / 'text-section.png'), text_section)
    text_bounds = extractor.text_bounds(text_section)

    ####################
    # VERSO PROCESSING #
    ####################
    verso_name = re.sub(RECTO_SUBSTRING, VERSO_SUBSTRING, str(file))
    image = raw_converter.to_cv2(verso_name)
    page = document.crop_cardboard(image)
    cv2.imwrite(str(current_folder / 'cardboard-ve.png'), page)
    im_info = Info(barcode.detect(page), text_bounds)
    pretty_json = json.dumps(json.loads(jsonpickle.encode(im_info, unpicklable=False)), indent=4, sort_keys=True)
    with (current_folder / ("%s-fronte.json" % im_info.barcode)).open('w') as f:
        f.write(pretty_json)

    ######################
    # Saving to log file #
    ######################
    if name not in processed_images:
        processed_images.append(name)

        with log_file.open('a') as f:
            f.write(name + '\n')

    print("End processing image %s" % name)
    print()
