import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import logging
import os.path
import glob

from ciniocr.shared import *
from ciniocr import DocumentInfo, RawScan


ap = argparse.ArgumentParser()
ap.add_argument("-r", "--raws", required=True,
                help="Folder with raw images")
ap.add_argument("-d", "--destination", required=True,
                help="Folder where the results will be saved")
ap.add_argument("-s", "--skip-processed", required=False, default=False, help="Skips already processed images")
ap.add_argument("-l", "--log-file", required=False, default='pipeline.log', help="Log file")
ap.add_argument("-w", "--nb-workers", required=False, default='1', help="Number of workers for parallelization")
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
processed_file = destination_folder / VISITED_LOG_FILE_NAME
if processed_file.exists():
    with processed_file.open() as f:
        processed_images = [line.rstrip('\n') for line in f]


##########################
# Looping over raw scans #
##########################
def process_one(file):
    file = Path(file)
    filename = file.name
    recto = RECTO_SUBSTRING in filename
    if not recto:
        return
    base_path = re.sub(RECTO_SUBSTRING, '', str(file))
    name = re.sub(RECTO_SUBSTRING, '', filename)
    # name = re.sub('_', '', base_path)

    if name in processed_images and skip_processed:
        return

    current_folder = destination_folder / name
    if not current_folder.exists():
        os.makedirs(str(current_folder))

    processed_successfully = True
    ####################
    # VERSO PROCESSING #
    ####################
    doc_info = DocumentInfo(base_path, side='verso')
    detected_barcode = ""
    try:
        verso_raw_scan = RawScan(doc_info, base_path)
        verso_raw_scan.crop_cardboard()
        detected_barcode = verso_raw_scan.get_cardboard().detect_barcode()
    except Exception as e:
        processed_successfully = False
        doc_info.logger.error(e)

    ####################
    # RECTO PROCESSING #
    ####################
    doc_info = DocumentInfo(base_path, side='recto')
    extracted_data = []
    try:
        recto_raw_scan = RawScan(doc_info, base_path, check_md5=True)
        recto_raw_scan.crop_cardboard()
        recto_raw_scan.save_cardboard(str(current_folder / 'cardboard.jpg'))

        cardboard = recto_raw_scan.get_cardboard()
        cardboard.extract_image_and_text()
        cardboard.save_image(str(current_folder / 'image.jpg'))

        txt_section = cardboard.get_text_section()
        txt_section.extract_text()
        txt_section.make_visualization(str(current_folder / 'extraction.jpg'))
        extracted_data = txt_section.get_extracted_data()
        doc_info.logger.debug("Done!")
    except Exception as e:
        processed_successfully = False
        doc_info.logger.error(e)

    data = {'barcode': detected_barcode,
            'segments': extracted_data}
    with open(str(current_folder / 'data.json'), 'w') as f:
        json.dump(data, f)


    ######################
    # Saving to log file #
    ######################
    if name not in processed_images and processed_successfully:
        processed_images.append(name)
        with processed_file.open('a') as f:
            f.write(name + '\n')

nb_workers = int(args['nb_workers'])

log_file = args['log_file']
if os.path.exists(log_file):
    raise IOError('Log file "{}" already exists'.format(log_file))

logger = logging.getLogger()
fhandler = logging.FileHandler(filename=log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)


with ProcessPoolExecutor(max_workers=nb_workers) as e:
    e.map(process_one, glob.glob('{}/**/*'.format(raws_folder)))
