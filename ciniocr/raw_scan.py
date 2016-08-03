from typing import Union
import os.path

import cv2
from skimage.transform import rotate
import numpy as np

from . import document_analysis
from . import shared
from . import utils
from .utils import md5
from .base import DocumentInfo
from .cardboard import RectoCardboard, VersoCardboard


class RawScan:
    def __init__(self, document_info: DocumentInfo, base_path: str, check_md5=False):
        """
        :param document_info:
        :param base_path: Base path of the files to be examined (ex : '/mnt/Cini/1A/1A_37')
        :param check_md5:
        :return:
        """
        self.document_info = document_info
        self.cropped_cardboard = None

        if self.document_info.side == 'recto':
            self.image_raw_path = base_path + shared.RECTO_SUBSTRING
        else:
            self.image_raw_path = base_path + shared.VERSO_SUBSTRING

        # Checks
        assert os.path.exists(self.image_raw_path)
        if check_md5:
            if self.document_info.side == 'recto':
                if not md5.check_md5(self.image_raw_path, base_path + shared.RECTO_MD5_SUBSTRING):
                    self.document_info.logger.error("Invalid md5")
                else:
                    self.document_info.logger.debug("Valid md5")
            else:
                if not md5.check_md5(self.image_raw_path, base_path + shared.VERSO_MD5_SUBSTRING):
                    self.document_info.logger.error("Invalid md5")
                else:
                    self.document_info.logger.debug("Valid md5")

        # Loads the image
        self.raw_scan = utils.load_raw_file_to_cv2(self.image_raw_path)
        if self.document_info.side == 'recto':
            self.output_filename = shared.RECTO_CARDBOARD_DEFAULT_FILENAME
        else:
            self.output_filename = shared.VERSO_CARDBOARD_DEFAULT_FILENAME

    def crop_cardboard(self):
        # Performs the crop
        self.cropped_cardboard = document_analysis.crop_biggest_rectangle(self.raw_scan)
        # Performs the checks
        h, w = self.cropped_cardboard.shape[:2]
        if h < w:
            self.cropped_cardboard = self.cropped_cardboard.transpose(1, 0, 2)[::-1]
            self.document_info.logger.info('Rotated the cardboard')
            h, w = self.cropped_cardboard.shape[:2]
        if not self._validate_height(h):
            self.document_info.logger.warning('Unusual cardboard height : {}'.format(h))
        if not self._validate_width(w):
            self.document_info.logger.warning('Unusual cardboard width : {}'.format(w))
        if not self._validate_ratio(h / w):
            self.document_info.logger.warning('Unusual cardboard ratio : {}'.format(h / w))

    def get_cardboard(self) -> Union['RectoCardboard', 'VersoCardboard']:
        assert self.cropped_cardboard is not None, 'Call crop_cardboard first'
        if self.document_info.side == 'recto':
            return RectoCardboard(self.document_info, self.cropped_cardboard)
        else:
            return VersoCardboard(self.document_info, self.cropped_cardboard)

    def save_cardboard(self, path=None):
        assert self.cropped_cardboard is not None, 'Call crop_cardboard first'
        if path is None:
            self.document_info.check_output_folder()
            cv2.imwrite(os.path.join(self.document_info.output_folder, self.output_filename), self.cropped_cardboard)
        else:
            cv2.imwrite(path, self.cropped_cardboard)

    @staticmethod
    def _validate_width(width):
        return shared.CARDBOARD_MIN_WIDTH <= width <= shared.CARDBOARD_MAX_WIDTH

    @staticmethod
    def _validate_height(height):
        return shared.CARDBOARD_MIN_HEIGHT <= height <= shared.CARDBOARD_MAX_HEIGHT

    @staticmethod
    def _validate_ratio(ratio):
        return shared.CARDBOARD_MIN_RATIO <= ratio <= shared.CARDBOARD_MAX_RATIO