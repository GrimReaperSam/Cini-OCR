import os.path

import cv2
import numpy as np

from . import document_analysis
from . import shared
from . import utils
from .utils import barcode
from .base import DocumentInfo
from .text_section import TextSection


class RectoCardboard:
    def __init__(self, document_info: DocumentInfo, document=None):
        self.document_info = document_info
        if document is not None:
            self.cardboard = document
        else:
            self.cardboard = cv2.imread(os.path.join(self.document_info.output_folder,
                                                     shared.RECTO_CARDBOARD_DEFAULT_FILENAME))
        self._image = None
        self._image_bounds = None
        self._text_section = None

    def extract_image_and_text(self):
        doc = self.cardboard
        ratio = doc.shape[0] / shared.RESIZE_HEIGHT
        doc = cv2.resize(doc, (int(doc.shape[1] / ratio), int(shared.RESIZE_HEIGHT)))

        gray = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)
        gray_filtered = cv2.medianBlur(gray, 9)

        opened = cv2.morphologyEx(gray_filtered, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
        mask = np.ones(opened.shape, dtype=np.uint8)
        border_margin = int(gray_filtered.shape[1]*0.05)
        mask[:border_margin, :], mask[-border_margin:, :] = 0, 0
        mask[:, border_margin], mask[:, -border_margin:] = 0, 0

        image_bounds_resized = document_analysis.find_convex_hull_rectangle(opened, mask)
        # image_bounds_resized = document_analysis.find_biggest_rectangle(gray_filtered,
        #                                                                int(0.1*shared.RESIZE_HEIGHT),
        #                                                                mask=mask)

        self._image = utils.crop_rectangle_warp(self.cardboard, image_bounds_resized, ratio)
        self._image_bounds = (image_bounds_resized*ratio).astype(np.int)

        mask = np.ones(gray.shape, dtype=np.uint8)
        mask = cv2.polylines(mask, image_bounds_resized[np.newaxis], True, 0, 10)
        mask = cv2.fillConvexPoly(mask, image_bounds_resized[np.newaxis], 0)
        mask[mask.shape[0]//2:] = 0

        lowest_line = document_analysis.find_lowest_horizontal_line(gray, mask)
        lowest_line = int(lowest_line*ratio)
        self._text_section = self.cardboard[:lowest_line]

        # Checks
        text_section_height = self._text_section.shape[0]
        image_center_x = self._image_bounds[:, 0].sum()/4
        if not self._validate_image_section(image_center_x, self.cardboard.shape[1]):
            self.document_info.logger.warning("The image crop isn't horizontally centered "
                                              "(img_x:{}, page_width:{})".format(image_center_x,
                                                                                 self.cardboard.shape[1]))
        if not self._validate_text_section(lowest_line):
            self.document_info.logger.warning("The height of this text section is not usual : {}".format(text_section_height))

    def save_image(self, path=None):
        assert self._image is not None
        if path is None:
            self.document_info.check_output_folder()
            cv2.imwrite(os.path.join(self.document_info.output_folder, shared.IMAGE_DEFAULT_FILENAME), self._image)
        else:
            cv2.imwrite(path, self._image)

    def save_text_section(self, path=None):
        assert self._text_section is not None
        if path is None:
            self.document_info.check_output_folder()
            cv2.imwrite(os.path.join(self.document_info.output_folder, shared.TEXT_SECTION_DEFAULT_FILENAME),
                        self._text_section)
        else:
            cv2.imwrite(path, self._text_section)

    def get_text_section(self) -> TextSection:
        return TextSection(self.document_info, self._text_section)

    @staticmethod
    def _validate_image_section(x, page_width):
        page_mid_x = page_width / 2
        acceptable_x_min = page_mid_x - 0.06 * page_width
        acceptable_x_max = page_mid_x + 0.06 * page_width
        return acceptable_x_min <= x <= acceptable_x_max

    @staticmethod
    def _validate_text_section(y_value):
        valid = False
        for (mini, maxi) in shared.ACCEPTABLE_TEXT_SECTIONS_Y_RANGES:
            if mini <= y_value <= maxi:
                valid = True
                break
        return valid


class VersoCardboard:
    def __init__(self, document_info: DocumentInfo, image=None):
        self.document_info = document_info
        if image is not None:
            self.cardboard = image
        else:
            self.cardboard = cv2.imread(os.path.join(self.document_info.output_folder,
                                                     shared.VERSO_CARDBOARD_DEFAULT_FILENAME))
        self.barcode = None

    def detect_barcode(self) -> str:
        self.barcode = barcode.detect_and_read(self.cardboard)
        if self.barcode == "":
            self.document_info.logger.error("No barcode detected")
        if not self.document_info.validate_barcode(self.barcode):
            self.document_info.logger.error("Barcode mismatch, detected barcode : '{}'".format(self.barcode))
        else:
            self.document_info.logger.debug("Correct barcode : '{}'".format(self.barcode))
        return self.barcode