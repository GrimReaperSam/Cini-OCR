from typing import Union, List, Dict

import numpy as np
import cv2
import pytesseract
from PIL import Image

from .base import DocumentInfo
from .document_analysis.line import detect_riddes, get_line_mask, close_lines
from .utils import get_enclosing_contours


class TextFragment:
    def __init__(self, area_index=None, bb=None, txt=""):
        self.box = bb
        self.text = txt
        self.area_index = area_index

    def to_dict(self):
        x, y, w, h = self.box
        return {
            "h": h,
            "w": w,
            "x": x,
            "y": y,
            "transcription": self.text,
            "area_index": self.area_index
        }

    def __repr__(self):
        return "TextFragment(txt={},box={},area_index={})".format(self.text, self.box, self.area_index)


class TextSection:
    def __init__(self, document_info: DocumentInfo, image: np.ndarray):
        self.document_info = document_info
        self.text_section = image
        self._area_contours = None
        self._extracted_data = None  # type: List[TextFragment]

    def extract_text(self):
        # Convert to gray
        binarized = cv2.cvtColor(self.text_section, cv2.COLOR_BGR2GRAY)

        # Filter the noise
        binarized = cv2.fastNlMeansDenoising(binarized, h=8, searchWindowSize=50)
        # binarized = cv2.medianBlur(binarized, 11)
        # binarized = cv2.adaptiveThreshold(binarized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 4)

        # Detect the black lines (actually detect well text too)
        binarized = detect_riddes(binarized)
        # Clean up to only keep the vertical and horizontal lines
        mask = get_line_mask(binarized) + get_line_mask(binarized, num_iterations=3, vertical=True)
        # Close the boundaries
        header_mask = close_lines(mask)

        # Get the area contours
        contours = get_enclosing_contours(255-header_mask)
        if len(contours) != 8:
            self.document_info.logger.warning('Number of area contours unusual : {}'.format(len(contours)))
        self._area_contours = contours

        # Get the text contours
        self._extracted_data = []
        for i in range(len(contours)):
            tmp_cnt_mask = cv2.drawContours(np.zeros_like(binarized), contours, i, 1, cv2.FILLED)
            txt_mask = binarized*tmp_cnt_mask
            txt_mask = cv2.morphologyEx(txt_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            txt_mask = cv2.morphologyEx(txt_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80)))
            inside_contours = get_enclosing_contours(txt_mask)
            for cnt in inside_contours:
                if cv2.contourArea(cnt) > 600:
                    bb = cv2.boundingRect(cnt)
                    _, _, w, h = bb
                    if h > 25 and w > 25:
                        self._extracted_data.append(TextFragment(i, bb))
        self.document_info.logger.debug('Number of text boxes : {}'.format(len(self._extracted_data)))

        # Recognize text
        margin = 40
        for d in self._extracted_data:
            x, y, w, h = d.box
            txt_img = self.text_section[
                      max(y-margin, 0):min(y+h+margin, self.text_section.shape[0]),
                      max(x-margin, 0):min(x+w+margin, self.text_section.shape[1])]
            d.text = pytesseract.image_to_string(Image.fromarray(txt_img))

    def make_visualization(self, save_path=None) -> np.ndarray:
        assert self._extracted_data is not None, 'Run extract_text first'
        drawing = self.text_section.copy()
        # Draw areas
        drawing = cv2.drawContours(drawing, self._area_contours, -1, (255, 0, 0), 3)
        # Draw text boxes
        for d in self._extracted_data:
            x, y, w, h = d.box
            cv2.rectangle(drawing, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)

        if save_path is not None:
            cv2.imwrite(save_path, drawing)
        return drawing

    def get_extracted_data(self) -> List[Dict]:
        assert self._extracted_data is not None, 'Run extract_text first'
        data = []
        for d in self._extracted_data:
            data.append(d.to_dict())
        return data

    def get_extraced_text(self) -> List[TextFragment]:
        assert self._extracted_data is not None, 'Run extract_text first'
        return self._extracted_data
