import unittest
from crop_and_skew import *
import re
import os
import cv2
import barcode
import json

VERSO_SUBSTR = '_verso.cr2'
SAMPLES_DIR = '../samples'

versos = filter(lambda f: VERSO_SUBSTR in f, os.listdir(SAMPLES_DIR))


class DecoderTest(unittest.TestCase):
    def test(self):
        with open("test-results/barcode-results.json") as results:
            expected = json.loads(results.read())

        for filename in sorted(versos):
            name = re.sub(VERSO_SUBSTR, '', filename)
            name = re.sub('_', '', name)
            im = cv2.imread("%s/%s" % (SAMPLES_DIR, filename))
            crop = crop_and_skew(im)
            bar_code = barcode.detect(crop)
            self.assertEqual(bar_code, expected[name])
