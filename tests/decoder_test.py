import json
import os
import re
import unittest

import cv2

from ciniocr.utils import barcode


class DecoderTest(unittest.TestCase):
    def test(self):
        with open("test-results/barcode-results.json") as results:
            expected = json.loads(results.read())

        for filename in sorted(os.listdir("../results/cardboards")):
            if 're' in filename:
                continue
            name = re.sub('ve.png', '', filename)
            im = cv2.imread("../results/cardboards/%s" % filename)
            bar_code = barcode.detect_and_read(im)
            self.assertEqual(bar_code, expected[name],
                             "Failed at %s, found %s, expected %s" % (name, bar_code, expected[name]))
