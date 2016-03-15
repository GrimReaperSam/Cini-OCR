import unittest
import re
import os
import cv2
import barcode
import json


class DecoderTest(unittest.TestCase):
    def test(self):
        with open("test-results/barcode-results.json") as results:
            expected = json.loads(results.read())

        for filename in sorted(os.listdir("../results/pages")):
            name = re.sub('ve.png', '', filename)
            im = cv2.imread("../results/pages/%s" % filename)
            bar_code = barcode.detect(im)
            self.assertEqual(bar_code, expected[name])