from rawkit.raw import Raw
import numpy as np
import cv2


def to_cv2(filename):
    with Raw(filename=filename) as raw:
        raw.options.rotation = 0
        raw.options.brightness = 2.5
        raw.options.auto_brightness = False

        w = raw.data.contents.sizes.width
        h = raw.data.contents.sizes.height
        na = np.frombuffer(raw.to_buffer(), np.int8)
        na = na.reshape(h, w, 3).astype("uint8")
        return cv2.cvtColor(na, cv2.COLOR_BGR2RGB)
