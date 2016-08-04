import cv2
import numpy as np
import matplotlib.pyplot as plt

from .frangi_filter import frangi_filter_2d


def find_lowest_horizontal_line(gray_image, mask=None, canny_threshold=50):
    gray_filtered = cv2.fastNlMeansDenoising(gray_image)
    edge_map = cv2.Canny(gray_filtered, canny_threshold, 3 * canny_threshold, apertureSize=3)

    if mask is not None:
        edge_map = edge_map*mask
    # plt.imshow(mask)
    lines = cv2.HoughLines(edge_map, 1, (np.pi / 180)/2,
                           threshold=int(gray_image.shape[1]/4))

    horizontal_lines = []
    for l in lines[:, 0, :]:
        if np.pi/2 - np.pi / 12 < l[1] < np.pi/2 + np.pi / 12:
            horizontal_lines.append(l)

    def get_y(line):
        return abs(line[0])

    # print(sorted([get_y(h) for h in horizontal_lines]))
    lowest_y = get_y(sorted(horizontal_lines, key=get_y, reverse=True)[0])
    return lowest_y


def get_line_mask(image, num_iterations=6, vertical=False):
    kernel = np.ones((1, 5), np.uint8)
    kernel_2 = np.ones((3, 1), np.uint8)
    kernel_3 = np.ones((1, 5), np.uint8)
    if vertical:
        kernel, kernel_2, kernel_3 = kernel.transpose(), kernel_2.transpose(), kernel_3.transpose()
    tmp = image.copy()
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_CROSS, kernel_3)
    for i in range(num_iterations):
        tmp = cv2.erode(tmp, kernel, iterations=5)
        tmp = cv2.dilate(tmp, kernel_2)
    return cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def close_lines(image, kernel_size=150):
    image[-10:, :] = 255
    image[:10, :] = 255
    image[:, :10] = 255
    image[:, -10:] = 255

    kernel_size_1 = kernel_size
    kernel_size_2 = kernel_size
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size_1,kernel_size_1))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_2,kernel_size_2))

    return cv2.erode(cv2.dilate(image, kernel_1), kernel_2, borderType=cv2.BORDER_REPLICATE)


def detect_riddes(gray_image, resized_height=500, ridge_scale=1.5, threshold=0.3):
    ratio = gray_image.shape[0] / resized_height
    img = cv2.resize(gray_image, (int(gray_image.shape[1] / ratio), resized_height))
    filter_response = frangi_filter_2d(img, FrangiScaleRange=np.array([ridge_scale]))
    filter_response = cv2.resize(filter_response, (gray_image.shape[1], gray_image.shape[0]))
    return (filter_response > threshold).astype(np.uint8)
