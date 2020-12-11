import numpy as np
from numpy.core.multiarray import ndarray
from skimage import color
from typing import Tuple

from src.cc.ColorContrastFoundation import ColorContrastFoundation


def __histogram_of_windows(img: ndarray, left: int, top: int, right: int, bottom: int) -> Tuple[ndarray, ndarray]:
    hist, bins = np.histogram(img[top:bottom, left:right], bins=16, range=(np.min(img), np.max(img)))
    hist = hist / np.sum(hist)
    return hist, bins


def __chi_square_distance(hist_1: ndarray, hist_2: ndarray):
    def addend(pair):
        divisor = (pair[0] + pair[1])
        if divisor == 0.0:
            divisor = 0.0000001
        return ((pair[0] - pair[1]) ** 2) / divisor

    assert len(hist_1) == len(hist_2)
    return np.sum(list(map(addend, zip(hist_1, hist_2))))


def image_2_foundation(img: ndarray) -> ColorContrastFoundation:
    return ColorContrastFoundation(color.rgb2lab(img))


# TODO: use mask instead of window (how could SURR be calculated?)
def get_objectness(foundation: ColorContrastFoundation,
                   left: int, top: int, right: int, bottom: int,
                   theta_cc: float = 2.0) -> float:

    width = right - left
    height = bottom - top

    half_delta_width: int = int(width * theta_cc - width) // 2
    half_delta_height: int = int(height * theta_cc - height) // 2

    left_surr = max(left - half_delta_width, 0)
    top_surr = max(top - half_delta_height, 0)
    right_surr = min(right + half_delta_width, len(foundation.img_lab[0]) - 1)
    bottom_surr = max(bottom + half_delta_height, 0)

    img_l = np.array([[px[0] for px in row] for row in foundation.img_lab])
    img_a = np.array([[px[1] for px in row] for row in foundation.img_lab])
    img_b = np.array([[px[2] for px in row] for row in foundation.img_lab])

    img_l_hist = np.array(__histogram_of_windows(img_l, left, top, right, bottom))
    img_l_surr_hist = __histogram_of_windows(img_l, left_surr, top_surr, right_surr, bottom_surr)
    img_a_hist = __histogram_of_windows(img_a, left, top, right, bottom)
    img_a_surr_hist = __histogram_of_windows(img_a, left_surr, top_surr, right_surr, bottom_surr)
    img_b_hist = __histogram_of_windows(img_b, left, top, right, bottom)
    img_b_surr_hist = __histogram_of_windows(img_b, left_surr, top_surr, right_surr, bottom_surr)

    chi_l = __chi_square_distance(img_l_hist[0], img_l_surr_hist[0])
    chi_a = __chi_square_distance(img_a_hist[0], img_a_surr_hist[0])
    chi_b = __chi_square_distance(img_b_hist[0], img_b_surr_hist[0])
    return chi_a + chi_b + chi_l