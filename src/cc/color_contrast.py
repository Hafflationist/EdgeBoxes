import math
import numpy as np
from numpy.core.multiarray import ndarray
from scipy.ndimage import binary_dilation
from skimage import color
from typing import Tuple, Set, List

from cc.ColorContrastFoundation import ColorContrastFoundation
from utils.utils import rect_2_coords, coords_2_filtercoords


def __histogram_of_windows(img: ndarray, filtercoords: Tuple[List[int], List[int]]) -> Tuple[ndarray, ndarray]:
    img_filtered = img[filtercoords]
    hist, bins = np.histogram(img_filtered, bins=16, range=(np.min(img), np.max(img)))
    hist = hist / np.sum(hist)
    if np.isnan(np.sum(hist)):
        print("filtercoords:{}".format(filtercoords))
    return hist, bins


def __chi_square_distance(hist_1: ndarray, hist_2: ndarray) -> float:
    def addend(pair):
        divisor: float = (pair[0] + pair[1])
        if divisor == 0.0:
            divisor = 0.0000001
        return ((pair[0] - pair[1]) ** 2) / divisor

    assert len(hist_1) == len(hist_2)
    result = sum(list(map(addend, zip(hist_1, hist_2))))
    return 0.0 if math.isnan(result) else result


def image_2_foundation(img: ndarray) -> ColorContrastFoundation:
    return ColorContrastFoundation(color.rgb2lab(img))


def get_objectness(foundation: ColorContrastFoundation,
                   mask_coords: Set[Tuple[int, int]],
                   theta_cc: float = 2.0) -> float:
    core_filtercoords = coords_2_filtercoords(mask_coords)

    # surrounding
    binary_array = np.zeros(foundation.img_lab.shape)
    binary_array[core_filtercoords] = 1
    dilation_iterations = int(((foundation.img_lab.shape[0] + foundation.img_lab.shape[1]) / 2.0) * theta_cc)
    surr_filtercoords: Tuple[List[int], List[int]] = \
        np.where(binary_dilation(input=binary_array, iterations=dilation_iterations))

    # CIE-LAB
    img_l: ndarray = np.array([[px[0] for px in row] for row in foundation.img_lab])
    img_a: ndarray = np.array([[px[1] for px in row] for row in foundation.img_lab])
    img_b: ndarray = np.array([[px[2] for px in row] for row in foundation.img_lab])

    # histograms
    img_l_hist = __histogram_of_windows(img_l, core_filtercoords)
    img_l_surr_hist = __histogram_of_windows(img_l, surr_filtercoords)
    img_a_hist = __histogram_of_windows(img_a, core_filtercoords)
    img_a_surr_hist = __histogram_of_windows(img_a, surr_filtercoords)
    img_b_hist = __histogram_of_windows(img_b, core_filtercoords)
    img_b_surr_hist = __histogram_of_windows(img_b, surr_filtercoords)

    # chi² distances
    chi_l: float = __chi_square_distance(img_l_hist[0], img_l_surr_hist[0])
    chi_a: float = __chi_square_distance(img_a_hist[0], img_a_surr_hist[0])
    chi_b: float = __chi_square_distance(img_b_hist[0], img_b_surr_hist[0])
    return chi_a + chi_b + chi_l
