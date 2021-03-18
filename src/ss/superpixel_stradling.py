import numpy as np
from skimage.transform import rescale
from skimage.segmentation import felzenszwalb
from tqdm import tqdm

from typing import List, Set, Tuple
from numpy.core.multiarray import ndarray

from ss.SuperpixelStradlingFoundation import SuperpixelStradlingFoundation



def __segmentate(img: ndarray, theta_ss: float, use_bilateral_filter: bool = False) -> List[Set[Tuple[int, int]]]:
    scale_k = (img.shape[0] + img.shape[1]) * theta_ss
    result_map = felzenszwalb(img, scale=scale_k, sigma=1.5, min_size=5)
    segmentation: List[Set[Tuple[int, int]]] = [set()] * (np.max(result_map) + 1)
    for idx in range(len(segmentation)):
        segmentation[idx] = set()

    for i in range(len(result_map)):
        for j in range(len(result_map[0])):
            segmentation[result_map[i, j]].add((i, j))

    return segmentation


def image_2_foundation(img: ndarray,
                       theta_ss: float = 1.0,
                       use_bilateral_filter: bool = False) -> SuperpixelStradlingFoundation:
    r, c, _ = img.shape
    factor = 1.0
    if r * c > (128 ** 2):
        factor = ((128.0 ** 2.0) / float(r * c)) ** 0.5
        img = rescale(img, (factor, factor, 1.0))
    return SuperpixelStradlingFoundation(__segmentate(img, theta_ss, use_bilateral_filter), factor)


def get_objectness(foundation: SuperpixelStradlingFoundation,
                   mask_coords: ndarray) -> float:
    segmentation = foundation.segmentation
    mask_coords_scaled: Set[Tuple[int, int]]
    mask_coords_scaled = set(map(lambda x: (x[0], x[1]), np.rint(mask_coords * foundation.scale).astype(int)))
    mask_coords_scaled_len = len(mask_coords_scaled)

    def calc_stradling(component: Set[Tuple[int, int]]):
        intersection_len = len(mask_coords_scaled.intersection(component))
        return min(intersection_len, len(component) - intersection_len)

    return 1.0 - (sum(list(map(calc_stradling, segmentation))) / mask_coords_scaled_len)
