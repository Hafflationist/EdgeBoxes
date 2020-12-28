from typing import List, Set, Tuple

import numpy as np
import colorsys
import random

from numpy.core.multiarray import ndarray


def find_idx_of_element(partitioning: List[Set[Tuple[int, int]]], element: Tuple[int, int]):
    for i in range(len(partitioning)):
        if element in partitioning[i]:
            return i
    return -1


def color_segmentation(img: ndarray, S: List[Set[Tuple[int, int]]]) -> ndarray:
    def calculate_pixel(row_idx, px_idx):
        hue: float = partitions_color[find_idx_of_element(S, (row_idx, px_idx))]
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return [rgb[0], rgb[1], rgb[2], 1.0]

    partitions_color = [random.random() for _ in S]
    segmentation_colored = [[calculate_pixel(row_idx, px_idx)
                             for px_idx in range(len(img[0]))]
                            for row_idx in range(len(img))]
    return np.array(segmentation_colored) / np.max(segmentation_colored)
