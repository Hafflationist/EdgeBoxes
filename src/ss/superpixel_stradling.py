from itertools import dropwhile

import numpy as np
import cv2
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage.filters import gaussian
from skimage.transform import rescale
from tqdm import tqdm

from typing import List, Set, Iterable, Tuple
from numpy.core.multiarray import ndarray

from src.ss.DisjointSet import DisjointSet
from src.ss.SuperpixelStradlingFoundation import SuperpixelStradlingFoundation
from src.utils.utils import get_n8


def __generate_weight_matrix(img: ndarray) -> lil_matrix:
    (rows, columns, _) = img.shape
    spare_weights = lil_matrix((rows * columns, rows * columns))
    i = 0
    for row_idx in range(rows):
        for px_idx in range(columns):
            for (row_2_idx, px_2_idx) in get_n8(img, row_idx, px_idx):
                linear = row_idx * columns + px_idx
                linear_2 = row_2_idx * columns + px_2_idx
                i += 1
                if spare_weights[linear, linear_2] > 0.0:
                    continue
                weight = (img[row_idx, px_idx, 0] - img[row_2_idx, px_2_idx, 0]) ** 2.0 \
                         + (img[row_idx, px_idx, 1] - img[row_2_idx, px_2_idx, 1]) ** 2.0 \
                         + (img[row_idx, px_idx, 2] - img[row_2_idx, px_2_idx, 2]) ** 2.0 \
                         + 0.01
                spare_weights[linear, linear_2] = weight
                spare_weights[linear_2, linear] = weight
    return spare_weights


def __int(component: Set[Tuple[int, int]], img_shape: (int, int), weights: csr_matrix) -> float:
    (rows, columns) = img_shape
    weight_coords = np.array([pair[0] * columns + pair[1] for pair in component])
    sub_weights_matrix = weights[weight_coords, :][:, weight_coords]
    mst = minimum_spanning_tree(csgraph=sub_weights_matrix)
    result = csr_matrix.max(mst)
    return result


def __mint(component_1: Set[Tuple[int, int]],
           component_2: Set[Tuple[int, int]],
           img_shape: (int, int),
           weights: csr_matrix,
           theta_ss: float) -> float:
    def tau(c: Set):
        k = ((img_shape[0] + img_shape[1]) / 2000.0) * theta_ss
        return k / len(c)

    int_1 = __int(component_1, img_shape, weights)
    int_2 = __int(component_2, img_shape, weights)

    return min(int_1 + tau(component_1), int_2 + tau(component_2))


def __segmentate(img: ndarray, theta_ss: float, use_bilateral_filter: bool = False) -> List[Set[Tuple[int, int]]]:
    if use_bilateral_filter:
        img = cv2.bilateralFilter(np.uint8((img / np.max(img)) * 255), 9, 75, 75)
        img = img / 255
        img = gaussian(img, sigma=1.0)
    else:
        img = gaussian(img, sigma=1.5)
    # img = gaussian(img, sigma=1.5)
    weights = __generate_weight_matrix(img)
    weights_csr = weights.tocsr()
    weights_coo: coo_matrix = weights.tocoo()
    edge_list: Iterable[(int, int, float)] = zip(weights_coo.row, weights_coo.col, weights_coo.data)
    edge_list_sorted: List[Tuple[int, int, float]] = sorted(edge_list, key=lambda triplet: triplet[2])
    edge_list_sorted = list(dropwhile(lambda x: x[2] == 0, edge_list_sorted))
    S: DisjointSet = DisjointSet[Tuple[int, int]]()
    (rows, columns, _) = img.shape
    for i in tqdm(range(len(edge_list_sorted))):
        (linear_1, linear_2, weight) = edge_list_sorted[i]
        if linear_1 == linear_2:
            continue

        row_1_idx = linear_1 // columns
        px_1_idx = linear_1 % columns

        row_2_idx = linear_2 // columns
        px_2_idx = linear_2 % columns
        if S.connected((row_1_idx, px_1_idx), (row_2_idx, px_2_idx)):
            continue

        partition_1 = set(S.iter_specific_set((row_1_idx, px_1_idx)))
        partition_2 = set(S.iter_specific_set((row_2_idx, px_2_idx)))
        mint = __mint(partition_1, partition_2, (rows, columns), weights_csr, theta_ss)
        if weight > mint:
            continue

        S.union((row_1_idx, px_1_idx), (row_2_idx, px_2_idx))

    return list(S.iter_sets())


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
