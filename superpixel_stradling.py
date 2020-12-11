import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from skimage.filters import gaussian
from tqdm import tqdm

from typing import List, Set, Iterable, Tuple
from numpy.core.multiarray import ndarray

from utils.DisjointSet import DisjointSet
from utils.utils import get_n8


# TODO check performance
def __generate_weight_matrix(img: ndarray) -> lil_matrix:
    (rows, columns, _) = img.shape
    spare_weights = lil_matrix((rows * columns, rows * columns))
    print("lil_matrix.shape: " + str(spare_weights.shape))
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
                         + (img[row_idx, px_idx, 2] - img[row_2_idx, px_2_idx, 2]) ** 2.0
                spare_weights[linear, linear_2] = weight
                spare_weights[linear_2, linear] = weight
    print("__generate_weight_matrix.i: " + str(i))
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
           weights: csr_matrix) -> float:
    def tau(c: Set):
        k = (img_shape[0] + img_shape[1]) / 2000.0
        return k / len(c)

    int_1 = __int(component_1, img_shape, weights)
    int_2 = __int(component_2, img_shape, weights)

    return min(int_1 + tau(component_1), int_2 + tau(component_2))


def segmentate(img: ndarray) -> List[Set[Tuple[int, int]]]:
    img = gaussian(img, sigma=1.5)
    print("Generate weight for a " + str(img.shape))
    weights = __generate_weight_matrix(img)
    weights_csr = weights.tocsr()
    print("generated!")
    weights_coo: coo_matrix = weights.tocoo()
    print("rows: " + str(len(weights_coo.row)))
    edge_list: Iterable[(int, int, float)] = zip(weights_coo.row, weights_coo.col, weights_coo.data)
    edge_list_sorted: List[Tuple[int, int, float]] = sorted(edge_list, key=lambda triplet: triplet[2])
    S: DisjointSet = DisjointSet[Tuple[int, int]]()
    (rows, columns, _) = img.shape
    print("len(edge_sorted): " + str(len(edge_list_sorted)))
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
        mint = __mint(partition_1, partition_2, (rows, columns), weights_csr)
        if weight > mint:
            continue

        S.union((row_1_idx, px_1_idx), (row_2_idx, px_2_idx))

    return list(S.iter_sets())


def get_objectness(img: ndarray,
                   left: int, top: int, right: int, bottom: int,
                   theta_ms: float = 0.0,
                   learned: bool = False) -> float:
    S = segmentate(img)

    return 0.0
