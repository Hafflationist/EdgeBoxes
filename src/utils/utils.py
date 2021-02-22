import numpy as np
from typing import List, Tuple, Set
from numpy.core.multiarray import ndarray


def get_n8(matrix: ndarray, r_idx: int, p_idx: int) -> List[Tuple[int, int]]:
    all_possibilities = [(r_idx - 1, p_idx - 1),
                         (r_idx - 1, p_idx),
                         (r_idx - 1, p_idx + 1),
                         (r_idx, p_idx - 1),
                         (r_idx, p_idx),
                         (r_idx, p_idx + 1),
                         (r_idx + 1, p_idx - 1),
                         (r_idx + 1, p_idx),
                         (r_idx + 1, p_idx + 1)]
    px_idx_max = len(matrix[0])
    row_idx_max = len(matrix)
    result = list(filter(lambda x: 0 <= x[0] < row_idx_max and 0 <= x[1] < px_idx_max, all_possibilities))
    return result

def ndarraycoords_2_rect(coords: ndarray):
    coords = np.transpose(coords)
    row_coords = coords[0]
    col_coords = coords[1]
    if len(row_coords) == 0:
        return 0, 0, 0, 0
    return np.min(col_coords), np.min(row_coords), np.max(col_coords), np.max(row_coords)


def rect_2_coords(left: int, top: int, right: int, bottom: int) -> Set[Tuple[int, int]]:
    return {(tb, lr)
        for tb in range(top, bottom + 1)
        for lr in range(left, right + 1)
    }

def coords_2_filtercoords(coords: Set[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
    if len(coords) == 0:
        return [0], [0]
    coords_T = np.transpose(np.array(list(coords)))
    return coords_T[0], coords_T[1]


def coords_2_rect(coords: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    coords_T = np.transpose(list(coords))
    row_coords = coords_T[0]
    col_coords = coords_T[1]
    if len(row_coords) == 0:
        return 0, 0, 0, 0
    return np.min(col_coords), np.min(row_coords), np.max(col_coords), np.max(row_coords)
