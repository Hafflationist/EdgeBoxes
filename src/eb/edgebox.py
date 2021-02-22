from typing import List, Tuple

import cv2
import numpy as np
import math

from functools import reduce
from numpy.core.multiarray import ndarray

from eb.EdgeboxFoundation import EdgeboxFoundation
from utils.utils import get_n8


def detect_edges(img: ndarray) -> Tuple[ndarray, ndarray]:
    img_processed = (img / np.max(img)).astype(np.float32)
    modelFilename = "model/model.yml.gz"
    pDollar = cv2.ximgproc.createStructuredEdgeDetection(modelFilename)
    edges = pDollar.detectEdges(cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))
    orientation_map = pDollar.computeOrientation(edges)
    edges_nms = pDollar.edgesNms(edges, orientation_map)
    return edges_nms, orientation_map


# each pixel consists of:
# 1. edge magnitude (0.0 to 1.0)
# 2. group id
# returns list<list<[int, int]>>     (Matrix of [edge, group id])
#       * list<list<int*int>>        (List of all group members(coords))
def group_edges(edges_nms_orig: ndarray, orientation_map: ndarray) -> Tuple[ndarray, ndarray]:

    def get_new_todo(matrix: ndarray) -> Tuple[int, int]:
        todo = [coord for coord in coords_of_edges if matrix[coord[0], coord[1], 1] == -1]
        if len(todo) == 0:
            return -1, -1
        return todo[0]

    def get_next_todo(matrix: ndarray, curr_r_idx: int, curr_p_idx: int) -> Tuple[int, int]:
        root_coord = groups_members[edges_with_grouping[curr_r_idx][curr_p_idx][1]][0]
        for (ro, pi) in sorted(get_n8(matrix, curr_r_idx, curr_p_idx),
                               key=lambda coord: ((coord[0] - root_coord[0])**2 + (coord[1] - root_coord[1])**2)):
            if edges_with_grouping[ro][pi][0] != 1 or edges_with_grouping[ro][pi][1] != -1:
                continue
            return ro, pi
        return get_new_todo(matrix)

    edges_nms = edges_nms_orig
    edges_nms[edges_nms < 0.1] = 0      # thresholding
    edges_nms[edges_nms >= 0.1] = 1.0   # thresholding
    edges_nms = np.uint8(edges_nms)
    new_group_id: int = 0
    groups_diff_cum: List[float] = []
    groups_members: List[List[List[int]]] = []
    edges_with_grouping = np.array([[[edges_nms[row_idx, px_idx], -1]
                                     for px_idx in range(len(edges_nms[0]))]
                                    for row_idx in range(len(edges_nms))])
    coords_of_edges = [(row_idx, px_idx)
                       for px_idx in range(len(edges_with_grouping[0]))
                       for row_idx in range(len(edges_with_grouping))
                       if edges_with_grouping[row_idx, px_idx, 0] == 1]
    half_pi = math.pi / 2.0

    (row_idx, px_idx) = get_new_todo(edges_with_grouping)
    while True:
        if row_idx == -1 or px_idx == -1:
            break

        new_group_id_candidate: int = new_group_id
        # check N8 neighborhood
        px_orientation = orientation_map[row_idx, px_idx]
        for (r, p) in get_n8(edges_nms, row_idx, px_idx):
            if edges_nms[r, p] != 1 \
                    or edges_with_grouping[r][p][1] == -1 \
                    or groups_diff_cum[edges_with_grouping[r][p][1]] > (half_pi * 2.0):     # TODO Hier sollte man nicht verdoppeln
                continue
            current_diff: float = abs(px_orientation - orientation_map[r, p])
            current_diff = min(math.pi - current_diff, current_diff)  # difference in a circle
            new_group_id_candidate = edges_with_grouping[r][p][1]
            # update group information...
            groups_members[new_group_id_candidate].append([row_idx, px_idx])
            groups_diff_cum[new_group_id_candidate] += current_diff
            break
        else:
            # new group created:
            groups_diff_cum.append(0.0)
            groups_members.append([[row_idx, px_idx]])
            new_group_id += 1

        edges_with_grouping[row_idx][px_idx] = [edges_nms[row_idx, px_idx], new_group_id_candidate]
        edges_with_grouping[row_idx][px_idx][0] = edges_nms[row_idx, px_idx]
        edges_with_grouping[row_idx][px_idx][1] = new_group_id_candidate
        (row_idx, px_idx) = get_next_todo(edges_with_grouping, row_idx, px_idx)

    return edges_with_grouping, np.array(groups_members)


# returns list<list<float>> (Adjazenzmatrix)
def calculate_affinities(groups_members: ndarray, orientation_map: ndarray):
    def mean_of_coords(idx: int) -> ndarray:
        rows = [coord[0] for coord in groups_members[idx]]
        columns = [coord[1] for coord in groups_members[idx]]
        # Falls Zeit übrig: Investigieren, weshalb hier np.array steht, statt einfach ein Paar zurück zu geben
        return np.array([sum(rows) / len(rows), sum(columns) / len(columns)])

    def mean_of_orientations(idx: int) -> float:
        orientations = [orientation_map[(coord[0], coord[1])] for coord in groups_members[idx]]
        return sum(orientations) / len(orientations)

    groups_mean_position = [mean_of_coords(idx) for idx in range(len(groups_members))]
    groups_mean_orientation = [mean_of_orientations(idx) for idx in range(len(groups_members))]
    groups_min_row_idx = [np.min([g[0] for g in groups_members[idx]]) for idx in range(len(groups_members))]
    groups_max_row_idx = [np.max([g[0] for g in groups_members[idx]]) for idx in range(len(groups_members))]
    groups_min_col_idx = [np.min([g[1] for g in groups_members[idx]]) for idx in range(len(groups_members))]
    groups_max_col_idx = [np.max([g[1] for g in groups_members[idx]]) for idx in range(len(groups_members))]

    def calc_angle_between_points(coord_1: (int, int), coord_2: (int, int)) -> float:
        coord_diff = list(map(lambda a, b: a - b, coord_1, coord_2))
        if coord_diff[1] == 0.0:
            coord_diff[1] = 0.0001
        return (np.arctan(coord_diff[0]/coord_diff[1]) + (math.pi / 2.0)) / math.pi

    def calc_distance(group_id_1: int, group_id_2: int) -> float:
        distance = 10.0
        if(groups_min_row_idx[group_id_1] - groups_max_row_idx[group_id_2] > distance
                or groups_min_row_idx[group_id_2] - groups_max_row_idx[group_id_1] > distance
                or groups_min_col_idx[group_id_1] - groups_max_col_idx[group_id_2] > distance
                or groups_min_col_idx[group_id_2] - groups_max_col_idx[group_id_1] > distance):
            return math.inf
        mean_1 = groups_mean_position[group_id_1]
        mean_2 = groups_mean_position[group_id_2]
        c_with_d_1 = [(r, p, (r - mean_2[0])**2 + (p - mean_2[1])**2) for (r, p) in groups_members[group_id_1]]
        c_with_d_2 = [(r, p, (r - mean_1[0])**2 + (p - mean_1[1])**2) for (r, p) in groups_members[group_id_2]]
        nearest_1: Tuple[int, int, float] = sorted(c_with_d_1, key=lambda triple: triple[2])[0]
        nearest_2: Tuple[int, int, float] = sorted(c_with_d_2, key=lambda triple: triple[2])[0]
        return (nearest_1[0] - nearest_2[0])**2 + (nearest_1[1] - nearest_2[1])**2

    def calculate_affinity(group_id_1: int, group_id_2: int) -> float:
        if group_id_column == group_id_row:
            return 1.0
        max_distance = 8
        if calc_distance(group_id_1, group_id_2) > max_distance:
            return 0.0
        pos_1 = groups_mean_position[group_id_1]
        pos_2 = groups_mean_position[group_id_2]
        theta_12: float = calc_angle_between_points((pos_1[0], pos_1[1]), (pos_2[0], pos_2[1]))
        theta_1: float = groups_mean_orientation[group_id_1]
        theta_2: float = groups_mean_orientation[group_id_2]
        aff = abs(math.cos(theta_1 - theta_12) * math.cos(theta_2 - theta_12)) ** 2.0
        # aff = abs(math.cos(theta_1 - theta_12) * math.cos(theta_2 - theta_12)) ** 0.25  #** 2.0 TODO Eigentlich sollte hier quadriert werden
        if aff <= 0.05:
            return 0.0
        return aff

    # Code um die Ausrichtung des Winkels zu testen
    # def calculate_color_from_group(group_id: int):
    #     if group_id == -1:
    #         return [0.0, 0.0, 0.0, 0.0]
    #     group_coord = groups_mean_position[group_id]
    #     angle = calc_angle_between_points(group_coord, (len(edges_with_grouping) // 2, len(edges_with_grouping) // 2))
    #     rgb = colorsys.hsv_to_rgb(angle, 1.0, 1.0)
    #     return [rgb[0], rgb[1], rgb[2], 1.0]
    #
    #
    # return np.array([[calculate_color_from_group(edges_with_grouping[row_idx, px_idx, 1])
    #                   for px_idx in range(len(edges_with_grouping[0]))]
    #                  for row_idx in range(len(edges_with_grouping))])

    number_of_groups: int = len(groups_members)
    affinities: ndarray = np.zeros(shape=(number_of_groups, number_of_groups))
    for group_id_row in range(number_of_groups):
        for group_id_column in range(number_of_groups):  # range(group_id_row, number_of_groups):
            affinities[group_id_row, group_id_column] = calculate_affinity(group_id_row, group_id_column)
    return affinities


def get_weights(edges_with_grouping_orig: ndarray,
                groups_members: ndarray,
                affinities: ndarray,
                left: int, top: int, right: int, bottom: int) -> List[float]:
    edges_with_grouping = edges_with_grouping_orig.copy()
    edges_with_grouping[top:bottom, left:right, 1] = -1
    groups_not_in_box = np.unique(edges_with_grouping[:, :, 1])

    def calculate_weight(affs: ndarray, group_id: int):
        def generate_paths(group_len: int, length: int):
            paths: list = [[group_id]]
            for _ in range(length):
                paths = [p + [new_group_id]
                         for p in paths
                         for new_group_id in range(group_len)
                         if new_group_id != p[-1]
                         and affs[new_group_id, p[-1]] > 0.0
                         and not (new_group_id in p)]
            return list(filter(lambda p: p[-1] in groups_not_in_box, paths))

        if group_id in groups_not_in_box:
            return 0.0
        max_path_length = 10
        max_chained_affinity = 0.0
        for i in range(max_path_length):
            for path in generate_paths(len(groups_members), i):
                path1 = path[0:-1]
                path2 = path[1:]
                adjacent_path = zip(path1, path2)
                affinity_path = map(lambda v12: affinities[v12[0], v12[1]], adjacent_path)
                affinity_reduced = reduce(lambda a1, a2: a1 * a2, affinity_path)
                max_chained_affinity = max(affinity_reduced, max_chained_affinity)
        return 1.0 - max_chained_affinity

    w = [calculate_weight(affinities, group_id) for group_id in range(len(groups_members))]
    return w


def image_2_foundation(img: ndarray) -> EdgeboxFoundation:
    edges_nms, orientation_map = detect_edges(img)
    edges_with_grouping, groups_members = group_edges(edges_nms, orientation_map)
    affinities = calculate_affinities(groups_members, orientation_map)
    return EdgeboxFoundation(edges_nms, edges_with_grouping, groups_members, affinities)


def get_objectness(foundation: EdgeboxFoundation,
                   left: int, top: int, right: int, bottom: int) -> (float, float):
    def sum_magnitudes(matrix: ndarray, members: ndarray):
        return np.sum(list(map(lambda coord: matrix[coord[0], coord[1]], members)))

    groups_in_box = np.unique(foundation.edges_with_grouping[top:bottom, left:right, 1])
    sum_of_magnitudes: list = [sum_magnitudes(foundation.edges_nms, members) for members in foundation.groups_members]

    w = get_weights(foundation.edges_with_grouping, foundation.groups_members, foundation.affinities,
                    left, top, right, bottom)
    h = np.sum(list(map(lambda group_id: w[group_id] * sum_of_magnitudes[group_id], groups_in_box)))
    h /= 2 * (((right - left) + (bottom - top)) ** 1.5)
    # TODO: Grober Fehler entdeck! "sub" sollte durch eine kleinere Box berechnet werden!
    # Entweder der Fehler wird korrigiert oder die zweite Berechnungsmethode wird nicht mehr angeboten.
    # Immerhin stehen bereits einige andere zur Verfügung
    # => Die fehlerhafte Version h_in scheint besser zu performen als das korrekte h. Ist h wirklich korrekt?
    relevant_for_sub = filter(lambda group_id: w[group_id] < 1.0, groups_in_box)
    sub = np.sum(list(map(lambda group_id: sum_of_magnitudes[group_id] * (1.0 - w[group_id]), relevant_for_sub)))
    sub /= 2 * (((right - left) + (bottom - top)) ** 1.5)
    h_in = h - sub
    return h, h_in


def do_all(img: ndarray, left: int, top: int, right: int, bottom: int) -> (float, float):
    edges_nms, orientation_map = detect_edges(img)
    edges_nms_grouped, groups_members = group_edges(edges_nms, orientation_map)
    affinities = calculate_affinities(groups_members, orientation_map)
    # Der Wertebereich der Ergebnisse hier ist unklar
    # 1. Man ermittelt das Optimum eines gegebenen Bildes und vergleicht daran
    # 2. Man berechnet erst die Werte aller Eingaben und vergleicht sie untereinander
    return get_objectness(edges_nms, edges_nms_grouped, groups_members, affinities, left, top, right, bottom)


