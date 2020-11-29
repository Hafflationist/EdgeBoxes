import cv2
import numpy as np
import math

from functools import reduce


def detect_edges(img):
    img_processed = (img / np.max(img)).astype(np.float32)
    modelFilename = "model/model.yml.gz"
    pDollar = cv2.ximgproc.createStructuredEdgeDetection(modelFilename)
    edges = pDollar.detectEdges(cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))
    orientation_map = pDollar.computeOrientation(edges)
    edges_nms = pDollar.edgesNms(edges, orientation_map)
    return edges_nms, orientation_map


# return list<int*int>
def get_n8(matrix, r_idx: int, p_idx: int):
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


# each pixel consists of:
# 1. edge magnitude (0.0 to 1.0)
# 2. group id
# returns list<list<[int, int]>>     (Matrix of [edge, group id])
#       * list<list<int*int>>        (List of all group members(coords))
def group_edges(edges_nms_orig, orientation_map):

    def get_new_todo(matrix):
        todo = [coord for coord in coords_of_edges if matrix[coord[0], coord[1], 1] == -1]
        # print(len(todo))
        if len(todo) == 0:
            return -1, -1
        return todo[0]

    def get_next_todo(matrix, curr_r_idx: int, curr_p_idx: int):
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
    groups_diff_cum: list = []          # list<float>
    groups_members: list = []           # list<list<int*int>>
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

        new_group_id_candidate = new_group_id
        # check N8 neighborhood
        px_orientation = orientation_map[row_idx, px_idx]
        for (r, p) in get_n8(edges_nms, row_idx, px_idx):
            if edges_nms[r, p] != 1 \
                    or edges_with_grouping[r][p][1] == -1 \
                    or groups_diff_cum[edges_with_grouping[r][p][1]] > half_pi:
                continue
            current_diff = abs(px_orientation - orientation_map[r, p])
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

    print("#groups: " + str(new_group_id))
    print("#edgepxs: " + str(len(np.where(edges_nms == 1)[0])))
    return edges_with_grouping, np.array(groups_members)


# returns list<list<float>> (Adjazenzmatrix)
def calculate_affinities(groups_members, orientation_map):
    def mean_of_coords(idx: int) -> (float, float):
        rows = [coord[0] for coord in groups_members[idx]]
        columns = [coord[1] for coord in groups_members[idx]]
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

    def calc_distance(group_id_1: int, group_id_2: int):
        if(groups_min_row_idx[group_id_1] - groups_max_row_idx[group_id_2] > 2
                or groups_min_row_idx[group_id_2] - groups_max_row_idx[group_id_1] > 2
                or groups_min_col_idx[group_id_1] - groups_max_col_idx[group_id_2] > 2
                or groups_min_col_idx[group_id_2] - groups_max_col_idx[group_id_1] > 2):
            return 999999999.999
        mean_1 = groups_mean_position[group_id_1]
        mean_2 = groups_mean_position[group_id_2]
        c_with_d_1 = [(r, p, (r - mean_2[0])**2 + (p - mean_2[1])**2) for (r, p) in groups_members[group_id_1]]
        c_with_d_2 = [(r, p, (r - mean_1[0])**2 + (p - mean_1[1])**2) for (r, p) in groups_members[group_id_2]]
        nearest_1 = sorted(c_with_d_1, key=lambda triple: triple[2])[0]
        nearest_2 = sorted(c_with_d_2, key=lambda triple: triple[2])[0]
        return (nearest_1[0] - nearest_2[0])**2 + (nearest_1[1] - nearest_2[1])**2

    def calculate_affinity(group_id_1: int, group_id_2: int) -> float:
        if calc_distance(group_id_1, group_id_2) > 8:
            return 0.0
        pos_1 = groups_mean_position[group_id_1]
        pos_2 = groups_mean_position[group_id_2]
        theta_12 = calc_angle_between_points((pos_1[0], pos_1[1]), (pos_2[0], pos_2[1]))
        theta_1 = groups_mean_orientation[group_id_1]
        theta_2 = groups_mean_orientation[group_id_2]
        aff = abs(math.cos(theta_1 - theta_12) * math.cos(theta_2 - theta_12)) ** 2.0
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

    number_of_groups = len(groups_members)
    affinities = np.zeros(shape=(number_of_groups, number_of_groups))
    for group_id_row in range(number_of_groups):
        for group_id_column in range(number_of_groups): # range(group_id_row, number_of_groups):
            if group_id_column == group_id_row:
                affinities[group_id_row, group_id_column] = 1.0
                continue
            affinities[group_id_row, group_id_column] = calculate_affinity(group_id_row, group_id_column)
    return affinities


def get_weights(edges_with_grouping_orig, groups_members, affinities, left: int, top: int, right: int, bottom: int):
    edges_with_grouping = edges_with_grouping_orig
    # def touches_box(members) -> bool:
    #     for (row_idx, px_idx) in members:
    #         if top <= row_idx <= bottom and left <= px_idx <= right:
    #             return True
    #     return False

    # groups_in_box = set(edges_with_grouping[top:bottom, left:right, 1])
    edges_with_grouping[top:bottom, left:right, 1] = -1
    groups_not_in_box = set(edges_with_grouping[:, :, 1])

    # def is_only_in_box(members) -> bool:
    #     for (row_idx, px_idx) in members:
    #         if not (top <= row_idx <= bottom and left <= px_idx <= right):
    #             return False
    #     return True

    def calculate_weight(affinities, group_id: int):
        def generate_paths(group_len: int, length: int):
            paths: list = [[group_id]]
            for _ in range(length):
                paths = [p + [new_group_id]
                         for p in paths
                         for new_group_id in range(group_len)
                         if new_group_id != p[-1]
                         and affinities[new_group_id, p[-1]] > 0.0
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

    return [calculate_weight(affinities, group_id) for group_id in range(len(groups_members))]


def get_objectness(edges_nms, edges_with_grouping_orig, groups_members, affinities, left: int, top: int, right: int, bottom: int):
    def sum_magnitudes(matrix, members):
        mag_sum = 0.0
        for (row_idx, px_idx) in members:
            mag_sum += matrix[row_idx, px_idx]
        return mag_sum

    sum_of_magnitudes: list = [sum_magnitudes(edges_nms, members) for members in groups_members]

    w = get_weights(edges_with_grouping_orig, groups_members, affinities, left, top, right, bottom)
    # TODO implement algorithms
