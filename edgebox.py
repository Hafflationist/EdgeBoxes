import cv2
import numpy as np
import colorsys
import random
import math


def detect_edges(img):
    img_processed = (img / np.max(img)).astype(np.float32)
    modelFilename = "model/model.yml.gz"
    pDollar = cv2.ximgproc.createStructuredEdgeDetection(modelFilename)
    edges = pDollar.detectEdges(cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))
    orientation_map = pDollar.computeOrientation(edges)
    edges_nms = pDollar.edgesNms(edges, orientation_map)
    return edges_nms, orientation_map


def color_edges(edges_nms, orientation_map):
    def calculate_pixel(row_idx, px_idx):
        if edges_nms[row_idx, px_idx] < 0.1:
            return [0.0, 0.0, 0.0, 0.0]
        o = orientation_map[row_idx, px_idx]
        rgb = colorsys.hsv_to_rgb(o / math.pi, 1.0, 1.0)
        return [rgb[0], rgb[1], rgb[2], 1.0]

    edges_nms_colored = [[calculate_pixel(row_idx, px_idx)
                          for px_idx in range(len(edges_nms[0]))]
                         for row_idx in range(len(edges_nms))]
    return np.array(edges_nms_colored)


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
def group_edges(edges_nms, orientation_map):

    def get_new_todo(matrix):
        todo = [coord for coord in coords_of_edges if matrix[coord[0], coord[1], 1] == -1]
        print(len(todo))
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
            groups_members[new_group_id_candidate].append((row_idx, px_idx))
            groups_diff_cum[new_group_id_candidate] += current_diff
            break
        else:
            # new group created:
            groups_diff_cum.append(0.0)
            groups_members.append([(row_idx, px_idx)])
            new_group_id += 1

        edges_with_grouping[row_idx][px_idx] = [edges_nms[row_idx, px_idx], new_group_id_candidate]
        (row_idx, px_idx) = get_next_todo(edges_with_grouping, row_idx, px_idx)

    print("#groups: " + str(new_group_id))
    print("#edgepxs: " + str(len(np.where(edges_nms == 1)[0])))
    return edges_with_grouping, np.array(groups_members)


# returns RGB-image
def color_grouped_edges(edges_with_grouping):
    def calculate_color_from_group(edge_magnitude: float, group_id: int):
        if edge_magnitude < 0.1:
            return [0.0, 0.0, 0.0, 0.0]
        rgb = colorsys.hsv_to_rgb(group_id_2_hue[group_id], 1.0, 1.0)
        return [rgb[0], rgb[1], rgb[2], 1.0]

    group_id_2_hue = {i: random.random() for i in range(np.max(edges_with_grouping) + 1)}
    # group_id_2_hue = {i: 0.5 for i in range(np.max(edges_with_grouping) + 1)}
    edges_nms_colored = [[calculate_color_from_group(px[0], px[1])
                          for px in row]
                         for row in edges_with_grouping]
    return np.array(edges_nms_colored)




def generate_test():
    def calc_angle(px, row):
        divisor = (px - 250.0)
        if (px - 250.0) == 0:
            divisor = 0.0001

        o = (np.arctan((row - 250.0)/divisor) + (math.pi / 2.0)) / math.pi
        rgb = colorsys.hsv_to_rgb(o, 1.0, 1.0)
        return [rgb[0], rgb[1], rgb[2], 1.0]

    return np.array([[calc_angle(row, px)
                      for row in range(501)]
                     for px in range(501)])


# returns list<list<float>> (Adjazenzmatrix)
def calculate_affinities(groups_members, orientation_map):
    def mean_of_coords(idx: int) -> (float, float):
        rows = [coord[0] for coord in groups_members[idx]]
        columns = [coord[1] for coord in groups_members[idx]]
        return sum(rows) / len(rows), sum(columns) / len(columns)

    def mean_of_orientations(idx: int) -> float:
        orientations = [orientation_map[coord] for coord in groups_members[idx]]
        return sum(orientations) / len(orientations)

    groups_mean_position = [mean_of_coords(idx) for idx in range(len(groups_members))]
    groups_mean_orientation = [mean_of_orientations(idx) for idx in range(len(groups_members))]

    def calc_angle_between_points(coord_1: (int, int), coord_2: (int, int)) -> float:
        coord_diff = list(map(lambda a, b: a - b, coord_1, coord_2))
        if coord_diff[1] == 0.0:
            coord_diff[1] = 0.0001
        return (np.arctan(coord_diff[0]/coord_diff[1]) + (math.pi / 2.0)) / math.pi

    def calculate_affinity(group_id_1: int, group_id_2: int) -> float:
        pos_1 = groups_mean_position[group_id_1]
        pos_2 = groups_mean_position[group_id_2]
        theta_12 = calc_angle_between_points(pos_1, pos_2)
        theta_1 = groups_mean_orientation[group_id_1]
        theta_2 = groups_mean_orientation[group_id_2]
        return abs(math.cos(theta_1 - theta_12) * math.cos(theta_2 - theta_12)) ** 2.0

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

