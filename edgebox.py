import cv2
import numpy as np
import colorsys
import random


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
        pi = 3.14159265358979
        rgb = colorsys.hsv_to_rgb(o / pi, 1.0, 1.0)
        return [rgb[0], rgb[1], rgb[2], 1.0]

    edges_nms_colored = [[calculate_pixel(row_idx, px_idx)
                          for px_idx in range(len(edges_nms[0]))]
                         for row_idx in range(len(edges_nms))]
    return np.array(edges_nms_colored)


# each pixel consists of:
# 1. edge magnitude (0.0 to 1.0)
# 2. group id
def group_edges(edges_nms, orientation_map):
    def get_testable_coords(matrix, r_idx: int, p_idx: int):
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

    def get_new_todo(matrix):
        todo = [coord for coord in coords_of_edges if matrix[coord[0], coord[1], 1] == -1]
        print(len(todo))
        if len(todo) == 0:
            return -1, -1
        return todo[0]

    def get_next_todo(matrix, curr_r_idx: int, curr_p_idx: int):
        root_coord = groups_members[edges_with_grouping[curr_r_idx][curr_p_idx][1]][0]
        for (ro, pi) in sorted(get_testable_coords(matrix, curr_r_idx, curr_p_idx),
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
    half_pi = 3.14159265358979 / 2.0

    (row_idx, px_idx) = get_new_todo(edges_with_grouping)
    while True:
        if row_idx == -1 or px_idx == -1:
            break

        new_group_id_candidate = new_group_id
        # check N8 neighborhood
        px_orientation = orientation_map[row_idx, px_idx]
        for (r, p) in get_testable_coords(edges_nms, row_idx, px_idx):
            if edges_nms[r, p] != 1 \
                    or edges_with_grouping[r][p][1] == -1 \
                    or groups_diff_cum[edges_with_grouping[r][p][1]] > half_pi:
                continue
            current_diff = abs(px_orientation - orientation_map[r, p])
            current_diff = min(3.14159265358979 - current_diff, current_diff)  # difference in a circle
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
    edges_nms_colored = [[calculate_color_from_group(px[0], px[1])
                          for px in row]
                         for row in edges_with_grouping]
    return np.array(edges_nms_colored)
