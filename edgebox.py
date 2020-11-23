import cv2
import numpy as np
import colorsys


def detect_edges(img):
    img_processed = (img / np.max(img)).astype(np.float32)
    modelFilename = "model/model.yml.gz"
    pDollar = cv2.ximgproc.createStructuredEdgeDetection(modelFilename)
    edges = pDollar.detectEdges(cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))
    orientation_map = pDollar.computeOrientation(edges)
    edges_nms = pDollar.edgesNms(edges, orientation_map)
    edges_nms[edges_nms < 0.1] = 0  # thresholding
    edges_nms[edges_nms >= 0.1] = 1.0  # thresholding
    edges_nms = np.uint8(edges_nms)
    return edges_nms, orientation_map


def color_edges(edges_nms, orientation_map):
    def calculate_pixel(row_idx, px_idx):
        if edges_nms[row_idx, px_idx] < 0.1:
            return [0.0, 0.0, 0.0, 0.0]
        o = orientation_map[row_idx, px_idx]
        pi = 3.14159265358979
        rgb = colorsys.hsv_to_rgb(o / pi, 1.0, 1.0)
        # f = edges_nms[row_idx, px_idx] / np.max(edges_nms)
        # return [edges_nms[row_idx, px_idx] / np.max(edges_nms), rgb[0]*f, rgb[1]*f, rgb[2]*f]
        return [1.0, rgb[0], rgb[1], rgb[2]]

    edges_nms_colored = [[calculate_pixel(row_idx, px_idx)
                          for px_idx in range(len(edges_nms[0]))]
                         for row_idx in range(len(edges_nms))]
    return np.array(edges_nms_colored)


# each pixel consists of:
# 1. edge magnitude (0 or 1)
# 2. group id
def group_edges(edges_nms, orientation_map):
    def get_testable_coords(matrix, r_idx: int, p_idx: int):
        all_possibilities = [(r_idx - 1, p_idx - 1),
                             (r_idx - 1, p_idx),
                             (r_idx - 1, p_idx + 1),
                             (r_idx, p_idx - 1)]
        px_idx_max = len(matrix[0])
        result = list(filter(lambda x: 0 <= x[0] and 0 <= x[1] < px_idx_max, all_possibilities))
        return result

    new_group_id: int = 0
    groups: list = []
    edges_with_grouping = np.array([[[0, -1] for _ in edges_nms[0]] for _ in edges_nms])
    half_pi = 3.14159265358979 / 2.0

    for row_idx in range(len(edges_nms)):
        for px_idx in range(len(edges_nms[0])):
            if edges_nms[row_idx][px_idx] != 1:
                continue

            new_group_id_candidate = new_group_id
            # check N8 neighborhood
            current_diff = 0.0
            px_orientation = orientation_map[row_idx, px_idx]
            for (r, p) in get_testable_coords(edges_nms, row_idx, px_idx):
                if edges_nms[r, p] != 1:
                    continue
                current_diff = abs(px_orientation - orientation_map[r, p])
                current_diff = min(3.14159265358979 - current_diff, current_diff)  # difference in a circle
                if groups[edges_with_grouping[row_idx][px_idx][1]] < half_pi:
                    new_group_id_candidate = edges_with_grouping[r][p][1]
                    # update group information...
                    groups[new_group_id_candidate] += current_diff
                    break
            else:
                # new group created:
                groups.append(current_diff)
                new_group_id += 1

            edges_with_grouping[row_idx][px_idx] = [edges_nms[row_idx, px_idx], new_group_id_candidate]
    print("#groups: " + str(new_group_id))
    print("#edgepxs: " + str(len(np.where(edges_nms == 1)[0])))
    return edges_with_grouping


# returns RGB-image
def color_grouped_edges(edges_with_grouping):
    def calculate_color_from_group(edge_magnitude, group_id):
        if edge_magnitude < 0.1:
            return [0.0, 0.0, 0.0, 0.0]
        rgb = colorsys.hsv_to_rgb(group_id * 0.09 % 1.0, 1.0, 1.0)
        return [1.0, rgb[0], rgb[1], rgb[2]]

    edges_nms_colored = [[calculate_color_from_group(px[0], px[1])
                          for px in row]
                         for row in edges_with_grouping]
    return np.array(edges_nms_colored)

