import numpy as np
import colorsys
import random
import math

from numpy.core.multiarray import ndarray


def color_edges(edges_nms: ndarray, orientation_map: ndarray):
    def calculate_pixel(row_idx, px_idx):
        if edges_nms[row_idx, px_idx] < 0.2:
            return [0.0, 0.0, 0.0, 0.0]
        o = orientation_map[row_idx, px_idx]
        rgb = colorsys.hsv_to_rgb(o / math.pi, 1.0, 1.0)
        intensity = edges_nms[row_idx, px_idx]
        return [intensity * rgb[0], intensity * rgb[1],
                intensity * rgb[2], 1.0]

    edges_nms_colored = [[calculate_pixel(row_idx, px_idx)
                          for px_idx in range(len(edges_nms[0]))]
                         for row_idx in range(len(edges_nms))]
    return np.array(edges_nms_colored) / np.max(edges_nms_colored)


# returns RGB-image
def color_grouped_edges(edges_with_grouping: ndarray, groups_members: ndarray, edges_nms: ndarray):
    def get_summed_magnitude(matrix, members):
        mag_sum = 0.0
        for (row_idx, px_idx) in members:
            mag_sum += edges_nms[row_idx, px_idx]
        return mag_sum

    def calculate_color_from_group(edge_magnitude: float, group_id: int):
        if edge_magnitude < 0.1:
            return [0.0, 0.0, 0.0, 0.0]
        rgb = colorsys.hsv_to_rgb(group_id_2_hue[group_id], 1.0, 1.0)
        alpha = sum_of_magnitudes[group_id]
        return [alpha * rgb[0], alpha * rgb[1], alpha * rgb[2], 1.0]

    sum_of_magnitudes: list = [get_summed_magnitude(edges_nms, members) for members in groups_members]
    sum_of_magnitudes = sum_of_magnitudes / np.max(sum_of_magnitudes)
    group_id_2_hue = {i: random.random() for i in range(np.max(edges_with_grouping) + 1)}
    # group_id_2_hue = {i: 0.5 for i in range(np.max(edges_with_grouping) + 1)}
    edges_nms_colored = [[calculate_color_from_group(px[0], px[1])
                          for px in row]
                         for row in edges_with_grouping]
    return np.array(edges_nms_colored)


def add_visual_box(img_orig: ndarray, left: int, top: int, right: int, bottom: int):
    img = img_orig.copy()
    max_value = np.max(img)
    img[top:bottom + 2, left:left + 3, :] = max_value           # left
    img[top:bottom + 2, right - 1:right + 2, :] = max_value     # right
    img[top:top + 3, left:right + 2, :] = max_value             # top
    img[bottom - 1:bottom + 2, left:right + 2, :] = max_value   # bottom
    return img


def color_weights(edges_with_grouping: ndarray, groups_members: ndarray, edges_nms: ndarray, w: ndarray):
    def get_summed_magnitude(matrix, members):
        mag_sum = 0.0
        for (row_idx, px_idx) in members:
            mag_sum += edges_nms[row_idx, px_idx]
        return mag_sum

    def calculate_color_from_group(edge_magnitude: float, group_id: int):
        weight = w[group_id]
        if edge_magnitude < 0.1:
            return [0.0, 0.0, 0.0, 0.0]
        return [0.0, weight * sum_of_magnitudes[group_id], (1.0 - weight) * sum_of_magnitudes[group_id], 1.0]

    sum_of_magnitudes: list = [get_summed_magnitude(edges_nms, members) for members in groups_members]
    sum_of_magnitudes = sum_of_magnitudes / np.max(sum_of_magnitudes)
    edges_nms_colored = [[calculate_color_from_group(px[0], px[1])
                          for px in row]
                         for row in edges_with_grouping]
    return np.array(edges_nms_colored)


def generate_test():
    def calc_angle(px, row):
        divisor = (px - 250.0)
        if (px - 250.0) == 0:
            divisor = 0.0001

        o = (np.arctan((row - 250.0) / divisor) + (math.pi / 2.0)) / math.pi
        rgb = colorsys.hsv_to_rgb(o, 1.0, 1.0)
        return [rgb[0], rgb[1], rgb[2], 1.0]

    return np.array([[calc_angle(row, px)
                      for row in range(501)]
                     for px in range(501)])


def generate_test_2():
    def calc_color(row, px):
        if row < 125:
            rgb = colorsys.hsv_to_rgb(px / 500.0, 1.0, 1.0)
            return [rgb[0], rgb[1], rgb[2], 1.0]
        else:
            return [px / 500.0, px / 500.0, px / 500.0, 1.0]
    return np.array([[calc_color(row, px)
                      for px in range(500)]
                     for row in range(250)])
