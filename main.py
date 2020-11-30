import cv2
import datetime
import json

import edgebox as eb
import edgeboxcol as ebc
import numpy as np
from multiprocessing import Pool
from attentionmask.mask import decode


def do_things_with_visualizations(img, left: int, top: int, right: int, bottom: int):
    b2 = datetime.datetime.now()
    edges_nms, orientation_map = eb.detect_edges(img)
    a = datetime.datetime.now()
    print("detect_edges:\t" + str(a - b2))

    b = datetime.datetime.now()
    edges_nms_grouped, groups_members = eb.group_edges(edges_nms, orientation_map)
    a = datetime.datetime.now()
    print("group_edges:\t" + str(a - b))

    b = datetime.datetime.now()
    affinities = eb.calculate_affinities(groups_members, orientation_map)
    a = datetime.datetime.now()
    print("affinities:\t\t" + str(a - b))
    # cv2.imshow("affinities", affinities)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    edges_nms_grouped_colored = ebc.color_grouped_edges(edges_nms_grouped, groups_members, edges_nms)
    cv2.imshow("nms grouped (colored)", edges_nms_grouped_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    colored_weights = ebc.add_visual_box(
        ebc.color_weights(
            edges_nms_grouped, groups_members, edges_nms,
            eb.get_weights(edges_nms_grouped, groups_members, affinities, left, top, right, bottom)
        ),
        left, top, right, bottom
    )
    cv2.imshow("colored_weights", colored_weights)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    b = datetime.datetime.now()
    obj = eb.get_objectness(edges_nms, edges_nms_grouped, groups_members, affinities, left, top, right, bottom)
    a = datetime.datetime.now()
    print("get_objectness:\t" + str(a - b))
    a2 = datetime.datetime.now()
    print("all:\t\t\t" + str(a2 - b2))
    print(obj)


def segmentation_2_borders(seg):
    mask = decode(seg)
    coords = np.where(mask == 1)
    print(coords)
    row_coords = coords[0]
    col_coords = coords[1]
    return np.min(col_coords), np.min(row_coords), np.max(col_coords), np.max(row_coords)


def parallel_calc(mask_path, image_path):
    def process_single_proposal(proposal):
        left, top, right, bottom = segmentation_2_borders(proposal['segmentation'])
        objectness = eb.do_all(img, left, top, right, bottom)
        return objectness

    img = cv2.imread(image_path)
    with open(mask_path) as file:
        data = json.load(file)
    with Pool(6) as pool:
        pool.map(process_single_proposal, data)


if __name__ == '__main__':
    with open('assets/attentionMask-8-128.json') as f:
        d = json.load(f)
    print(segmentation_2_borders(d[0]['segmentation']))
    exit()
    # group_id = 0
    # groups_not_in_box = [2,3]
    # affinities = np.array([
    #     [1.0, 1.0, 1.0, 0.0, 0.0],
    #     [1.0, 1.0, 1.0, 0.0, 0.0],
    #     [1.0, 1.0, 1.0, 1.0, 1.0],
    #     [0.0, 0.0, 1.0, 1.0, 1.0],
    #     [0.0, 0.0, 1.0, 1.0, 1.0]
    # ])
    #
    # def generate_paths(group_len: int, length: int):
    #     paths: list = [[group_id]]
    #     for _ in range(length - 1):
    #         paths = [p + [new_group_id]
    #                  for p in paths
    #                  for new_group_id in range(group_len)
    #                  if new_group_id != p[-1]
    #                  and affinities[new_group_id, p[-1]] > 0.0
    #                  and not(new_group_id in p)]
    #     return list(filter(lambda p: p[-1] in groups_not_in_box, paths))
    #
    # print(generate_paths(5, 1))
    # print(generate_paths(5, 2))
    # print(generate_paths(5, 3))
    # print(generate_paths(5, 4))
    # print(generate_paths(5, 5))
    # exit()

    test_img = cv2.imread("assets/testImage.jpg")
    # test_img = cv2.imread("assets/testImage2.jpg")
    # do_things_with_visualizations(test_img, 0, 40, 550, 150)
    before = datetime.datetime.now()
    input = [(test_img, 0, 40, 550, 150), (test_img, 0, 40, 550, 266)]

    def hugo(x):
        return eb.do_all(x[0], x[1], x[2], x[3], x[4])

    with Pool(2) as p:
        result = p.map(hugo , input)
    # eb.do_all(test_img, 0, 40, 550, 150)    # halbes Gebäude
    # eb.do_all(test_img, 0, 40, 550, 266)    # fast ganzes Gebäude
    print(result)
    after = datetime.datetime.now()
    print("double run: \t" + str(after - before))
