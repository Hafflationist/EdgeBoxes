import cv2
import datetime

import edgebox as eb
import edgeboxcol as ebc
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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
    test_img_box = ebc.add_visual_box(test_img, 50, 50, 100, 100)
    cv2.imshow("test_img_box", test_img_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

    test_edges_nms, orientation_map = eb.detect_edges(test_img)
    # cv2.imshow("nms", test_edges_nms * 255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    test_edges_nms_colored = ebc.color_edges(test_edges_nms, orientation_map)
    cv2.imshow("nms (colored)", test_edges_nms_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    b = datetime.datetime.now()
    test_edges_nms_grouped, groups_members = eb.group_edges(test_edges_nms, orientation_map)
    a = datetime.datetime.now()
    print("group_edges:\t" + str(a - b))

    b = datetime.datetime.now()
    affinities = eb.calculate_affinities(groups_members, orientation_map)
    a = datetime.datetime.now()
    print("affinities:\t" + str(a - b))
    cv2.imshow("affinities", affinities)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    b = datetime.datetime.now()
    test_edges_nms_grouped_colored = ebc.color_grouped_edges(test_edges_nms_grouped, groups_members, test_edges_nms)
    a = datetime.datetime.now()
    print("color_grouped_edges:\t" + str(a - b))

    cv2.imshow("nms grouped (colored)", test_edges_nms_grouped_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("finished#")
