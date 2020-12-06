import cv2
import datetime
import json

import edgebox as eb
import edgeboxcol as ebc
import multiscale_saliency as ms
import numpy as np
from numpy.core.multiarray import ndarray
from multiprocessing import Pool
from attentionmask.mask import decode

from skimage.transform import resize


def do_things_with_visualizations(img: ndarray, left: int, top: int, right: int, bottom: int):
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


def segmentation_2_borders_and_mask(seg):
    mask = decode(seg)
    coords = np.where(mask == 1)
    row_coords = coords[0]
    col_coords = coords[1]
    return np.min(col_coords), np.min(row_coords), np.max(col_coords), np.max(row_coords), (mask == 1)


def process_single_proposal(img: ndarray, proposal: dict):
    left, top, right, bottom, mask = segmentation_2_borders_and_mask(proposal['segmentation'])
    objectness = eb.do_all(img, left, top, right, bottom)
    return objectness


def parallel_calc(mask_path, image_path):
    img = cv2.imread(image_path)
    with open(mask_path) as file:
        data = json.load(file)
    tupled_data = list(map(lambda x: (img, x), data))
    with Pool(6) as pool:
        return pool.starmap(process_single_proposal, tupled_data)


if __name__ == '__main__':

    test_img = cv2.imread("assets/testImage_brutalismus.jpg")
    # cc.get_objectness(test_img, 0, 0, 100, 100)
    # exit()
    #
    # test_img = cv2.imread("assets/testImage_giraffe.jpg")
    # original_shape = test_img.shape
    # saliency = ms.calculate_multiscale_saliency(test_img, 1)
    # cv2.imshow("saliency1", resize(saliency, original_shape))
    # saliency = ms.calculate_multiscale_saliency(test_img, 2)
    # cv2.imshow("saliency2", resize(saliency, original_shape))
    # saliency = ms.calculate_multiscale_saliency(test_img, 3)
    # cv2.imshow("saliency3", resize(saliency, original_shape))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit()

    # test_img = cv2.imread("assets/testImage_schreibtisch.jpg")
    # do_things_with_visualizations(test_img, 0, 40, 550, 150)
    before = datetime.datetime.now()
    input_map = [(test_img, 0, 40, 550, 150), (test_img, 0, 40, 550, 266)]

    def hugo(x):
        return eb.do_all(x[0], x[1], x[2], x[3], x[4])

    with Pool(2) as p:
        result = p.map(hugo, input_map)
    # eb.do_all(test_img, 0, 40, 550, 150)    # halbes Gebäude
    # eb.do_all(test_img, 0, 40, 550, 266)    # fast ganzes Gebäude
    print(result)
    after = datetime.datetime.now()
    print("double run: \t" + str(after - before))
