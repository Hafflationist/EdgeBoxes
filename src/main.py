import argparse
import cv2
import datetime
import itertools
import json
import sys
import math

from numpy import random
from scipy.ndimage import binary_dilation
from skimage.filters import gaussian
from skimage.transform import resize
from attentionmask.mask import decode
from skimage.transform import rescale

from eb import edgebox_coloring as ebc, edgebox as eb, EdgeboxFoundation
from cc import color_contrast as cc, ColorContrastFoundation
from ms import multiscale_saliency as ms, MultiscaleSaliencyFoundation
from ss import superpixel_stradling_coloring as ssc, superpixel_stradling as ss, SuperpixelStradlingFoundation
import numpy as np

from numpy.core.multiarray import ndarray
from multiprocessing import Pool
from typing import Tuple, List, Set, Union

from utils.utils import ndarraycoords_2_rect


def do_things_with_visualizations(img: ndarray, left: int, top: int, right: int, bottom: int) -> None:
    b2 = datetime.datetime.now()
    edges_nms, orientation_map = eb.detect_edges(img)
    a = datetime.datetime.now()
    print("detect_edges:\t" + str(a - b2))
    cv2.imshow("edges_nms", edges_nms)

    b = datetime.datetime.now()
    edges_nms_grouped, groups_members = eb.group_edges(edges_nms, orientation_map)
    a = datetime.datetime.now()
    print("group_edges:\t" + str(a - b))

    b = datetime.datetime.now()
    affinities = eb.calculate_affinities(groups_members, orientation_map)
    a = datetime.datetime.now()
    print("affinities:\t\t" + str(a - b))
    # cv2.imshow("affinities", affinities)

    # edges_nms_grouped_colored = ebc.color_grouped_edges(edges_nms_grouped, groups_members, edges_nms, False)
    # cv2.imshow("nms grouped (colored, without magnitude)", edges_nms_grouped_colored)
    # edges_nms_grouped_colored = ebc.color_grouped_edges(edges_nms_grouped, groups_members, edges_nms, True)
    # cv2.imshow("nms grouped (colored with magnitude)", edges_nms_grouped_colored)

    colored_weights = ebc.add_visual_box(
        ebc.color_weights(
            edges_nms_grouped, groups_members, edges_nms,
            eb.get_weights(edges_nms_grouped, groups_members, affinities, left, top, right, bottom),
            True
        ),
        left, top, right, bottom
    )
    cv2.imshow("colored_weights (with mag)", colored_weights)
    colored_weights = ebc.add_visual_box(
        ebc.color_weights(
            edges_nms_grouped, groups_members, edges_nms,
            eb.get_weights(edges_nms_grouped, groups_members, affinities, left, top, right, bottom),
            False
        ),
        left, top, right, bottom
    )
    cv2.imshow("colored_weights (without mag)", colored_weights)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    b = datetime.datetime.now()
    obj = eb.get_objectness(edges_nms, edges_nms_grouped, groups_members, affinities, left, top, right, bottom)
    a = datetime.datetime.now()
    print("get_objectness:\t" + str(a - b))
    a2 = datetime.datetime.now()
    print("all:\t\t\t" + str(a2 - b2))
    print(obj)


def equalize(value: Union[float, int], value_min: Union[float, int], value_max: Union[float, int]) -> float:
    if value_max == value_min:
        return -1.0
    return float(value - value_min) / float(value_max - value_min)


def segmentation_2_mask(seg) -> ndarray:
    mask = decode(seg)
    coords = np.where(mask == 1)
    mask_coords = np.transpose(coords)
    if len(coords[0]) == 0:
        return np.array([(0, 0)])
    return mask_coords


def segmentation_2_borders_and_mask(seg) -> Tuple[int, int, int, int, ndarray]:
    mask = decode(seg)
    coords = np.where(mask == 1)
    mask_coords = np.transpose(coords)
    row_coords = coords[0]
    col_coords = coords[1]
    if len(row_coords) == 0:
        return 0, 0, 0, 0, np.array([(0, 0)])
    return np.min(col_coords), np.min(row_coords), np.max(col_coords), np.max(row_coords), mask_coords


def minmax_of_mask_area(proposals: List[dict]) -> Tuple[int, int]:
    areas = list(map(lambda prop: len(segmentation_2_mask(prop['segmentation'])), proposals))
    return min(areas), max(areas)


def process_single_proposal(mask: ndarray,
                            mask_scale: float,
                            cc_foundation: ColorContrastFoundation,
                            eb_foundation: EdgeboxFoundation,
                            ms_foundation: MultiscaleSaliencyFoundation,
                            ss_foundation: SuperpixelStradlingFoundation,
                            weights: Tuple[float, float, float, float, float],
                            theta_cc: float,
                            theta_ms: float) -> Tuple[float, float, float, float, float, float, float, float, float]:
    left, top, right, bottom = ndarraycoords_2_rect(mask)
    cc_objectness, eb_objectness, ms_objectness_1, ms_objectness_2, ms_objectness_3, ms_objectness_4, ms_objectness_5, ss_objectness = \
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if abs(weights[0]) > 0.0001:
        cc_objectness = cc.get_objectness(cc_foundation, mask, mask_scale, theta_cc)
    if abs(weights[1]) > 0.0001:
        eb_objectness = eb.get_objectness(eb_foundation, left, top, right, bottom)[0]
        if math.isnan(eb_objectness):
            print("NAN-value found in EB!")
    if abs(weights[2]) > 0.0001:
        ms_objectness_1, ms_objectness_2, ms_objectness_3, ms_objectness_4, ms_objectness_5 = \
            ms.get_objectness(ms_foundation, mask, mask_scale, theta_ms)
    if abs(weights[3]) > 0.0001:
        ss_objectness = ss.get_objectness(ss_foundation, mask)

    return cc_objectness, \
           eb_objectness, \
           ms_objectness_1, ms_objectness_2, ms_objectness_3, ms_objectness_4, ms_objectness_5, \
           ss_objectness, \
           random.uniform(0.0, 0.2)


def process_proposal_group(image_id: int,
                           proposals: List[dict],
                           weights: Tuple[float, float, float, float, float],
                           theta_cc: float,
                           theta_ms: float,
                           theta_ss: float) -> List[dict]:
    area_min, area_max = minmax_of_mask_area(proposals)

    img = cv2.imread("/export2/scratch/8robohm/ba/val2014/COCO_val2014_" + str(image_id).zfill(12) + ".jpg")
    if img is None:
        with open("../missingFiles.sh", "a") as missingFiles:
            missingFiles.write("scp /data_c/coco/val2014/COCO_val2014_{}.jpg ccblade3:/export2/scratch/8robohm/ba/val2014\n".format(str(image_id).zfill(12)))
        print("Image COCO_val2014_{0}.jpg not found!".format(str(image_id).zfill(12)))
        return proposals
    assert (img is not None)

    cc_foundation, eb_foundation, ms_foundation, ss_foundation = None, None, None, None
    if abs(weights[0]) > 0.0001:
        cc_foundation: ColorContrastFoundation = cc.image_2_foundation(img)
        print("cc_foundation calculated!")
    if abs(weights[1]) > 0.0001:
        eb_foundation: EdgeboxFoundation = eb.image_2_foundation(img)
        print("eb_foundation calculated!")
    if abs(weights[2]) > 0.0001:
        ms_foundation: MultiscaleSaliencyFoundation = ms.image_2_foundation(img)
        print("ms_foundation calculated!")
    if abs(weights[3]) > 0.0001:
        ss_foundation: SuperpixelStradlingFoundation = ss.image_2_foundation(img, theta_ss)
        print("ss_foundation calculated!")

    def new_proposal(old_proposal: dict) -> Tuple[dict, float, float, float, float, float, float, float, float, float]:
        mask = segmentation_2_mask(old_proposal['segmentation'])
        cc_objectness, \
        eb_objectness, \
        ms_objectness_1, ms_objectness_2, ms_objectness_3, ms_objectness_4, ms_objectness_5, \
        ss_objectness, \
        random_objectness = \
            process_single_proposal(mask,
                                    equalize(len(mask), area_min, area_max),
                                    cc_foundation,
                                    eb_foundation,
                                    ms_foundation,
                                    ss_foundation,
                                    weights,
                                    theta_cc,
                                    theta_ms)
        return old_proposal, \
               cc_objectness, \
               eb_objectness, \
               ms_objectness_1, ms_objectness_2, ms_objectness_3, ms_objectness_4, ms_objectness_5, \
               ss_objectness, \
               random_objectness

    new_proposals: List[Tuple[dict, float, float, float, float, float, float, float, float, float]] = list(map(new_proposal, proposals))

    def min_max_from_idx(idx: int) -> Tuple[float, float]:
        objn_list = list(map(lambda p: p[idx], new_proposals))
        if idx == 2 and (math.isnan(np.min(objn_list)) or math.isnan(np.max(objn_list))): # EB
            print("NaN in min_max_from_idx detected! ({})".format(objn_list))
            exit()
        return np.min(objn_list), np.max(objn_list)

    cc_objn_list_min, cc_objn_list_max = min_max_from_idx(1)
    eb_objn_list_min, eb_objn_list_max = min_max_from_idx(2)
    ms_objn_1_list_min, ms_objn_1_list_max = min_max_from_idx(3)
    ms_objn_2_list_min, ms_objn_2_list_max = min_max_from_idx(4)
    ms_objn_3_list_min, ms_objn_3_list_max = min_max_from_idx(5)
    ms_objn_4_list_min, ms_objn_4_list_max = min_max_from_idx(6)
    ms_objn_5_list_min, ms_objn_5_list_max = min_max_from_idx(7)
    ss_objn_list_min, ss_objn_list_max = min_max_from_idx(8)
    for proposal, cc_objn, eb_objn, ms_objn_1, ms_objn_2, ms_objn_3, ms_objn_4, ms_objn_5, ss_objn, rand_objn in new_proposals:
        cc_objn_eq = equalize(cc_objn, cc_objn_list_min, cc_objn_list_max) * weights[0]
        eb_objn_eq = equalize(eb_objn, eb_objn_list_min, eb_objn_list_max) * weights[1]
        ms_objn_1_eq = equalize(ms_objn_1, ms_objn_1_list_min, ms_objn_1_list_max) * weights[2]
        ms_objn_2_eq = equalize(ms_objn_2, ms_objn_2_list_min, ms_objn_2_list_max) * weights[2]
        ms_objn_3_eq = equalize(ms_objn_3, ms_objn_3_list_min, ms_objn_3_list_max) * weights[2]
        ms_objn_4_eq = equalize(ms_objn_4, ms_objn_4_list_min, ms_objn_4_list_max) * weights[2]
        ms_objn_5_eq = -1.0 # equalize(ms_objn_5, ms_objn_5_list_min, ms_objn_5_list_max) * weights[2]
        ss_objn_eq = equalize(ss_objn, ss_objn_list_min, ss_objn_list_max) * weights[3]
        final_objn = 0.0
        if abs(weights[0]) > 0.0001:
            final_objn += cc_objn_eq
        if abs(weights[1]) > 0.0001:
            final_objn += eb_objn_eq
        if abs(weights[2]) > 0.0001:
            final_objn += max(ms_objn_1_eq, ms_objn_2_eq, ms_objn_3_eq, ms_objn_4_eq, ms_objn_5_eq)
        if abs(weights[3]) > 0.0001:
            final_objn += ss_objn_eq
        if abs(weights[4]) > 0.0001:
            final_objn += rand_objn * weights[4]
        final_objn = 0.0 if math.isnan(final_objn) else final_objn
        proposal['objn'] = final_objn
        proposal['score'] = final_objn

        if math.isnan(final_objn):
            print("final_objn = NAN!! (eb_objn_eq = {}; eb_objn = {}; eb_objn_list_min = {}; eb_objn_list_max = {};)"
                  .format(eb_objn_eq, eb_objn, eb_objn_list_min, eb_objn_list_max))

    return list(map(lambda x: x[0], new_proposals))


def parallel_calc(proposals_path: str,
                  images_nmax: int,
                  weights: Tuple[float, float, float, float, float],
                  suffix: str,
                  theta_cc: float,
                  theta_ms: float,
                  theta_ss: float) -> None:
    open("../missingFiles.sh", "w").close()
    with open(proposals_path) as file:
        data = json.load(file)
    image_ids: Set[int] = set(map(lambda proposal: proposal['image_id'], data))
    data_grouped: List[Tuple[int, List[dict], Tuple[float, float, float, float, float], float, float, float]]
    data_grouped = [(iid,
                     list(filter(lambda proposal: proposal['image_id'] == iid, data)),
                     weights,
                     theta_cc,
                     theta_ms,
                     theta_ss)
                     for iid in image_ids][:images_nmax]
    for group in data_grouped:
        print("Searching for image Image COCO_val2014_{0}.jpg".format(str(group[0]).zfill(12)))
    new_data_grouped_nested: List[List[dict]]
    print("{0} proposal groups found".format(len(data_grouped)))
    with Pool(40) as pool:
        new_data_grouped_nested = pool.starmap(process_proposal_group, data_grouped)
    new_data_grouped: List[dict] = list(itertools.chain.from_iterable(new_data_grouped_nested))
    with open(proposals_path + suffix, "w") as file:
        json.dump(new_data_grouped, file)


def parse_args() -> Tuple[str, int, int, str, float, float, float]:
    parser = argparse.ArgumentParser(description="Objectnessscoring")
    parser.add_argument("-p", "--proposals", help="Path of file with proposals", required=True, default="")
    parser.add_argument("-n", "--nmax",
                        help="Maximum number of images to be processed",
                        required=False,
                        default="infinity")
    parser.add_argument("-c", "--cue",
                        help="Used cue to calculate objectness score (default=all, -1=all, 0=CC, 1=EB, 2=MS, 3=SS, 4=Random, 5=Constant 0)",
                        required=False,
                        default="-1")
    parser.add_argument("-s", "--suffixofoutput",
                        help="Suffix of output file (default=\"\")",
                        required=False,
                        default="")
    parser.add_argument("-x", "--theta_cc",
                        help="Learned parameter for CC (1.0 < theta_cc < 4.0)",
                        required=False,
                        default="2.0")
    parser.add_argument("-y", "--theta_ms",
                        help="Learned parameter for MS (0.0 < theta_ms < 1.0)",
                        required=False,
                        default="0.6")
    parser.add_argument("-z", "--theta_ss",
                        help="Learned parameter for SS (0.0 < theta_ss < 2.0)",
                        required=False,
                        default="1.0")

    argument = parser.parse_args()
    assert (-1 <= int(argument.cue) <= 5)
    nmax = sys.maxsize
    if argument.nmax != "infinity":
        nmax = int(argument.nmax)

    return argument.proposals, \
           nmax, \
           int(argument.cue), \
           argument.suffixofoutput, \
           float(argument.theta_cc), \
           float(argument.theta_ms), \
           float(argument.theta_ss)


def main() -> None:
    proposals_path, images_nmax, cue, suffix, theta_cc, theta_ms, theta_ss = parse_args()
    print("searching for max {0} proposal groups in {1}".format(images_nmax, proposals_path))
    weights = (0.25, 0.25, 0.25, 0.25, 0.0)
    if cue == 0:
        weights = (1.0, 0.0, 0.0, 0.0, 0.0)
    elif cue == 1:
        weights = (0.0, 1.0, 0.0, 0.0, 0.0)
    elif cue == 2:
        weights = (0.0, 0.0, 1.0, 0.0, 0.0)
    elif cue == 3:
        weights = (0.0, 0.0, 0.0, 1.0, 0.0)
    elif cue == 4:
        weights = (0.0, 0.0, 0.0, 0.0, 1.0)
    elif cue == 5:
        weights = (0.0, 0.0, 0.0, 0.0, 0.0)
    parallel_calc(proposals_path, images_nmax, weights, suffix, theta_cc, theta_ms, theta_ss)
    exit()


if __name__ == '__main__':
    main()
    # test_img = cv2.imread("assets/testImage_kantendetektion.png")
    # do_things_with_visualizations(test_img, 350, 350, 550, 600)
    # exit()
    # test_img = np.resize(cv2.imread("assets/testImage_schreibtisch.jpg"), (100, 100))
    # test_img = cv2.imread("assets/testImage_batterien.jpg")
    # test_img = cv2.imread("assets/testImage_bahn.jpg")
    # cv2.imshow("test_img", cv2.imread("assets/testImage_batterien.jpg"))

    test_img = cv2.imread("../assets/testImage_auto.jpg")
    seg = ss.__segmentate(test_img, 0.05)
    cv2.imshow("seg0.05", ssc.color_segmentation(test_img, seg))
    cv2.waitKey(0)
    cv2.destroyAllWindows()





    exit()
    original_shape = test_img.shape
    saliency = ms.__calculate_multiscale_saliency(test_img, 1)
    cv2.imshow("saliency1", resize(saliency, original_shape))
    saliency = ms.__calculate_multiscale_saliency(test_img, 2)
    cv2.imshow("saliency2", resize(saliency, original_shape))
    saliency = ms.__calculate_multiscale_saliency(test_img, 3)
    cv2.imshow("saliency3", resize(saliency, original_shape))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

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
