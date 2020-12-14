import argparse
import cv2
import datetime
import itertools
import json
import sys

from skimage.filters import gaussian
from src.attentionmask.mask import decode
from skimage.transform import rescale

from src.eb import edgebox_coloring as ebc, edgebox as eb, EdgeboxFoundation
from src.cc import color_contrast as cc, ColorContrastFoundation
from src.ms import multiscale_saliency as ms, MultiscaleSaliencyFoundation
from src.ss import superpixel_stradling_coloring as ssc, superpixel_stradling as ss, SuperpixelStradlingFoundation
import numpy as np

from numpy.core.multiarray import ndarray
from multiprocessing import Pool
from typing import Tuple, Any, List, Set


def do_things_with_visualizations(img: ndarray, left: int, top: int, right: int, bottom: int) -> None:
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


def segmentation_2_borders_and_mask(seg) -> Tuple[int, int, int, int, ndarray]:
    mask = decode(seg)
    coords = np.where(mask == 1)
    mask_coords = np.transpose(coords)
    row_coords = coords[0]
    col_coords = coords[1]
    return np.min(col_coords), np.min(row_coords), np.max(col_coords), np.max(row_coords), mask_coords


def process_single_proposal(proposal: dict,
                            cc_foundation: ColorContrastFoundation,
                            eb_foundation: EdgeboxFoundation,
                            ms_foundation: MultiscaleSaliencyFoundation,
                            ss_foundation: SuperpixelStradlingFoundation) -> Tuple[float, float, float, float]:
    left, top, right, bottom, mask = segmentation_2_borders_and_mask(proposal['segmentation'])

    cc_objectness = cc.get_objectness(cc_foundation, left, top, right, bottom)
    eb_objectness = eb.get_objectness(eb_foundation, left, top, right, bottom)[1]
    ms_objectness = ms.get_objectness(ms_foundation, mask)
    ss_objectness = ss.get_objectness(ss_foundation, mask)

    return cc_objectness, eb_objectness, ms_objectness, ss_objectness


def process_proposal_group(image_id: int,
                           proposals: List[dict],
                           weights: Tuple[float, float, float, float]) -> List[dict]:
    img = cv2.imread("/data_c/coco/val2014/COCO_val2014_" + str(image_id).zfill(12) + ".jpg")
    assert (img is not None)
    cc_foundation: ColorContrastFoundation = cc.image_2_foundation(img)
    eb_foundation: EdgeboxFoundation = eb.image_2_foundation(img)
    ms_foundation: MultiscaleSaliencyFoundation = ms.image_2_foundation(img)
    ss_foundation: SuperpixelStradlingFoundation = ss.image_2_foundation(img)

    def new_proposal(old_proposal: dict):
        cc_objectness, eb_objectness, ms_objectness, ss_objectness = process_single_proposal(old_proposal,
                                                                                             cc_foundation,
                                                                                             eb_foundation,
                                                                                             ms_foundation,
                                                                                             ss_foundation)
        return old_proposal, cc_objectness, eb_objectness, ms_objectness, ss_objectness

    new_proposals = list(map(new_proposal, proposals))

    def min_max_from_idx(idx: int) -> Tuple[float, float]:
        objn_list = list(map(lambda p: p[idx], new_proposals))
        return np.min(objn_list), np.max(objn_list)

    def equalize(value: float, value_min: float, value_max: float) -> float:
        return (value - value_min) / (value_max - value_min)

    cc_objn_list_min, cc_objn_list_max = min_max_from_idx(1)
    eb_objn_list_min, eb_objn_list_max = min_max_from_idx(2)
    ms_objn_list_min, ms_objn_list_max = min_max_from_idx(3)
    ss_objn_list_min, ss_objn_list_max = min_max_from_idx(4)
    for proposal, cc_objn, eb_objn, ms_objn, ss_objn in new_proposals:
        cc_objn_eq = equalize(cc_objn, cc_objn_list_min, cc_objn_list_max) * weights[0]
        eb_objn_eq = equalize(eb_objn, eb_objn_list_min, eb_objn_list_max) * weights[1]
        ms_objn_eq = equalize(ms_objn, ms_objn_list_min, ms_objn_list_max) * weights[2]
        ss_objn_eq = equalize(ss_objn, ss_objn_list_min, ss_objn_list_max) * weights[3]
        proposal['objn'] = cc_objn_eq + eb_objn_eq + ms_objn_eq + ss_objn_eq
    return list(map(lambda x: x[0], new_proposals))


def parallel_calc(proposals_path: str,
                  proposals_nmax: int,
                  weights: Tuple[float, float, float, float],
                  suffix: str) -> None:
    with open(proposals_path) as file:
        data = json.load(file)[:proposals_nmax]
    image_ids: Set[int] = set(map(lambda proposal: proposal['image_id'], data))
    data_grouped: List[Tuple[int, List[dict], Tuple[float, float, float, float]]]
    data_grouped = [(iid,
                     list(filter(lambda proposal: proposal['image_id'] == iid, data)),
                     weights)
                    for iid in image_ids]
    new_data_grouped_nested: List[List[dict]]
    print("{0} proposal groups found".format(len(data_grouped)))
    with Pool(1) as pool:
        new_data_grouped_nested = pool.starmap(process_proposal_group, data_grouped)
    new_data_grouped: List[dict] = list(itertools.chain.from_iterable(new_data_grouped_nested))
    with open(proposals_path + suffix, "w") as file:
        json.dump(new_data_grouped, file)


def parse_args() -> Tuple[str, int, int, str]:
    parser = argparse.ArgumentParser(description="Description for my parser")
    parser.add_argument("-p", "--proposals", help="Path of file with proposals", required=True, default="")
    parser.add_argument("-n", "--nmax",
                        help="Maximum number of proposals to be processed",
                        required=False,
                        default="infinity")
    parser.add_argument("-c", "--cue",
                        help="Used cue to calculate objectness score (default=all, -1=all, 0=CC, 1=EB, 2=MS, 3=SS)",
                        required=False,
                        default="-1")
    parser.add_argument("-s", "--suffixofoutput",
                        help="Suffix of output file (default=\"\")",
                        required=False,
                        default="")

    argument = parser.parse_args()
    assert(-1 <= int(argument.cue) <= 3)
    nmax = sys.maxsize
    if argument.nmax != "infinity":
        nmax = int(argument.nmax)

    return argument.proposals, nmax, int(argument.cue), argument.suffixofoutput


def main() -> None:
    proposals_path, proposals_nmax, cue, suffix = parse_args()
    print("searching for max {0} proposals in {1}".format(proposals_nmax, proposals_path))
    weights = (0.25, 0.25, 0.25, 0.25)
    if cue == 0:
        weights = (1.0, 0.0, 0.0, 0.0)
    elif cue == 1:
        weights = (0.0, 1.0, 0.0, 0.0)
    elif cue == 2:
        weights = (0.0, 0.0, 1.0, 0.0)
    elif cue == 3:
        weights = (0.0, 0.0, 0.0, 1.0)
    parallel_calc(proposals_path, proposals_nmax, weights, suffix)
    exit()


if __name__ == '__main__':
    main()
    # test_img = np.resize(cv2.imread("assets/testImage_schreibtisch.jpg"), (100, 100))
    test_img = cv2.imread("assets/testImage_kubus.jpg")
    # test_img = rescale(cv2.imread("assets/testImage_strand.jpg"), (0.3, 0.3, 1.0))
    # test_img = rescale(cv2.imread("assets/testImage_brutalismus.jpg"), (1.0, 1.0, 1.0))
    cv2.imshow("test_img", np.array(test_img))
    cv2.imshow("test_img (blur)", gaussian(np.array(test_img), sigma=1.5))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    S = ss.__segmentate(test_img)
    seg_test = ssc.color_segmentation(test_img, S)
    cv2.imshow("seg_test", seg_test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
