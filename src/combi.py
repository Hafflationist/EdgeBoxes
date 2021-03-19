import argparse
import json
import cv2
import copy
import numpy as np
from numpy.core.multiarray import ndarray
import math
from sklearn.svm import SVR
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from attentionmask.mask import decode

from joblib import dump, load

from typing import Tuple, List, Optional, Dict, Any, Callable


def load_proposals(proposals_path: str) -> List[dict]:
    with open(proposals_path) as file:
        proposals = json.load(file)
    return proposals


def extract_scores(proposals: List[dict]) -> List[float]:
    return list(map(lambda p: p['objn'], proposals))


def segmentation_2_mask(seg) -> ndarray:
    mask = decode(seg)
    coords = np.where(mask == 1)
    mask_coords = np.transpose(coords)
    if len(coords[0]) == 0:
        return np.array([(0, 0)])
    return mask_coords


def get_X_and_proposals(path_am: str,
                        path_cc: str,
                        path_eb: str,
                        path_ms: str,
                        path_ss: str,
                        algorithms: str) -> Tuple[List[List[float]], List[dict]]:

    scores_list: List[List[float]] = []
    proposals: List[dict] = []

    if "am" in algorithms:  # dirty implicit conversion to true/false
        proposals = load_proposals(path_am)
        scores_am = extract_scores(proposals)
        scores_list.append(scores_am)
        if any(math.isnan(s) for s in scores_am):
            print("AM-scores contain NAN!")
        if any(math.isinf(s) for s in scores_am):
            print("AM-scores contain INF!")

    if "cc" in algorithms:  # dirty implicit conversion to true/false
        proposals = load_proposals(path_cc)
        scores_cc = extract_scores(proposals)
        scores_list.append(scores_cc)
        if any(math.isnan(s) for s in scores_cc):
            print("CC-scores contain NAN!")
        if any(math.isinf(s) for s in scores_cc):
            print("CC-scores contain INF!")

    if "eb" in algorithms:  # dirty implicit conversion to true/false
        proposals = load_proposals(path_eb)
        scores_eb = extract_scores(proposals)
        scores_list.append(scores_eb)
        if any(math.isnan(s) for s in scores_eb):
            print("EB-scores contain NAN!")
        if any(math.isinf(s) for s in scores_eb):
            print("EB-scores contain INF!")

    if "ms" in algorithms:  # dirty implicit conversion to true/false
        proposals = load_proposals(path_ms)
        scores_ms = extract_scores(proposals)
        scores_list.append(scores_ms)
        if any(math.isnan(s) for s in scores_ms):
            print("MS-scores contain NAN!")
        if any(math.isinf(s) for s in scores_ms):
            print("MS-scores contain INF!")

    if "ss" in algorithms:  # dirty implicit conversion to true/false
        proposals = load_proposals(path_ss)
        scores_ss = extract_scores(proposals)
        scores_list.append(scores_ss)
        if any(math.isnan(s) for s in scores_ss):
            print("SS-scores contain NAN!")
        if any(math.isinf(s) for s in scores_ss):
            print("SS-scores contain INF!")

    if "ra" in algorithms:  # dirty implicit conversion to true/false
        images = {image_id : cv2.imread("/export2/scratch/8robohm/ba/val2014/COCO_val2014_" + str(image_id).zfill(12) + ".jpg")
                    for image_id in set(map(lambda p: p['image_id'], proposals))}

        def img_2_r(proposal, img) -> float:
            return float(len(segmentation_2_mask(proposal['segmentation']))) / float(len(img) * len(img[0]))

        def img_2_a(proposal) -> float:
            return float(len(segmentation_2_mask(proposal['segmentation'])))

        scores_r = [img_2_r(p, images[p['image_id']])
                    for p in proposals]
        scores_a = [img_2_a(p)
                    for p in proposals]
        scores_list.append(scores_r)
        scores_list.append(scores_a)

    X: List[List[float]] = [list(i) for i in zip(*scores_list)]  # transposing list of lists
    return  X, proposals
