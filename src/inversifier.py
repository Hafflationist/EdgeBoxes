import argparse
import json
import copy
import math
from sklearn.svm import SVR

from attentionmask.mask import decode
from sklearn import svm

import numpy as np

from typing import Tuple, List

from mypycocotools.mycocoeval import COCOeval



def load_proposals(proposals_path: str) -> List[dict]:
    with open(proposals_path) as file:
        proposals = json.load(file)
    return proposals

def inv(path: str) -> List[dict]:
    proposals = load_proposals(path)
    for prop in proposals:
        prop['objn'] = 1.0 - prop['objn']
        prop['score'] = 1.0 - prop['score']
    return proposals

def gt(path: str) -> List[dict]:
    max_dets = [1, 10, 100, 1000]

    from spiders.coco_ssm_spider import COCOSSMDemoSpider
    spider = COCOSSMDemoSpider()
    cocoGt = spider.dataset

    cocoDt = cocoGt.loadRes(path)  # path to results
    cocoEval = COCOeval(cocoGt, cocoDt)

    cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
    cocoEval.params.maxDets = max_dets
    cocoEval.params.useSegm = True
    cocoEval.params.useCats = False
    cocoEval.params.iouThrs = [0.5]
    cocoEval.evaluate()

    counter = 0
    resultID_2_iou = {}
    for imgId, foo in cocoEval.ious.keys():
        counter += 1
        if cocoEval.ious[(imgId, foo)] != []:
            cocoEval.ious[(imgId, foo)] = cocoEval.ious[(imgId, foo)][:, :]
            for resultID in range(cocoEval.ious[(imgId, foo)].shape[0]):
                iou = np.max(cocoEval.ious[(imgId, foo)][resultID, :])
                detID = cocoEval._dtIDs[(imgId, -1)][resultID]  # ["resultID"]
                resultID_2_iou[detID] = iou

    proposals = load_proposals(path)
    for prop in proposals:
        prop['objn'] = resultID_2_iou[prop['resultID']]
        prop['score'] = resultID_2_iou[prop['resultID']]
    return proposals


def metamain():
    path1, path2, algo = parse_args()
    if algo == "inv":
        new_proposals = inv(path1)
    else:
        new_proposals = gt(path1)
    with open(path2, "w") as file:
        json.dump(list(new_proposals), file)

    print("metamain!")


def parse_args() -> Tuple[str, str, str]:
    parser = argparse.ArgumentParser(description="Objectnessscorings meta")
    parser.add_argument("-o", "--output_path",
                        required=True)
    parser.add_argument("-i", "--input_path",
                        required=True)
    parser.add_argument("-a", "--algo",
                        required=True)

    argument = parser.parse_args()
    assert(argument.algo == "inv" or argument.algo == "gt")
    return argument.input_path, argument.output_path, argument.algo


if __name__ == '__main__':
    metamain()
