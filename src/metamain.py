import argparse
import json
import copy
from sklearn.svm import SVR

from attentionmask.mask import decode
from sklearn import svm

import numpy as np

from typing import Tuple, List, Optional

def calc_ground_truth(proposals: List[dict]) -> List[float]:
    proposals_copy = copy.deepcopy(proposals)
    for proposal in proposals_copy:
        # TODO: Calculate real ground truth from SCRIPT
        proposal['objn'] = 0.0
        proposal['score'] = 0.0
    return list(map(lambda p: 0.0, proposals))

def calc_regressand(proposals: List[dict],
                    scores_am: Optional[List[float]],
                    scores_cc: Optional[List[float]],
                    scores_eb: Optional[List[float]],
                    scores_ms: Optional[List[float]],
                    scores_ss: Optional[List[float]],
                    svr: SVR) -> List[dict]:
    proposals_copy = copy.deepcopy(proposals)
    scores_list = [proposals_copy]
    if scores_am is not None:
        scores_list.append(scores_am)

    if scores_cc is not None:
        scores_list.append(scores_cc)

    if scores_eb is not None:
        scores_list.append(scores_eb)

    if scores_ms is not None:
        scores_list.append(scores_ms)

    if scores_ss is not None:
        scores_list.append(scores_ss)

    for tup in zip(*scores_list):
        proposal = tup[0]
        prediction = svr.predict(list(tup[1:]))
        proposal['objn'] = prediction
        proposal['score'] = prediction

    return proposals_copy


def load_proposals(proposals_path: str) -> List[dict]:
    with open(proposals_path) as file:
        proposals = json.load(file)
    return proposals


def extract_scores(proposals: List[dict]) -> List[float]:
    return list(map(lambda p: p['objn'], proposals))


def metamain():
    path_am, \
    path_cc, \
    path_eb, \
    path_ms, \
    path_ss, \
    output_path, \
    learning_method = parse_args()

    scores_am: Optional[List[float]]= None
    scores_cc: Optional[List[float]]= None
    scores_eb: Optional[List[float]]= None
    scores_ms: Optional[List[float]]= None
    scores_ss: Optional[List[float]]= None
    scores_list: List[List[float]] = []
    proposals: List[dict] = []

    if path_am: # dirty implicit conversion to true/false
        proposals = load_proposals(path_am)
        scores_am = extract_scores(proposals)
        scores_list.append(scores_am)

    if path_cc: # dirty implicit conversion to true/false
        proposals = load_proposals(path_cc)
        scores_cc = extract_scores(proposals)
        scores_list.append(scores_cc)

    if path_eb: # dirty implicit conversion to true/false
        proposals = load_proposals(path_eb)
        scores_eb = extract_scores(proposals)
        scores_list.append(scores_eb)

    if path_ms: # dirty implicit conversion to true/false
        proposals = load_proposals(path_ms)
        scores_ms = extract_scores(proposals)
        scores_list.append(scores_ms)

    if path_ss: # dirty implicit conversion to true/false
        proposals = load_proposals(path_ss)
        scores_ss = extract_scores(proposals)
        scores_list.append(scores_ss)

    X = [list(i) for i in zip(*scores_list)]   # transposing list of lists
    y = calc_ground_truth(proposals)

    svr = svm.SVR()
    svr.fit(X, y)

    new_proposals = calc_regressand(proposals, scores_am, scores_cc, scores_eb, scores_ms, scores_ss, svr)

    with open(output_path, "w") as file:
        json.dump(new_proposals, file)

    print("metamain!")


def parse_args() -> Tuple[str, str, str, str, str, str, str]:
    parser = argparse.ArgumentParser(description="Objectnessscorings meta")
    parser.add_argument("-a", "--am",
                        help="Path to AM results",
                        required=True)
    parser.add_argument("-c", "--cc",
                        help="Path to CC results",
                        required=True)
    parser.add_argument("-e", "--eb",
                        help="Path to EB results",
                        required=True)
    parser.add_argument("-m", "--ms",
                        help="Path to MS results",
                        required=True)
    parser.add_argument("-s", "--ss",
                        help="Path to SS results",
                        required=True)
    parser.add_argument("-L", "--learning_method",
                        help="ML method (\"trees\" or \"svm\")",
                        required=True)
    parser.add_argument("-o", "--output_path",
                        help="Path of output file with filename",
                        required=True)

    argument = parser.parse_args()
    assert(argument.am or argument.cc or argument.eb or argument.ms or argument.ss)
    assert argument.output_path
    return argument.am, \
           argument.cc, \
           argument.eb, \
           argument.ms, \
           argument.ss, \
           argument.output_path, \
           argument.learning_method


if __name__ == '__main__':
    metamain()