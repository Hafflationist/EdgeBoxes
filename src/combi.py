import argparse
import json
import copy
import math
from sklearn.svm import SVR
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from joblib import dump, load

from typing import Tuple, List, Optional, Dict, Any, Callable


def load_proposals(proposals_path: str) -> List[dict]:
    with open(proposals_path) as file:
        proposals = json.load(file)
    return proposals


def extract_scores(proposals: List[dict]) -> List[float]:
    return list(map(lambda p: p['objn'], proposals))



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

    X: List[List[float]] = [list(i) for i in zip(*scores_list)]  # transposing list of lists
    return  X, proposals
