import argparse
import json
import copy
import math
from sklearn.svm import SVR

from sklearn import svm

from typing import Tuple, List, Optional, Dict


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
        prediction = svr.predict([list(tup[1:])])[0]
        proposal['objn'] = prediction
        proposal['score'] = prediction

    return proposals_copy


def load_proposals(proposals_path: str) -> List[dict]:
    with open(proposals_path) as file:
        proposals = json.load(file)
    return proposals


def extract_scores(proposals: List[dict]) -> List[float]:
    return list(map(lambda p: p['objn'], proposals))


def create_ordered_gt_results(proposals: List[dict], path_gt: str) -> List[float]:
    proposals_with_gt = load_proposals(path_gt)
    gt_dict: Dict[int, float] = {p['resultID']: p['score'] for p in proposals_with_gt}
    return [gt_dict[p['resultID']] for p in proposals]


def metamain():
    path_gt, \
    path_am, \
    path_cc, \
    path_eb, \
    path_ms, \
    path_ss, \
    learning_method, \
    output_path, \
    algorithms = parse_args()

    scores_am: Optional[List[float]]= None
    scores_cc: Optional[List[float]]= None
    scores_eb: Optional[List[float]]= None
    scores_ms: Optional[List[float]]= None
    scores_ss: Optional[List[float]]= None
    scores_list: List[List[float]] = []
    proposals: List[dict] = []

    if "am" in algorithms: # dirty implicit conversion to true/false
        proposals = load_proposals(path_am)
        scores_am = extract_scores(proposals)
        scores_list.append(scores_am)
        if any(math.isnan(s) for s in scores_am):
            print("AM-scores contain NAN!")
        if any(math.isinf(s) for s in scores_am):
            print("AM-scores contain INF!")

    if "cc" in algorithms: # dirty implicit conversion to true/false
        proposals = load_proposals(path_cc)
        scores_cc = extract_scores(proposals)
        scores_list.append(scores_cc)
        if any(math.isnan(s) for s in scores_cc):
            print("CC-scores contain NAN!")
        if any(math.isinf(s) for s in scores_cc):
            print("CC-scores contain INF!")

    if "eb" in algorithms: # dirty implicit conversion to true/false
        proposals = load_proposals(path_eb)
        scores_eb = extract_scores(proposals)
        scores_list.append(scores_eb)
        if any(math.isnan(s) for s in scores_eb):
            print("EB-scores contain NAN!")
        if any(math.isinf(s) for s in scores_eb):
            print("EB-scores contain INF!")

    if "ms" in algorithms: # dirty implicit conversion to true/false
        proposals = load_proposals(path_ms)
        scores_ms = extract_scores(proposals)
        scores_list.append(scores_ms)
        if any(math.isnan(s) for s in scores_ms):
            print("MS-scores contain NAN!")
        if any(math.isinf(s) for s in scores_ms):
            print("MS-scores contain INF!")

    if "ss" in algorithms: # dirty implicit conversion to true/false
        proposals = load_proposals(path_ss)
        scores_ss = extract_scores(proposals)
        scores_list.append(scores_ss)
        if any(math.isnan(s) for s in scores_ss):
            print("SS-scores contain NAN!")
        if any(math.isinf(s) for s in scores_ss):
            print("SS-scores contain INF!")

    X: List[List[float]] = [list(i) for i in zip(*scores_list)]   # transposing list of lists
    y = create_ordered_gt_results(proposals, path_gt)

    svr = svm.SVR()
    svr.fit(X, y)

    new_proposals = calc_regressand(proposals, scores_am, scores_cc, scores_eb, scores_ms, scores_ss, svr)

    with open(output_path + "/attentionMask-8-128.json.combi." + algorithms + ".json", "w") as file:
        json.dump(list(new_proposals), file)

    print("metamain!")


def parse_args() -> Tuple[str, str, str, str, str, str, str, str, str]:
    parser = argparse.ArgumentParser(description="Objectnessscorings meta")
    parser.add_argument("-g", "--gt",
                        help="Path to GT data",
                        required=True)
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
                        help="Path of output file without filename",
                        required=True)
    parser.add_argument("-A", "--algorithms",
                        help="Example: \"am.cc.eb.ms.ss\"; This string will be also used as file name suffix",
                        required=True)

    argument = parser.parse_args()
    assert argument.gt
    assert(argument.am or argument.cc or argument.eb or argument.ms or argument.ss)
    assert argument.output_path
    return argument.gt, \
           argument.am, \
           argument.cc, \
           argument.eb, \
           argument.ms, \
           argument.ss, \
           argument.learning_method, \
           argument.output_path, \
           argument.algorithms


if __name__ == '__main__':
    metamain()