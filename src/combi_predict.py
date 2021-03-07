import argparse
import json
import copy
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier

from joblib import load

from typing import Tuple, List, Callable

from combi import get_X_and_proposals


def calc_regressand(proposals: List[dict],
                    X: List[List[float]],
                    predictor: Callable[[List[List[float]]], float]) -> List[dict]:
    proposals_copy = copy.deepcopy(proposals)

    for tup in zip(proposals_copy, X):
        proposal = tup[0]
        vector = tup[1]
        prediction = predictor([vector])
        proposal['objn'] = prediction
        proposal['score'] = prediction

    return proposals_copy


def extract_algos(path: str):
    algo_string = ""
    if "am." in path:
        algo_string += "am"
    if ".cc." in path:
        algo_string += ".cc"
    if ".eb." in path:
        algo_string += ".eb"
    if ".ms." in path:
        algo_string += ".ms"
    if ".ss" in path:
        algo_string += ".ss"
    algo_string += "."
    return algo_string


def predict(path_am: str,
            path_cc: str,
            path_eb: str,
            path_ms: str,
            path_ss: str,
            model_input_path: str,
            output_path: str):
    algorithms = extract_algos(model_input_path)
    X, proposals = get_X_and_proposals(path_am, path_cc, path_eb, path_ms,path_ss, algorithms)

    predictor = lambda _: -1.0
    if "trees.combi" in output_path:
        trees: RandomForestClassifier = load(model_input_path)
        predictor = lambda data: trees.predict(data)[0]
    elif "svm.combi" in output_path:
        svr: SVR = load(model_input_path)
        predictor = lambda data: svr.predict(data)[0]

    new_proposals = calc_regressand(proposals, X, predictor)

    with open(output_path + "/attentionMask-8-128.json.combi." + algorithms + ".json", "w") as file:
        json.dump(list(new_proposals), file)
    print("Prediction written into " + output_path + "/attentionMask-8-128.json.combi." + algorithms + ".json")


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
    parser.add_argument("-i", "--model_input_path",
                        help="Path of file containing the learned model with filename",
                        required=True)
    parser.add_argument("-o", "--output_path",
                        help="Path of output file without filename",
                        required=True)

    argument = parser.parse_args()
    assert(argument.am or argument.cc or argument.eb or argument.ms or argument.ss)
    assert argument.model_input_path
    assert argument.output_path
    return argument.am, \
           argument.cc, \
           argument.eb, \
           argument.ms, \
           argument.ss, \
           argument.model_input_path, \
           argument.output_path


if __name__ == '__main__':
    p_am, \
    p_cc, \
    p_eb, \
    p_ms, \
    p_ss, \
    model_output_p, \
    output_p = parse_args()

    predict(p_am, p_cc, p_eb, p_ms, p_ss, model_output_p, output_p)