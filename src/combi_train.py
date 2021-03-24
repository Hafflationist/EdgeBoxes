import argparse
import cv2
from sklearn import svm

from joblib import dump
from sklearn.ensemble import RandomForestRegressor

from typing import Tuple, List, Optional, Dict

from combi import get_X_and_proposals, load_proposals, extract_algos


def create_ordered_gt_results(proposals: List[dict], path_gt: Optional[str]) -> List[float]:
    if path_gt is None: return []
    proposals_with_gt = load_proposals(path_gt)
    gt_dict: Dict[int, float] = {p['resultID']: p['score'] for p in proposals_with_gt}
    return [gt_dict[p['resultID']] for p in proposals]


def train(path_gt: str,
          path_am: str,
          path_cc: str,
          path_eb: str,
          path_ms: str,
          path_ss: str,
          learning_method: str,
          model_output_path: str,
          algorithms: str):

    X, proposals = get_X_and_proposals(path_am, path_cc, path_eb, path_ms,path_ss, algorithms)
    y = create_ordered_gt_results(proposals, path_gt)

    if "svm" in learning_method:
        svr = svm.SVR()
        svr.fit(X, y)
        dump(svr, model_output_path + "/svm.combi." + algorithms + ".gurke")
        print("Model written into " + model_output_path + "/svm.combi." + algorithms + ".gurke")

    if "trees" in learning_method:
        trees = RandomForestRegressor(max_depth=10, # performance?
                                       random_state=88, # determinism
                                       min_samples_leaf=1, # erhöhen für weichere Ergebnisse
                                       n_jobs=-1, # using 2 processors
                                       bootstrap=True,
                                       max_samples=None # could be reduced, default is 100%
                                       )
        trees.fit(X, y)
        dump(trees, model_output_path + "/trees.combi." + extract_algos(algorithms) + ".gurke")
        print("Model written into " + model_output_path + "/trees.combi." + extract_algos(algorithms) + ".gurke")


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
                        help="Path of output file (including model data) without filename",
                        required=True)
    parser.add_argument("-A", "--algorithms",
                        help="Example: \"am.cc.eb.ms.ss.ra\"; This string will be also used as file name suffix",
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
    path_gt, \
    path_am, \
    path_cc, \
    path_eb, \
    path_ms, \
    path_ss, \
    learning_method, \
    model_output_path, \
    algorithms = parse_args()

    train(path_gt, path_am, path_cc, path_eb, path_ms, path_ss, learning_method, model_output_path, algorithms)