import argparse
import json
import copy
import math
from sklearn.svm import SVR

from attentionmask.mask import decode
from sklearn import svm

import numpy as np

from typing import Tuple, List, Optional



def load_proposals(proposals_path: str) -> List[dict]:
    with open(proposals_path) as file:
        proposals = json.load(file)
    return proposals


def metamain():
    path1, path2 = parse_args()
    proposals = load_proposals(path1)
    for prop in proposals:
        prop['objn'] = 1.0 - prop['objn']
        prop['score'] = 1.0 - prop['score']

    with open(path2, "w") as file:
        json.dump(list(proposals), file)

    print("metamain!")


def parse_args() -> Tuple[str, str]:
    parser = argparse.ArgumentParser(description="Objectnessscorings meta")
    parser.add_argument("-o", "--output_path",
                        required=True)
    parser.add_argument("-i", "--input_path",
                        required=True)

    argument = parser.parse_args()
    return argument.input_path, argument.output_path


if __name__ == '__main__':
    metamain()
