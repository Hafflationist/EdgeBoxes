import argparse
import json


def write_new_file(proposals_path: str) -> None:
    with open(proposals_path) as file:
        data = json.load(file)
    i = 0
    for proposal in data:
        proposal['resultID'] = i
        i = i + 1
    with open(proposals_path + "_", "w") as file:
        json.dump(data, file)


def parse_args() -> str:
    parser = argparse.ArgumentParser(description="Objectnessscoring")
    parser.add_argument("-p", "--proposals", help="Path of file with proposals", required=True, default="")

    argument = parser.parse_args()

    return argument.proposals


def main() -> None:
    proposals_path = parse_args()
    write_new_file(proposals_path)


if __name__ == '__main__':
    main()