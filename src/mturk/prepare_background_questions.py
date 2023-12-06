import argparse
import json
import logging
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(file_path: Path):
    df = pd.read_csv(file_path, sep="\t", na_filter=False)
    return df


def clean_txt(txt: str):
    txt = txt.replace("\\n", " ")
    txt = txt.replace("\n", " ")
    txt = re.sub(r"[ ]+", r" ", txt)
    return txt


def main(gold_path: Path, out_path: Path, splits: list[str]):
    for split in splits:
        split_path = gold_path / "splits" / f"{split}.txt"
        with open(split_path, "r") as rf:
            events = [line.strip() for line in rf.readlines()]
        anns = ["annotator1", "annotator2", "annotator3"]
        logger.info(f"split: {split}")
        logger.info(f"events: {events}")
        logger.info(f"anns: {anns}")

        # load updates
        sys2inp = {}
        for evt in events:
            sys2inp[evt] = {}
            for ann in anns:
                sys2inp[evt][ann] = OrderedDict()
                ann_path = gold_path / "events" / evt / f"{ann}.tsv"
                df = load_data(ann_path)
                for idx, row in enumerate(df.itertuples()):
                    if idx > 1:
                        sys2inp[evt][ann][row.Date.strip("[]")] = clean_txt(row.Update)

        out_data = []
        for evt in events:
            for ann in anns:
                for ts in sys2inp[evt][ann]:
                    tuple_dict = {
                        "update": f"Date: {ts}, {sys2inp[evt][ann][ts]}",
                        "update_name": f"{evt}_{ann}_{ts}",
                    }
                    out_data += [tuple_dict]

        with open(out_path / f"{split}.jsonl", "w") as wf:
            for _item in out_data:
                wf.write(json.dumps(_item) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="prepare data for background questions"
    )
    parser.add_argument("--gold", type=Path, default="data", help="path to gold data")
    parser.add_argument("--splits", type=str, nargs="+", help="split (dev or test)")
    parser.add_argument("--out", type=Path, help="dir path to write output jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    main(
        gold_path=args.gold,
        out_path=args.out,
        splits=args.splits,
    )
