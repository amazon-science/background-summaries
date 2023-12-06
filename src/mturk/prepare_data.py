import argparse
import json
import logging
import random
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pyhocon

logger = logging.getLogger(__name__)


def load_data(file_path: Path):
    df = pd.read_csv(file_path, sep="\t", na_filter=False)
    return df


def clean_txt(txt: str):
    txt = txt.replace("\\n", " ")
    txt = txt.replace("\n", " ")
    txt = re.sub(r"[ ]+", r" ", txt)
    return txt


def main(sys_configs: dict, gold_path: Path, out_path: Path, splits: list[str]):
    for split in splits:
        split_path = gold_path / "splits" / f"{split}.txt"
        with open(split_path, "r") as rf:
            events = [line.strip() for line in rf.readlines()]
        anns = ["annotator1", "annotator2", "annotator3"]
        logger.info(f"split: {split}")
        logger.info(f"events: {events}")
        logger.info(f"anns: {anns}")

        # load system backgrounds
        sys2dat = {}
        for sys_name, sys_config in sys_configs.items():
            # predictions path
            if sys_config.get("model", None) == "gpt-3.5-turbo":
                preds_path = sys_config["output_path"] / "preds"
            else:
                preds_path = Path(sys_config["model_name_or_path"]) / "preds"

            sys2dat[sys_name] = {}
            for evt in events:
                sys2dat[sys_name][evt] = {}
                for ann in anns:
                    sys2dat[sys_name][evt][ann] = {}
                    ann_path = preds_path / evt / f"{ann}.tsv"
                    df = load_data(ann_path)
                    for idx, row in enumerate(df.itertuples()):
                        if idx > 1:
                            sys2dat[sys_name][evt][ann][row.Date] = clean_txt(
                                row.Background
                            )

        # load human backgrounds
        sys2dat["human"] = {}
        sys2inp = {}
        for evt in events:
            sys2dat["human"][evt] = {}
            sys2inp[evt] = {}
            for ann in anns:
                sys2dat["human"][evt][ann] = {}
                sys2inp[evt][ann] = OrderedDict()
                ann_path = gold_path / "events" / evt / f"{ann}.tsv"
                df = load_data(ann_path)
                for idx, row in enumerate(df.itertuples()):
                    if idx > 1:
                        sys2dat["human"][evt][ann][row.Date.strip("[]")] = clean_txt(
                            row.Background
                        )
                        sys2inp[evt][ann][row.Date.strip("[]")] = clean_txt(row.Update)

        out_data = []
        for evt in events:
            for ann in anns:
                for ts in sys2inp[evt][ann]:
                    tuples = []
                    for sys_name in sys2dat:
                        name_str = f"{evt}_{ann}_{ts}_{sys_name}"
                        tuples += [(name_str, sys2dat[sys_name][evt][ann][ts])]
                    # random shuffle of tuples
                    random.shuffle(tuples)
                    tuple_dict = {}
                    for idx, _item in enumerate(tuples):
                        tuple_dict.update(
                            {
                                f"summary{idx+1}_name": _item[0],
                                f"summary{idx+1}": _item[1],
                            }
                        )
                    tuple_dict.update(
                        {"update": f"Date: {ts}, Article: {sys2inp[evt][ann][ts]}"}
                    )
                    out_data += [tuple_dict]

        with open(out_path / f"{split}.jsonl", "w") as wf:
            for _item in out_data:
                wf.write(json.dumps(_item) + "\n")


def init_config(config_path: Path, config_name: str):
    config = pyhocon.ConfigFactory.parse_file(config_path)[config_name]

    for x in ["output_path"]:
        config[x] = Path(config[x])
        config[x] /= f"{config_name}"

    handlers = [
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )

    return config


def parse_args():
    parser = argparse.ArgumentParser(description="prepare data for A/B tests")
    parser.add_argument("--config", type=Path, nargs="+", help="config file")
    parser.add_argument("--config-name", type=str, nargs="+", help="config name")
    parser.add_argument("--gold", type=Path, default="data", help="path to gold data")
    parser.add_argument("--splits", type=str, nargs="+", help="split (dev or test)")
    parser.add_argument("--out", type=Path, help="dir path to write output jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # load system configs
    sys_configs = {}
    for sys_config_path, sys_config_name in zip(args.config, args.config_name):
        sys_configs[sys_config_name] = init_config(sys_config_path, sys_config_name)
    args.out.mkdir(exist_ok=True, parents=True)
    main(
        sys_configs=sys_configs,
        gold_path=args.gold,
        out_path=args.out,
        splits=args.splits,
    )
