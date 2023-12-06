import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyhocon

from bus import compute_bus
from evaluate_summ import compute_summ_metrics

logger = logging.getLogger(__name__)


def write_qa_pairs(
    file_path: Path,
    questions: list[list[str]],
    answers: list[list[str]],
    updates: list[str],
    backgrounds: list[str],
):
    with open(file_path, "w") as wf:
        for idx in range(len(updates)):
            qa_pairs = [
                {"Question": q, "Answer": a}
                for q, a in zip(questions[idx], answers[idx])
            ]
            out = {
                "Update": updates[idx],
                "Background": backgrounds[idx],
                "QA": qa_pairs,
            }
            wf.write(json.dumps(out) + "\n")


def main(args):
    bus_config = init_bus_config(args.bus_config, args.bus_config_name)
    with open(args.data / "splits" / f"{args.split}.txt", "r") as rf:
        events = [line.strip() for line in rf.readlines()]

    logger.info(f"events: {events}")
    annotators = ["annotator1", "annotator2", "annotator3"]
    logger.info(f"annotators: {annotators}")
    ann2pred = defaultdict(list)
    ann2src = defaultdict(list)
    ann2z = defaultdict(list)
    for event in events:
        event_path = args.data / "events" / event
        for ann_path in event_path.iterdir():
            ann = ann_path.stem
            df = pd.read_csv(ann_path, sep="\t", na_filter=False)
            # source
            updates = [
                f"Date: {df['Date'][idx]}, Article: {df['Update'][idx]}"
                for idx in range(len(df["Update"]))
            ]
            ann2src[ann] += [
                " ".join(updates[:idy]) for idy in range(1, len(df["Update"]))
            ]
            # pred
            ann2pred[ann] += list(df["Background"][1:])
            # z
            ann2z[ann] += list(df["Update"][1:])

    scores = defaultdict(list)
    args.out.mkdir(exist_ok=True, parents=True)
    for ann in annotators:
        generator_name = bus_config["model_kwargs"]["model_name_or_path"].replace(
            "/", "_"
        )
        # compute bus scores
        ann_bus = compute_bus(
            backgrounds=ann2pred[ann],
            updates=ann2z[ann],
            model_kwargs=bus_config["model_kwargs"],
        )
        scores["bus"] += [ann_bus["bus"]]
        write_qa_pairs(
            file_path=args.out / f"{ann}_{args.split}_{generator_name}.jsonl",
            questions=ann_bus["questions"],
            answers=ann_bus["answers"],
            updates=ann2z[ann],
            backgrounds=ann2pred[ann],
        )
        ann_scores = compute_summ_metrics(pred=ann2pred[ann], src=ann2src[ann])
        for k, v in ann_scores.items():
            scores[k] += [v]

        logger.info(scores)

    logger.info(scores)


def init_bus_config(config_path: Path, config_name: str):
    config = pyhocon.ConfigFactory.parse_file(config_path)[config_name]
    logger.info("BUS config")
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="get basic stats about the dataset")
    parser.add_argument("data", type=Path, help="path to annotations")
    parser.add_argument("split", type=str, help="split name")
    parser.add_argument("out", type=Path, help="output dir path")
    parser.add_argument(
        "--bus-config", type=Path, default="configs/bus.conf", help="bus config"
    )
    parser.add_argument(
        "--bus-config-name", type=str, default="gpt-3.5-turbo", help="bus config name"
    )
    args = parser.parse_args()
    main(args)
