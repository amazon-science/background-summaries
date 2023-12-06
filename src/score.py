"""
compute metrics on model predictions
"""
import argparse
import logging
import subprocess
from pathlib import Path

import pandas as pd
import pyhocon

from evaluate_summ import compute_summ_metrics

logger = logging.getLogger(__name__)


def load_data(file_path: Path):
    df = pd.read_csv(file_path, sep="\t", na_filter=False)
    return df


def main(config, no_ref: bool = False):
    # get events from the split
    split = "dev" if config["split"] == "validation" else config["split"]
    split_path = f"data/splits/{split}.txt"

    with open(split_path, "r") as rf:
        events = [line.strip() for line in rf.readlines()]

    logger.info(f"split: {split}")
    logger.info(f"events: {events}")

    gold_path = Path("data/events")
    if config.get("model", None) == "gpt-3.5-turbo":
        pred_path = config["output_path"] / "preds"
    else:
        # convert JSON-L file to per annotator tsv files
        # necessary for HF based models
        pred_jsonl_path = (
            Path(config["model_name_or_path"]) / f"{config['split']}_preds.jsonl"
        )
        pred_path = Path(config["model_name_or_path"]) / "preds"
        subprocess.run(
            [
                "python",
                "src/reconstruct_ann_pred.py",
                "--preds",
                str(pred_jsonl_path),
                "--data",
                "data",
                "--split",
                split,
                "--out",
                str(pred_path),
            ],
            check=True,
            capture_output=True,
        )

    logger.info(f"gold: {gold_path}")
    logger.info(f"pred: {pred_path}")

    src, tgt, pred = [], [], []
    for event in events:
        anns = [f"annotator{idx}" for idx in range(1, 4)]
        for ann in anns:
            gold_df = load_data(gold_path / event / f"{ann}.tsv")
            pred_df = load_data(pred_path / event / f"{ann}.tsv")

            # curate source
            updates = [
                f"Date: {gold_df['Date'][idx]}, Article: {gold_df['Update'][idx]}"
                for idx in range(len(gold_df["Update"]))
            ]
            src += [" ".join(updates[:idy]) for idy in range(1, len(gold_df["Update"]))]

            # tgt
            tgt += list(gold_df["Background"][1:])

            # pred
            pred += list(pred_df["Background"][1:])

    assert (
        len(src) == len(tgt) == len(pred)
    ), f"src: {len(src)}, tgt: {len(tgt)}, pred: {len(pred)}"

    logger.info(f"src: {len(src)}, tgt: {len(tgt)}, pred: {len(pred)}")

    if no_ref:
        tgt = None
        logger.info("running in reference-free mode")

    scores = compute_summ_metrics(src=src, tgt=tgt, pred=pred)

    logger.info(scores)


def init_config(config_path: Path, config_name: str):
    config = pyhocon.ConfigFactory.parse_file(config_path)[config_name]

    for x in ["output_path", "log_path"]:
        config[x] = Path(config[x])
        config[x] /= f"{config_name}"

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(
            config["log_path"] / "log_summ_scores.txt",
            mode="w",
        ),
    ]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )

    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    return config


def parse_args():
    parser = argparse.ArgumentParser(description="score predictions")
    parser.add_argument("--config", type=Path, help="config path")
    parser.add_argument("--config-name", type=str, help="config name")
    parser.add_argument(
        "--no-ref", action="store_true", help="only run reference-free metrics"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = init_config(args.config, args.config_name)
    main(config, no_ref=args.no_ref)
