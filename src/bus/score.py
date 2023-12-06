import argparse
import json
import logging
import subprocess
from pathlib import Path

import pandas as pd
import pyhocon

from bus import compute_bus

logger = logging.getLogger(__name__)


def load_data(file_path: Path):
    df = pd.read_csv(file_path, sep="\t", na_filter=False)
    return df


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


def main(config, bus_config):
    # get events from the split
    split = "dev" if config["split"] == "validation" else config["split"]
    split_path = f"data/splits/{split}.txt"

    with open(split_path, "r") as rf:
        events = [line.strip() for line in rf.readlines()]

    logger.info(f"split: {split}")
    logger.info(f"events: {events}")

    generator_name = bus_config["model_kwargs"]["model_name_or_path"].replace("/", "_")

    gold_path = Path("data/events")
    bus_pred_path = None
    if config.get("model", None) == "gpt-3.5-turbo":
        pred_path = config["output_path"] / "preds"
        bus_pred_path = config["output_path"] / f"bus-{generator_name}-{split}.jsonl"
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
        bus_pred_path = (
            Path(config["model_name_or_path"]) / f"bus-{generator_name}-{split}.jsonl"
        )

    logger.info(f"gold: {gold_path}")
    logger.info(f"pred: {pred_path}")
    logger.info(f"bus pred path: {bus_pred_path}")

    z, pred = [], []
    for event in events:
        anns = [f"annotator{idx}" for idx in range(1, 4)]
        for ann in anns:
            # gold_df = load_data(gold_path / event / f"{ann}.tsv")
            pred_df = load_data(pred_path / event / f"{ann}.tsv")

            z += list(pred_df["Update"][1:])
            pred += list(pred_df["Background"][1:])

    assert len(z) == len(pred), f"z: {len(z)}, pred: {len(pred)}"

    logger.info(f"z: {len(z)}, pred: {len(pred)}")

    output = compute_bus(
        backgrounds=pred, updates=z, model_kwargs=bus_config["model_kwargs"]
    )
    logger.info(f"bus: {output['bus']}")
    write_qa_pairs(
        file_path=bus_pred_path,
        questions=output["questions"],
        answers=output["answers"],
        updates=z,
        backgrounds=pred,
    )


def init_config(config_path: Path, config_name: str):
    config = pyhocon.ConfigFactory.parse_file(config_path)[config_name]

    for x in ["output_path", "log_path"]:
        config[x] = Path(config[x])
        config[x] /= f"{config_name}"

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(
            config["log_path"] / "log_bus_scores.txt",
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


def init_bus_config(config_path: Path, config_name: str):
    config = pyhocon.ConfigFactory.parse_file(config_path)[config_name]
    logger.info("BUS config")
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="compute BUS")
    parser.add_argument("--config", type=Path, help="config file")
    parser.add_argument("--config-name", type=str, help="config name")
    parser.add_argument(
        "--bus-config", type=Path, default="configs/bus.conf", help="BUS config file"
    )
    parser.add_argument(
        "--bus-config-name", type=str, default="gpt-3.5-turbo", help="BUS config name"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = init_config(args.config, args.config_name)
    bus_config = init_bus_config(args.bus_config, args.bus_config_name)
    main(config, bus_config)
