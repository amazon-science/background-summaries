import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from evaluate_summ import compute_summ_metrics

logger = logging.getLogger(__name__)


def get_scores(ann2txt: dict, annotators: list):
    scores = {}
    for ann in annotators:
        pred = ann2txt[ann]
        ref_anns = [_item for _item in annotators if _item != ann]
        tgt = [ann2txt[ref_ann] for ref_ann in ref_anns]
        tgt = [list(_item) for _item in zip(*tgt)]
        ann_scores = compute_summ_metrics(pred=pred, tgt=tgt)
        scores[ann] = ann_scores
    return scores


def main(args):
    annotators = ["annotator1", "annotator2", "annotator3"]
    logger.info(f"annotators: {annotators}")
    ann2update = defaultdict(list)
    ann2background = defaultdict(list)
    events = []
    for event_path in args.events.iterdir():
        if not event_path.is_dir():
            continue
        events += [event_path.stem]
        for ann_path in event_path.iterdir():
            ann = ann_path.stem
            df = pd.read_csv(ann_path, sep="\t", na_filter=False)
            ann2update[ann] += list(df["Update"])
            ann2background[ann] += list(df["Background"][1:])

    logger.info(f"events: {events}")
    logger.info("update IAA")
    scores = get_scores(ann2txt=ann2update, annotators=annotators)
    logger.info(scores)
    logger.info("background IAA")
    scores = get_scores(ann2txt=ann2background, annotators=annotators)
    logger.info(scores)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="compute IAA")
    parser.add_argument(
        "--events", type=Path, default="data/events", help="path to annotations"
    )
    args = parser.parse_args()
    main(args)
