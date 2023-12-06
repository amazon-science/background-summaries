import argparse
import logging
from pathlib import Path

import nltk
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def count_tokens(df):
    update = [len(nltk.word_tokenize(_item)) for _item in df["Update"]]
    background = [len(nltk.word_tokenize(_item)) for _item in df["Background"][1:]]
    return update, background


def main(args):
    for event_path in args.events.iterdir():
        if not event_path.is_dir():
            continue
        event = event_path.stem
        update, background = [], []
        for ann_path in event_path.iterdir():
            df = pd.read_csv(ann_path, sep="\t", na_filter=False)
            ann_update, ann_background = count_tokens(df)
            update += ann_update
            background += ann_background
        logger.info(f"{event}\t{np.mean(update):.0f}\t{np.mean(background):.0f}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="get basic stats about the dataset")
    parser.add_argument("events", type=Path)
    args = parser.parse_args()
    main(args)
