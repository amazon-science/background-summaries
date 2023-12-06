"""
HF dataset loading script
"""

import re
from pathlib import Path

import datasets
import pandas as pd

_DESCRIPTION = """Update-background tuples for 14 news event timelines."""

_URLS = {
    "events": "events.tar.gz",
    "train": "splits/train.txt",
    "dev": "splits/dev.txt",
    "test": "splits/test.txt",
}

_CITATION = """\
@article{pratapa-etal-2023-background,
title = {Background Summarization of Event Timelines},
author = {Pratapa, Adithya and Small, Kevin and Dreyer, Markus},
publisher = {EMNLP},
year = {2023}
}
"""
_HOMEPAGE = ""
_LICENSE = ""


class BackgroundSummConfig(datasets.BuilderConfig):
    def __init__(self, features, **kwargs) -> None:
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.features = features


class BackgroundSumm(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        BackgroundSummConfig(
            name="background-summ",
            description=_DESCRIPTION,
            features=["src", "tgt", "z"],
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {field: datasets.Value("string") for field in ["src", "tgt", "z"]}
            ),
        )

    def _split_generators(self, dl_manager):
        dl_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "events_path": Path(dl_files["events"]),
                    "splits_path": Path(dl_files["train"]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "events_path": Path(dl_files["events"]),
                    "splits_path": Path(dl_files["dev"]),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "events_path": Path(dl_files["events"]),
                    "splits_path": Path(dl_files["test"]),
                },
            ),
        ]

    def _generate_examples(self, events_path: Path, splits_path: Path):
        # load events for the split
        with open(splits_path, "r") as rf:
            event_names = [line.strip() for line in rf.readlines()]

        data_idx = 0
        for event in event_names:
            # separately load update and background summaries for each annotator
            annotators = ["annotator1", "annotator2", "annotator3"]
            for ann in annotators:
                # load tsv path
                tsv_path = events_path / "events" / event / f"{ann}.tsv"
                df = pd.read_csv(tsv_path, sep="\t")
                df = df.fillna("")
                timestamps, updates, backgrounds = [], [], []
                for idx, row in enumerate(df.itertuples()):
                    ts = row.Date.strip("[]")
                    update = row.Update.replace("\\n", " ")
                    update = re.sub(r"[ ]+", r" ", update).strip()
                    background = row.Background.replace("\\n", " ")
                    background = re.sub(r"[ ]+", r" ", background).strip()

                    timestamps += [ts]
                    updates += [update]
                    backgrounds += [background]

                    # source is a timestamped concatenation of past updates
                    src = [
                        f"Date: {_ts}, Update: {_update}"
                        for _ts, _update in zip(timestamps[:-1], updates[:-1])
                    ]
                    src = " ".join(src)
                    # target is current background
                    tgt = backgrounds[-1]
                    # guidance is current update
                    z = f"Date: {ts}, Update: {updates[-1]}"

                    if idx > 0:
                        yield data_idx, {"src": src, "tgt": tgt, "z": z}
                    data_idx += 1
