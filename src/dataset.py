import logging
from pathlib import Path

import pandas as pd
import torch

logger = logging.getLogger(__name__)


class TSDataset:
    def __init__(self, dir_path: Path) -> None:
        """load timeline summarization dataset with update-background tuples

        Args:
            dir_path (Path): path to directory with topic sub-directories
        """

        self.events = {}
        self.inputs = {}
        self.annotators = ["annotator1", "annotator2", "annotator3"]

        self.load_data(dir_path)

    def load_data(self, dir_path: Path):
        for topic_path in dir_path.iterdir():
            event = topic_path.stem
            if not topic_path.is_dir():
                continue
            self.events[event] = {}
            self.inputs[event] = {}
            for tsv_path in topic_path.iterdir():
                logger.info(f"tsv: {tsv_path}")
                annotator = tsv_path.stem
                assert annotator in self.annotators
                df = pd.read_csv(tsv_path, sep="\t")
                df = df.fillna("")
                for row in df.itertuples():
                    ts = row.Date.strip("[]")
                    if ts not in self.events[event]:
                        self.events[event][ts] = {}
                        self.inputs[event][ts] = row._2
                    self.events[event][ts][annotator] = {
                        "update_ann": row.Update.strip("\n"),
                        "background_ann": row.Background.strip("\n"),
                    }

    def get_events(self):
        return list(self.events.keys())

    def get_updates(self):
        pass

    def get_inputs(self):
        """input updates"""
        return self.inputs

    def get_summaries(self):
        event2data = {}
        for event in self.events:
            ts_sorted = sorted(self.events[event].keys())
            updates, backgrounds = [], []
            for ts in ts_sorted:
                updates += [
                    [
                        self.events[event][ts][ann]["update_ann"]
                        for ann in self.annotators
                    ]
                ]
                backgrounds += [
                    [
                        self.events[event][ts][ann]["background_ann"]
                        for ann in self.annotators
                    ]
                ]
            event2data[event] = [ts_sorted, updates, backgrounds]
        return event2data


class SummDataset(torch.utils.data.Dataset):
    def __init__(self, docs: list[str], summaries: list[str]) -> None:
        super().__init__()
        self.docs = docs
        self.summaries = summaries

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, index):
        return (self.docs[index], self.summaries[index])
