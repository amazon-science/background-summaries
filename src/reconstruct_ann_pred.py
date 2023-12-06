import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_json(file_path: Path):
    data = []
    with open(file_path, "r") as rf:
        for line in rf:
            data += [json.loads(line.strip())]
    return data


def generate_examples(events_path: Path, splits_path: Path):
    # load events for the split
    with open(splits_path, "r") as rf:
        event_names = [line.strip() for line in rf.readlines()]

    examples = []
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

                examples += [
                    {
                        "ts": ts,
                        "update": update,
                        "src": src,
                        "tgt": tgt,
                        "z": z,
                        "event": event,
                        "ann": ann,
                    }
                ]
    return examples


def map_preds(preds, gold, output_path: Path):
    logger.info(f"gold: {len(gold)}\tpreds: {len(preds)}")

    out = {}
    pidx = 0
    for gidx in range(len(gold)):
        event = gold[gidx]["event"]
        ann = gold[gidx]["ann"]
        if event not in out:
            out[event] = {}
        if ann not in out[event]:
            # create the first entry (no background)
            out[event][ann] = defaultdict(list)
            out[event][ann]["Date"] += [gold[gidx]["ts"]]
            out[event][ann]["Update"] += [gold[gidx]["update"]]
            out[event][ann]["Background"] += [""]
            continue
        out[event][ann]["Date"] += [gold[gidx]["ts"]]
        out[event][ann]["Update"] += [gold[gidx]["update"]]
        out[event][ann]["Background"] += [preds[pidx]["pred"]]
        pidx += 1

    for event in out:
        for ann in out[event]:
            df = pd.DataFrame.from_dict(out[event][ann])
            event_path = output_path / event
            event_path.mkdir(exist_ok=True, parents=True)
            ann_path = event_path / f"{ann}.tsv"
            df.to_csv(ann_path, sep="\t", index=False)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    parser = argparse.ArgumentParser(
        description="map model predictions to datasets input per annotator"
    )
    parser.add_argument("--preds", type=Path, help="model predictions")
    parser.add_argument("--data", type=Path, help="dataset path")
    parser.add_argument("--split", type=str, help="dataset split")
    parser.add_argument("--out", type=Path, help="dir to write per annotator outputs")
    args = parser.parse_args()

    preds = load_json(args.preds)
    gold = generate_examples(args.data, args.data / "splits" / f"{args.split}.txt")

    map_preds(preds, gold, args.out)
