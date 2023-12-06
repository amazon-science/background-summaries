"""
Create a single JSON-L file for evaluation results on 1k test examples.
- update
- 3 backgrounds (human, Flan-T5, GPT-3.5)
- BUS (GPT and human) questions and answers
- BW evaluation: best sys, worst sys, and justifications
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyhocon

logger = logging.getLogger(__name__)

SYSTEMS = ["human", "flan-t5-xl", "gpt-3.5-turbo"]


def load_data(file_path: Path):
    df = pd.read_csv(file_path, sep="\t", na_filter=False)
    return df


def load_bw_jsonl(file_path: Path):
    id2update, id2background = {}, {}
    id2bw = {}
    with open(file_path, "r") as rf:
        for line in rf:
            hit_data = json.loads(line)
            data_id = hit_data["data"]["summary1_name"].rsplit("_", 1)[0]
            id2update[data_id] = hit_data["data"]["update"]
            id2background[data_id] = {}
            for _sys in SYSTEMS:
                id2background[data_id][_sys] = ""
            sys1 = hit_data["data"]["summary1_name"].rsplit("_", 1)[1]
            id2background[data_id][sys1] = hit_data["data"]["summary1"]
            sys2 = hit_data["data"]["summary2_name"].rsplit("_", 1)[1]
            id2background[data_id][sys2] = hit_data["data"]["summary2"]
            sys3 = hit_data["data"]["summary3_name"].rsplit("_", 1)[1]
            id2background[data_id][sys3] = hit_data["data"]["summary3"]
            id2bw[data_id] = hit_data["answers"]
    return id2update, id2background, id2bw


def load_bus_human_all_qs_jsonl(file_path: Path):
    id2q = {}
    with open(file_path, "r") as rf:
        for line in rf:
            hit_data = json.loads(line)
            data_id = hit_data["data"]["summary1_name"].rsplit("_", 1)[0]
            questions = list(hit_data["answers"].values())
            # two sets of five questions each
            id2q[data_id] = [list(x) for x in zip(*questions)]
    return id2q


def load_bus_human_jsonl(qa_file_path: Path, q_file_path: Path):
    id2q = load_bus_human_all_qs_jsonl(q_file_path)
    id2qa = {}
    with open(qa_file_path, "r") as rf:
        for line in rf:
            hit_data = json.loads(line)
            data_id, sys = hit_data["data"]["name"].rsplit("_", 1)
            if data_id not in id2qa:
                id2qa[data_id] = {}
                for _sys in SYSTEMS:
                    id2qa[data_id][_sys] = {}
            id2qa[data_id][sys]["questions"] = [
                v for k, v in hit_data["data"].items() if k.startswith("question")
            ]
            id2qa[data_id][sys]["answers"] = [
                v[0] for _, v in hit_data["answers"].items()
            ]
            id2qa[data_id][sys]["extra_questions"] = id2q[data_id][1]
            assert id2q[data_id][0] == id2qa[data_id][sys]["questions"], (
                id2q[data_id][0],
                id2qa[data_id][sys]["questions"],
            )
    return id2qa


def main(args):
    split = "test"
    split_path = f"data/splits/{split}.txt"
    with open(split_path, "r") as rf:
        events = [line.strip() for line in rf.readlines()]
    anns = ["annotator1", "annotator2", "annotator3"]
    logger.info(f"split: {split}")
    logger.info(f"events: {events}")
    logger.info(f"anns: {anns}")

    # load best-worst scaling results
    id2update, id2background, id2bw = load_bw_jsonl(args.bw_jsonl)

    # load bus-human results
    id2human_qa = load_bus_human_jsonl(
        qa_file_path=args.bus_human_qa_jsonl, q_file_path=args.bus_human_q_jsonl
    )

    # load system configs
    sys_configs = {}
    for sys_config_path, sys_config_name in zip(args.config, args.config_name):
        sys_configs[sys_config_name] = init_config(sys_config_path, sys_config_name)

    id2gpt_qa = {}
    for bus_config_name in ["gpt-4-0613", "gpt-3.5-turbo"]:
        id2gpt_qa[bus_config_name] = {}
        # collect bus preds paths for each system
        sys2dat = defaultdict(list)
        for sys_name, sys_config in sys_configs.items():
            if sys_config.get("model", None) == "gpt-3.5-turbo":
                bus_path = sys_config["output_path"]
                bus_path /= f"bus-{bus_config_name}-{split}.jsonl"
            else:
                bus_path = Path(sys_config["model_name_or_path"])
                bus_path /= f"bus-{bus_config_name}-{split}.jsonl"

            with open(bus_path, "r") as rf:
                for line in rf:
                    sys2dat[sys_name] += [json.loads(line)]

        event2len = {}
        event2ts = defaultdict(list)
        for event in events:
            df = load_data(Path("data/events") / event / "annotator1.tsv")
            event2len[event] = len(df) - 1
            event2ts[event] = [ts.strip("[]") for ts in df["Date"]][1:]
        # logger.info(event2len)

        ann2dat = defaultdict(list)
        for ann in anns:
            if bus_config_name == "gpt-4-0613":
                ann_path = args.bus_gpt_ann / f"{ann}_{split}_{bus_config_name}.jsonl"
            elif bus_config_name == "gpt-3.5-turbo":
                ann_path = args.bus_gpt_ann / f"{ann}_{split}.jsonl"
            with open(ann_path, "r") as rf:
                for line in rf:
                    ann2dat[ann] += [json.loads(line)]

        ann_dat_idx = 0
        sys_dat_idx = 0
        for event in events:
            for ann in anns:
                filter_indices = [
                    idx
                    for idx in range(event2len[event])
                    if f"{event}_{ann}_{event2ts[event][idx]}" in id2update
                ]
                filter_ids = [
                    f"{event}_{ann}_{event2ts[event][idx]}" for idx in filter_indices
                ]
                ann_data = ann2dat[ann][ann_dat_idx : ann_dat_idx + event2len[event]]
                # filter out of the 1k examples
                ann_data = [ann_data[idx] for idx in filter_indices]
                # populate BUS-GPT qa for human backgrounds
                for idx in range(len(filter_ids)):
                    id2gpt_qa[bus_config_name][filter_ids[idx]] = {}
                    # id2update Data: <>, Article: <> format
                    # assert ann_data[idx]["Update"] == id2update[filter_ids[idx]], (
                    #     filter_ids[idx],
                    #     ann_data[idx]["Update"],
                    #     id2update[filter_ids[idx]],
                    # )
                    # assert (
                    #     ann_data[idx]["Background"]
                    #     == id2background[filter_ids[idx]]["human"]
                    # ), (
                    #     filter_ids[idx],
                    #     ann_data[idx]["Background"],
                    #     id2background[filter_ids[idx]]["human"],
                    # )
                    id2gpt_qa[bus_config_name][filter_ids[idx]]["human"] = {
                        "questions": [
                            ann_data[idx]["QA"][idy]["Question"]
                            for idy in range(len(ann_data[idx]["QA"]))
                        ],
                        "answers": [
                            ann_data[idx]["QA"][idy]["Answer"]
                            for idy in range(len(ann_data[idx]["QA"]))
                        ],
                    }
                for sys_name in sys2dat:
                    sys_data = sys2dat[sys_name][
                        sys_dat_idx : sys_dat_idx + event2len[event]
                    ]
                    # filter
                    sys_data = [sys_data[idx] for idx in filter_indices]
                    # populate BUS-GPT qa for system backgrounds
                    for idx in range(len(filter_ids)):
                        # assert sys_data[idx]["Update"] == id2update[filter_ids[idx]]
                        # assert (
                        #     sys_data[idx]["Background"]
                        #     == id2background[filter_ids[idx]][sys_name]
                        # )
                        id2gpt_qa[bus_config_name][filter_ids[idx]][sys_name] = {
                            "questions": [
                                sys_data[idx]["QA"][idy]["Question"]
                                for idy in range(len(sys_data[idx]["QA"]))
                            ],
                            "answers": [
                                sys_data[idx]["QA"][idy]["Answer"]
                                for idy in range(len(sys_data[idx]["QA"]))
                            ],
                        }
                sys_dat_idx += event2len[event]
            ann_dat_idx += event2len[event]

    with open(args.out, "w") as wf:
        for data_id in id2bw:
            data = {
                "id": data_id,
                "update": id2update[data_id],
                "background": id2background[data_id],
                "best_worst": id2bw[data_id],
                "BUS_human": id2human_qa[data_id],
                "BUS_gpt-3.5-turbo": id2gpt_qa["gpt-3.5-turbo"][data_id],
                "BUS_gpt-4-0613": id2gpt_qa["gpt-4-0613"][data_id],
            }
            wf.write(json.dumps(data) + "\n")


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
    parser = argparse.ArgumentParser(description="compile all human eval results")
    parser.add_argument("--config", type=Path, nargs="+", help="config file")
    parser.add_argument("--config-name", type=str, nargs="+", help="config name")
    parser.add_argument(
        "--bw-jsonl",
        type=Path,
        default="/projects/tir6/general/vpratapa/research/mturk_results/adithya-mturk-bestworst-3annotators-results-20230619/collect.json",
        help="path to jsonl",
    )
    parser.add_argument(
        "--bus-human-qa-jsonl",
        type=Path,
        default="/projects/tir6/general/vpratapa/research/mturk_results/adithya-mturk-bus-answers-merged-20230622/collect.json",
        help="path to jsonl",
    )
    parser.add_argument(
        "--bus-human-q-jsonl",
        type=Path,
        default="/projects/tir6/general/vpratapa/research/mturk_results/adithya-mturk-bus-20230621-continue/run/collect.json",
        help="path to jsonl",
    )
    parser.add_argument(
        "--bus-gpt-ann",
        type=Path,
        default="/projects/tir6/general/vpratapa/research/outputs/ts-dev/anns/",
        help="path to BUS--GPT on human backgrounds",
    )
    parser.add_argument("--out", type=Path, help="path to write jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
