"""
get best-worst counts (BUS-based) from eval_data.jsonl file
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def gpt_is_unanswered(answer: str):
    unanswerable_strings = [
        "text does not",
        "not provided in the background text",
        "not provided in the text",
        "not mentioned in the background text",
        "not mentioned in the text",
        "not specified in the background text",
        "not specified in the text",
        "not specifically mentioned in the background text",
        "not specifically mentioned in the text",
        "not explicitly stated in the background text",
        "not explicitly stated in the text",
        "not explicitly mentioned in the background text",
        "not explicitly mentioned in the text",
        "article does not provide",
        "article does not specify",
        "article does not mention",
        "Unanswerable",
        "unanswerable",
    ]
    for x in unanswerable_strings:
        if x in answer:
            return True
    return False


def mturk_is_unanswered(ans: str):
    if "none" in ans.lower():
        return True
    return False


def get_human_bw_scores(data: list[dict]):
    """
    best-worst ratings from humans
    """
    evt_sys2best, evt_sys2worst = {}, {}
    for idx in range(len(data)):
        # instance-level best and worst counts
        best_counts, worst_counts = defaultdict(int), defaultdict(int)
        evt = data[idx]["id"].rsplit("_", 2)[0]
        if evt not in evt_sys2best:
            evt_sys2best[evt], evt_sys2worst[evt] = defaultdict(int), defaultdict(int)
        for sys_ids in data[idx]["best_worst"]["best_summary"]:
            # each turker can select multiple systems as best
            for sys_id in sys_ids.split("|"):
                sys_name = sys_id.rsplit("_", 1)[-1]
                best_counts[sys_name] += 1
        for sys_ids in data[idx]["best_worst"]["worst_summary"]:
            for sys_id in sys_ids.split("|"):
                sys_name = sys_id.rsplit("_", 1)[-1]
                worst_counts[sys_name] += 1
        # majority vote
        best_majority = [sys for sys in best_counts if best_counts[sys] >= 2]
        worst_majority = [sys for sys in worst_counts if worst_counts[sys] >= 2]
        if len(set(best_majority) & set(worst_majority)) > 0:
            # system has both best and worst majority ratings
            # skip this instance
            continue
        for sys in best_majority:
            evt_sys2best[evt][sys] += 1
        for sys in worst_majority:
            evt_sys2worst[evt][sys] += 1

    return evt_sys2best, evt_sys2worst


def get_bus_bw_scores(data: list[dict], metric_name: str):
    """
    best-worst ratings from example-level BUS scores
    """
    if metric_name == "BUS_human":
        score_fn = mturk_is_unanswered
    elif metric_name in ["BUS_gpt-3.5-turbo", "BUS_gpt-4-0613"]:
        score_fn = gpt_is_unanswered

    evt_sys2best, evt_sys2worst = {}, {}
    for idx in range(len(data)):
        sys2bus = {}
        evt = data[idx]["id"].rsplit("_", 2)[0]
        if evt not in evt_sys2best:
            evt_sys2best[evt], evt_sys2worst[evt] = defaultdict(int), defaultdict(int)
        for sys in data[idx][metric_name]:
            sys2bus[sys] = np.mean(
                [not score_fn(ans) for ans in data[idx][metric_name][sys]["answers"]]
            )
        max_score = np.max(list(sys2bus.values()))
        min_score = np.min(list(sys2bus.values()))
        if max_score == min_score:
            # no best or worst system
            continue
        for sys in sys2bus:
            if sys2bus[sys] == max_score:
                evt_sys2best[evt][sys] += 1
            if sys2bus[sys] == min_score:
                evt_sys2worst[evt][sys] += 1

    return evt_sys2best, evt_sys2worst


def main(file_path: Path):
    data = []
    with open(file_path, "r") as rf:
        for line in rf:
            data += [json.loads(line)]

    # human best-worst counts
    evt_sys2best, evt_sys2worst = get_human_bw_scores(data)
    out_dict = {}
    for evt in evt_sys2best:
        out_dict[evt] = {}
        for sys in evt_sys2best[evt]:
            out_dict[evt][f"{sys}_best"] = evt_sys2best[evt][sys]
        for sys in evt_sys2worst[evt]:
            out_dict[evt][f"{sys}_worst"] = -1 * evt_sys2worst[evt][sys]

    print("---Human best-worst counts---")
    out = pd.DataFrame.from_dict(out_dict, orient="index")
    out.sort_index(axis=1, inplace=True)
    out.sort_index(axis=0, inplace=True)
    print(out.to_string())

    # get corpus-level counts
    out_dict = defaultdict(int)
    for evt in evt_sys2best:
        for sys in evt_sys2best[evt]:
            out_dict[f"{sys}_best"] += evt_sys2best[evt][sys]
        for sys in evt_sys2worst[evt]:
            out_dict[f"{sys}_worst"] += -1 * evt_sys2worst[evt][sys]
    print("---Human corpus-level best-worst counts---")
    out = pd.DataFrame.from_dict(out_dict, orient="index")
    out.sort_index(axis=0, inplace=True)
    print(out.transpose().to_string())

    # BUS best-worst counts
    for metric_name in ["BUS_gpt-3.5-turbo", "BUS_gpt-4-0613", "BUS_human"]:
        evt_sys2best, evt_sys2worst = get_bus_bw_scores(data, metric_name)
        out_dict = {}
        for evt in evt_sys2best:
            out_dict[evt] = {}
            for sys in evt_sys2best[evt]:
                out_dict[evt][f"{sys}_best"] = evt_sys2best[evt][sys]
            for sys in evt_sys2worst[evt]:
                out_dict[evt][f"{sys}_worst"] = -1 * evt_sys2worst[evt][sys]
        print(f"---{metric_name} best-worst counts---")
        out = pd.DataFrame.from_dict(out_dict, orient="index")
        out.sort_index(axis=1, inplace=True)
        out.sort_index(axis=0, inplace=True)
        print(out.to_string())

        # get corpus-level counts
        out_dict = defaultdict(int)
        for evt in evt_sys2best:
            for sys in evt_sys2best[evt]:
                out_dict[f"{sys}_best"] += evt_sys2best[evt][sys]
            for sys in evt_sys2worst[evt]:
                out_dict[f"{sys}_worst"] += -1 * evt_sys2worst[evt][sys]

        print(f"---{metric_name} corpus-level best-worst counts---")
        out = pd.DataFrame.from_dict(out_dict, orient="index")
        out.sort_index(axis=0, inplace=True)
        print(out.transpose().to_string())


if __name__ == "__main__":
    file_path = Path("results/eval_data.jsonl")
    main(file_path)
