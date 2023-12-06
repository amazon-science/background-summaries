import argparse
import logging
import re
from functools import partial
from pathlib import Path

import numpy as np
import spacy
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

from utils import init_config

logger = logging.getLogger(__name__)

spacy_nlp = spacy.load("en_core_web_lg")


def postprocess_label_sequences(sequences, tokenizer):
    """Replaced pad_token_id by -100 so that they are ignored in the training loss"""
    return [
        [(label if label != tokenizer.pad_token_id else -100) for label in seq]
        for seq in sequences
    ]


def guided_truncate(
    batch: list,
    tokenizer: AutoTokenizer,
    max_src_length: int,
    max_tgt_length: int,
    max_z_length: int,
    src_prefix: str,
    z_prefix: str,
    task_prefix: str,
    task_suffix: str = None,
    truncation_strategy: str = "left",
    z_type: str = "update",
):
    """prepare input examples for instruction fine-tuning

    template: Article: <src>. Guidance: <z>. \
        Summarize the article with focus on the information provided in the guidance.

    Args:
        batch (list): _description_
        tokenizer (AutoTokenizer): _description_
        max_src_length (int): _description_
        max_tgt_length (int): _description_
        max_z_length (int): _description_
    """

    # article prefix tokens
    src_prefix_toks = tokenizer(src_prefix, padding=False, truncation=False)
    assert src_prefix_toks["input_ids"][-1] == tokenizer.eos_token_id
    src_prefix_toks["input_ids"].pop()
    src_prefix_toks["attention_mask"].pop()
    max_src_length -= len(src_prefix_toks["input_ids"])

    # guidance prefix tokens
    z_prefix_toks = tokenizer(z_prefix, padding=False, truncation=False)
    assert z_prefix_toks["input_ids"][-1] == tokenizer.eos_token_id
    z_prefix_toks["input_ids"].pop()
    z_prefix_toks["attention_mask"].pop()
    max_z_length -= len(z_prefix_toks["input_ids"])

    # task prefix tokens
    task_prefix_toks = tokenizer(task_prefix, padding=False, truncation=False)
    assert task_prefix_toks["input_ids"][-1] == tokenizer.eos_token_id
    task_prefix_toks["input_ids"].pop()
    task_prefix_toks["attention_mask"].pop()
    max_src_length -= len(task_prefix_toks["input_ids"])

    assert max_src_length > 0 and max_z_length > 0

    # truncate updates (& dates) from the input to fit into max length
    truncated_docs = []
    for doc in batch["src"]:
        truncated_doc = doc
        tokens = tokenizer(truncated_doc)["input_ids"]
        while len(tokens) > max_src_length:
            doc_splits = re.split(r"(Date: [0-9]{4}-[0-9]{2}-[0-9]{2})", truncated_doc)
            if truncation_strategy == "left":
                truncated_doc = "".join(doc_splits[3:])
            elif truncation_strategy == "right":
                truncated_doc = "".join(doc_splits[:-2]).strip()
            tokens = tokenizer(truncated_doc)["input_ids"]
        truncated_docs += [truncated_doc]
    src_toks = tokenizer(truncated_docs, padding=False, truncation=False)

    if z_type == "update":
        logger.info(f"using {z_type} as Z")
        # truncate guidance update from the input to fit into max length
        z_toks = tokenizer(
            batch["z"], max_length=max_z_length, padding=False, truncation=True
        )
    elif z_type == "entities":
        logger.info(f"using {z_type} as Z")
        ents = []
        z_docs = [
            re.match(
                r"^Date: [0-9]{4}-[0-9]{2}-[0-9]{2}, Update: (?P<txt>.*)$", _item
            ).group("txt")
            for _item in batch["z"]
        ]
        for doc in spacy_nlp.pipe(z_docs, n_process=1):
            ents += [", ".join([ent.text for ent in doc.ents]) + "."]
        z_toks = tokenizer(
            ents, max_length=max_z_length, padding=False, truncation=True
        )
    else:
        raise NotImplementedError

    # prepend prefix tokens
    output = {"input_ids": [], "attention_mask": []}
    for idx in range(len(src_toks["input_ids"])):
        for key in ["input_ids", "attention_mask"]:
            output[key] += [
                task_prefix_toks[key]
                + z_prefix_toks[key]
                + z_toks[key][idx]
                + src_prefix_toks[key]
                + src_toks[key][idx]
            ]

    # tokenize target, with truncation (FIXME:?)
    labels = tokenizer(
        batch["tgt"], max_length=max_tgt_length, padding=False, truncation=True
    )
    output["labels"] = postprocess_label_sequences(labels["input_ids"], tokenizer)

    return output


def default_truncate(
    batch: list,
    tokenizer: AutoTokenizer,
    max_src_length: int,
    max_tgt_length: int,
    task_prefix: str,
    task_suffix: str = None,
    src_prefix: str = None,
    z_prefix: str = None,
    max_z_length: int = None,
    z_type: str = None,
    truncation_strategy: str = "left",
):
    assert truncation_strategy in ["left", "right"]

    max_doc_length = max_src_length
    # tokenize task prefix and suffix strings
    # add max document length to accomodate prefix and suffix
    if task_prefix:
        prefix_toks = tokenizer(task_prefix, padding=False, truncation=False)
        assert prefix_toks["input_ids"][-1] == tokenizer.eos_token_id
        prefix_toks["input_ids"].pop()
        prefix_toks["attention_mask"].pop()
        max_doc_length -= len(prefix_toks["input_ids"])
    if task_suffix:
        suffix_toks = tokenizer(task_suffix, padding=False, truncation=False)
        assert suffix_toks["input_ids"][0] == tokenizer.bos_token_id
        suffix_toks["input_ids"].pop(0)
        suffix_toks["attention_mask"].pop(0)
        max_doc_length -= len(suffix_toks["input_ids"])

    assert max_doc_length > 0

    # truncate updates (& dates) from the input to fit into max length
    truncated_docs = []
    for doc in batch["src"]:
        truncated_doc = doc
        tokens = tokenizer(truncated_doc)["input_ids"]
        while len(tokens) > max_doc_length:
            doc_splits = re.split(r"(Date: [0-9]{4}-[0-9]{2}-[0-9]{2})", truncated_doc)
            if truncation_strategy == "left":
                truncated_doc = "".join(doc_splits[3:])
            elif truncation_strategy == "right":
                truncated_doc = "".join(doc_splits[:-2]).strip()
            tokens = tokenizer(truncated_doc)["input_ids"]
        truncated_docs += [truncated_doc]

    input_toks = tokenizer(truncated_docs, padding=False, truncation=False)

    # prepend prefix and append suffix tokens
    for key in ["input_ids", "attention_mask"]:
        if task_prefix:
            input_toks[key] = [prefix_toks[key] + seq for seq in input_toks[key]]
        if task_suffix:
            input_toks[key] = [seq + suffix_toks[key] for seq in input_toks[key]]

    # tokenize target, with truncation (FIXME:?)
    labels = tokenizer(
        batch["tgt"], max_length=max_tgt_length, padding=False, truncation=True
    )
    labels = postprocess_label_sequences(labels["input_ids"], tokenizer)

    return {
        "input_ids": input_toks["input_ids"],
        "attention_mask": input_toks["attention_mask"],
        "labels": labels,
    }


def middle_truncate():
    return NotImplementedError


preprocess_fn_map = {
    "truncate_left": partial(default_truncate, truncation_strategy="left"),
    "truncate_middle": middle_truncate,
    "truncate_right": partial(default_truncate, truncation_strategy="right"),
    "truncate_guided_left": partial(guided_truncate, truncation_strategy="left"),
    "truncate_guided_right": partial(guided_truncate, truncation_strategy="right"),
}


def print_samples(
    dir_path: Path, tokenizer: AutoTokenizer, split: str = "train", N: int = 5
):
    dataset = load_from_disk(dir_path)
    N = min(len(dataset[split]), N)

    input_ids = dataset[split][:N]["input_ids"]
    src_toks = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    labels = dataset[split][:N]["labels"]
    # FIXME: all lists within labels might be of different sizes
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    tgt_toks = tokenizer.batch_decode(labels, skip_special_tokens=True)

    for idx in range(N):
        logger.warning(f"{idx+1}/{N}, src: {src_toks[idx]}")
        logger.warning(f"{idx+1}/{N}, tgt: {tgt_toks[idx]}")


def process_dataset(config):
    dataset = load_dataset(config["dataset_path"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if config["max_src_length"] > tokenizer.model_max_length:
        logger.info(
            f"specified max src length: {config['max_src_length']}"
            f" is higher than model limit ({tokenizer.model_max_length}."
            f" using model limit."
        )
        config["max_src_length"] = tokenizer.model_max_length

    preprocess_fn = partial(
        preprocess_fn_map[f"truncate_{config['truncation_strategy']}"],
        tokenizer=tokenizer,
        task_prefix=config.get("task_prefix", None),
        task_suffix=config.get("task_suffix", None),
        max_src_length=config["max_src_length"],
        max_tgt_length=config["max_tgt_length"],
        max_z_length=config.get("max_z_length", None),
        src_prefix=config.get("src_prefix", None),
        z_prefix=config.get("z_prefix", None),
        z_type=config.get("z_type", None),
    )
    logger.info("tokenizing dataset")
    tokenized_data = dataset.map(
        preprocess_fn, batched=True, remove_columns=list(dataset["train"].features)
    )
    output_path = config["hf_tokenized_dataset_path"]
    output_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"saving tokenized data to {output_path}")
    tokenized_data.save_to_disk(output_path)

    # print_samples(dir_path=output_path, tokenizer=tokenizer)


def parse_args():
    parser = argparse.ArgumentParser(description="convert into hf datasets format")
    parser.add_argument("--config", type=Path, help="config path")
    parser.add_argument("--config-name", type=str, help="config name")
    return parser.parse_args()


def main():
    args = parse_args()
    config = init_config(args.config, args.config_name)
    process_dataset(config)


if __name__ == "__main__":
    main()
