import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

from utils import convert_to_dict, init_config

logger = logging.getLogger(__name__)


def zero_to_fp32(model_dir: Path):
    """generate pytorch_model.bin

    Args:
        model_dir (Path): model directory
    """

    if (model_dir / "pytorch_model.bin").exists():
        return
    if (model_dir / "zero_to_fp32.py").exists():
        logger.info("collecting fp32 weights")
        args = [
            "python",
            str(model_dir / "zero_to_fp32.py"),
            str(model_dir),
            str(model_dir / "pytorch_model.bin"),
        ]
        subprocess.run(
            args,
            check=True,
            capture_output=True,
        )
    return


def evaluate(config):
    dataset = load_from_disk(config["hf_tokenized_dataset_path"])
    dataset = dataset[config["split"]]
    if config.get("debug", False):
        dataset = dataset[:32]

    accelerator = Accelerator()

    zero_to_fp32(Path(config["model_name_or_path"]))

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config["model_name_or_path"], torch_dtype="auto"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
    eval_dataloader = DataLoader(
        dataset, collate_fn=data_collator, batch_size=config["batch_size"]
    )

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    preds, labels, inputs = [], [], []
    model.eval()
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **config["gen_kwargs"],
            )
            preds += accelerator.gather(generated_tokens).cpu().tolist()
            # Replace -100 in the labels as we can't decode them.
            batch_labels = accelerator.gather(batch["labels"]).cpu().numpy()
            batch_labels = np.where(
                batch_labels != -100, batch_labels, tokenizer.pad_token_id
            )
            labels += batch_labels.tolist()
            inputs += accelerator.gather(batch["input_ids"]).cpu().tolist()

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

    pred_path = Path(config["model_name_or_path"]) / f"{config['split']}_preds.jsonl"
    with open(pred_path, "w") as wf:
        for src, tgt, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
            wf.write(json.dumps({"src": src, "tgt": tgt, "pred": pred}) + "\n")

    # result = compute_summ_metrics(
    #     pred=decoded_preds, tgt=decoded_labels, src=decoded_inputs
    # )
    # logger.info(result)


def main():
    args, _ = parse_args()
    config = init_config(args.config, args.config_name)

    # wandb init, update config and create alert
    wandb.init(project=f"ts-eval-{args.config_name}", dir=config["log_path"])
    wandb.config.update(convert_to_dict(config))
    alert_txt = f"starting eval for config: {args.config_name}"
    wandb.alert(title="starting job", text=alert_txt)

    evaluate(config)


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=Path, help="config path")
    parser.add_argument("--config-name", type=str, help="config name")
    return parser.parse_known_args()


if __name__ == "__main__":
    main()
