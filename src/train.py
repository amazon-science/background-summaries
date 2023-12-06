import argparse
import logging
from pathlib import Path

import numpy as np
import wandb
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from evaluate_summ import compute_summ_metrics
from utils import convert_to_dict, init_config

logger = logging.getLogger(__name__)


def train(config):
    dataset = load_from_disk(config["hf_tokenized_dataset_path"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config["model_name_or_path"],
        use_cache=False if config["gradient_checkpointing"] else True,
    )

    def compute_metrics(eval_preds):
        preds = eval_preds.predictions
        preds = preds[0] if isinstance(preds, tuple) else preds
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = eval_preds.label_ids
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # inputs = eval_preds.inputs
        # logger.info(inputs)
        # inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        # decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

        result = compute_summ_metrics(pred=decoded_preds, tgt=decoded_labels)
        return result

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_path"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        predict_with_generate=True,
        generation_max_length=config["max_tgt_length"],
        generation_num_beams=config["num_beams"],
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", False),
        learning_rate=config["lr"],
        num_train_epochs=config["epochs"],
        deepspeed=config.get("deepspeed_config", None),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        logging_dir=f"{config['output_path']}/logs",
        logging_strategy=config["logging_strategy"],
        logging_steps=config["logging_steps"],
        evaluation_strategy=config["evaluation_strategy"],
        save_strategy=config["save_strategy"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=config.get("load_best_model_at_end", False),
        metric_for_best_model=config["metric_for_best_model"],
        report_to="wandb" if not config.get("disable_wandb", False) else "none",
        include_inputs_for_metrics=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint="resume_checkpoint" in config)
    trainer.save_model()
    trainer.create_model_card()
    tokenizer.save_pretrained(config["output_path"])

    if config["load_best_model_at_end"]:
        best_path = trainer.state.best_model_checkpoint
        with open(f"{config['output_path']}/best_path.txt", "w") as f:
            f.write(best_path)
        logger.info(f"path to best model: {best_path}")


def main():
    args, _ = parse_args()
    config = init_config(args.config, args.config_name)

    if not config.get("disable_wandb", False):
        # wandb init, update config and create alert
        wandb.init(project=f"hf-train-{args.config_name}", dir=config["log_path"])
        wandb.config.update(convert_to_dict(config))
        alert_txt = f"starting hf-train for config: {args.config_name}"
        wandb.alert(title="starting job", text=alert_txt)

    train(config)


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=Path, help="config path")
    parser.add_argument("--config-name", type=str, help="config name")
    return parser.parse_known_args()


if __name__ == "__main__":
    main()
