import logging

import evaluate
import nltk
import numpy as np
import torch
from questeval.questeval_metric import QuestEval

logger = logging.getLogger(__name__)

nltk.download("punkt", quiet=True)


def postprocess_text(preds, labels=None):
    # rougeLSum expects newline after each sentence
    preds = [pred.strip() for pred in preds]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    if labels:
        processed_labels = []
        for label in labels:
            if isinstance(label, str):
                processed_labels += ["\n".join(nltk.sent_tokenize(label.strip()))]
            elif isinstance(label, list):
                # multi-reference
                processed_labels += [
                    [
                        "\n".join(nltk.sent_tokenize(label_item.strip()))
                        for label_item in label
                    ]
                ]
        labels = processed_labels

    return preds, labels


def compute_summ_metrics(pred: list, tgt: list = None, src: list = None):
    # basic pre-processing for rougeLSum computation
    pred, tgt = postprocess_text(preds=pred, labels=tgt)

    scores = {}
    if tgt:
        # if reference (tgt) is available
        rouge_metric = evaluate.load("rouge")
        # get ROUGE
        rouge_scores = rouge_metric.compute(
            predictions=pred, references=tgt, use_stemmer=True
        )
        for k, v in rouge_scores.items():
            scores[k] = round(v * 100, 1)

        # get BERTScore
        bertscore_metric = evaluate.load("bertscore")
        bertscore_scores = bertscore_metric.compute(
            predictions=pred, references=tgt, lang="en"
        )
        for k, v in bertscore_scores.items():
            if k == "hashcode":
                continue
            scores[f"bertscore_{k}"] = round(np.mean(v) * 100, 1)

    if src:
        # if src is available

        questeval_metric = QuestEval(
            no_cuda=not torch.cuda.is_available(), do_weighter=True
        )

        questeval_score = questeval_metric.corpus_questeval(
            hypothesis=pred,
            sources=src,
            list_references=tgt,
            batch_size=8,
        )
        scores["questeval"] = round(questeval_score["corpus_score"] * 100, 1)

        # BERTScore factuality
        bertscore_metric = evaluate.load("bertscore")
        bertscore_scores = bertscore_metric.compute(
            predictions=pred, references=src, lang="en"
        )
        scores["bertscore_fact_precision"] = round(
            np.mean(bertscore_scores["precision"]) * 100, 1
        )

    return scores
