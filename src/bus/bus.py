import logging

import numpy as np
from models import GPT

logger = logging.getLogger(__name__)


def postprocess(questions: list[list[str]], answers: list[list[str]]):
    assert len(questions) == len(answers)
    out = []
    for ts_questions, ts_answers in zip(questions, answers):
        ts_out = []
        for question, answer in zip(ts_questions, ts_answers):
            q_out = question.replace("\n", " ").strip()
            a_out = answer.replace("\n", " ").strip()
            ts_out += [f"Question: {q_out}, Answer: {a_out}"]
        out += ["\\n".join(ts_out)]
    return out


def is_unanswered(answer: str):
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


def compute_bus(
    backgrounds: list,
    updates: list,
    model_kwargs: dict,
    return_per_example: bool = False,
):
    output = {}
    # init model
    generator = GPT(**model_kwargs)

    # list[list[str]]
    questions = generator.gen_questions(contexts=updates)
    answers = generator.gen_answers(contexts=backgrounds, questions=questions)

    # log questions and answers
    output["questions"] = questions
    output["answers"] = answers

    if return_per_example:
        output["bus"] = []
        for idx in range(len(answers)):
            scores = [not is_unanswered(ans) for ans in answers[idx]]
            output["bus"] += [np.mean(scores)]
    else:
        questions = [y for x in questions for y in x]
        answers = [y for x in answers for y in x]
        scores = [not is_unanswered(ans) for ans in answers]
        output["bus"] = np.mean(scores)

    return output
