# Evaluation Data (BUS, Mechanical Turk)

`eval_data.jsonl` contains the evaluation results on a random sample of 1000 updates from our dataset. For each update, we include three background summaries from Flan-T5-XL, GPT-3.5-Turbo and (three) human annotators.

We include best-worst ratings from MTurk as well as question-answer pairs used for computing Background Utility Score (BUS).

Each line in `eval_data.jsonl` corresponds to one update. See `format.json` for more details on the format of each line.

Contents

- Best-Worst scaling results from MTurk (BWS)
    - 3 Turkers per update.
    - Each Turker rates the best, worst systems and optionally provides a justification.
- Background Utility Score (BUS) based on human question-answer pairs from MTurk (BUS--human)
    - We collected two sets of questions per update.
    - For the first set of questions, we collected answers from each of the three background summaries.
- BUS based on GPT-generated question-answer pairs (BUS--GPT-3.5 and BUS--GPT-4)
    - For each update-background pair, we prompted GPT for question-answer pairs.
    - We collected pairs by prompting both GPT-3.5 and GPT-4.
