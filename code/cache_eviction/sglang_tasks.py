"""AIME 2026 task files, prompt rendering and answer scoring for the gpt-oss clients.

Prompts are rendered with the checkpoint's harmony chat template and sent as
input_ids, so prompt token counts are exact.
"""
import argparse
from pathlib import Path
import re
import time
from .files import save

BOXED = r"\\boxed\{\s*([0-9]+)\s*\}"


AIME_SYSTEM = ("Solve the problem and put the final answer in \\boxed{}. "
               "The answer is an integer between 0 and 999, inclusive.")


def aime_tasks(parquet, revision):
    import pandas as pd
    tasks = []
    for row in pd.read_parquet(parquet).sort_values("problem_idx").itertuples():
        tasks.append({"kind": "aime", "id": f"aime2026-{row.problem_idx:02d}", "source": f"MathArena/aime_2026@{revision}",
                      "answer": str(row.answer), "answer_regex": BOXED, "problem": row.problem,
                      "messages": [{"role": "system", "content": AIME_SYSTEM}, {"role": "user", "content": row.problem}]})
    return tasks


def answer_text(text, finish):
    """The final channel is scored. A run that stopped without one is scored on its last
    boxed answer, and a run truncated inside its analysis has no answer."""
    marker = "<|channel|>final<|message|>"
    return text.rsplit(marker, 1)[1] if marker in text else text if finish == "stop" else ""


def score(text, task):
    matches = re.findall(task["answer_regex"], text)
    prediction = re.sub(r"\s", "", matches[-1]) if matches else None
    return {"prediction": prediction, "quality": int(prediction == task["answer"])}


def render(task, model_dir, effort, date=None):
    """Harmony prompt ids. The template stamps today's date, so date= reproduces an earlier run."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if date is None:
        return tokenizer.apply_chat_template(task["messages"], tokenize=True, add_generation_prompt=True,
                                             reasoning_effort=effort, return_dict=False)
    text = tokenizer.apply_chat_template(task["messages"], tokenize=False, add_generation_prompt=True,
                                         reasoning_effort=effort)
    stamp = "Current date: " + time.strftime("%Y-%m-%d")
    if text.count(stamp) != 1:
        raise ValueError("chat template date stamp not found")
    return tokenizer(text.replace(stamp, "Current date: " + date), add_special_tokens=False)["input_ids"]


def main():
    parser = argparse.ArgumentParser(description="Write the AIME 2026 task files")
    parser.add_argument("kind", choices=["aime"])
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for task in aime_tasks(args.parquet, args.revision):
        save(Path(args.output) / f"{task['id']}.json", task)


if __name__ == "__main__":
    main()
