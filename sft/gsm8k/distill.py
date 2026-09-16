"""Generate step-by-step solutions for GSM8K questions with a stronger model.

The original GSM8K answers are terse, and the base model has already seen
them during pretraining, so fine-tuning on them teaches nothing new. Detailed
solutions from a larger teacher do carry new information. Only solutions whose
final answer matches the gold answer are kept.
"""

import argparse
import re

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from evaluate import NUMBER_PATTERN, extract_strict_answer, load_jsonl, to_number
from prepare_data import DATA_DIR, save_jsonl

TEACHER = "Qwen/Qwen3-4B"
BATCH_SIZE = 128
MAX_TOKENS = 4096
MAX_SOLUTION_TOKENS = 1024  # longer outputs are runaway generations, not better solutions

INSTRUCTION = (
    "Solve the problem step by step. "
    "Finish with the final numeric answer on its own line in the form '#### <number>'."
)


def extract_teacher_answer(solution):
    """Like extract_strict_answer, but also accept a LaTeX \\boxed{} answer."""
    answer = extract_strict_answer(solution)
    if answer is None:
        boxed = re.findall(rf"\\boxed\{{\s*\$?\s*({NUMBER_PATTERN})\s*\}}", solution)
        answer = to_number(boxed[-1]) if boxed else None
    return answer


def distill_split(model, tokenizer, split, limit):
    examples = load_jsonl(DATA_DIR / f"{split}.jsonl", limit)
    kept = []

    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start : start + BATCH_SIZE]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{e['question']}\n\n{INSTRUCTION}"}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for e in batch
        ]
        result = batch_generate(
            model,
            tokenizer,
            prompts=prompts,
            max_tokens=MAX_TOKENS,
            sampler=make_sampler(temp=0.7, top_p=0.8, top_k=20),
            completion_batch_size=BATCH_SIZE,
            prefill_batch_size=8,
        )
        mx.clear_cache()

        for example, solution in zip(batch, result.texts):
            solution = solution.strip()
            answer = extract_teacher_answer(solution)
            if answer != extract_strict_answer(example["answer"]):
                continue
            if len(tokenizer.encode(solution)) > MAX_SOLUTION_TOKENS:
                continue
            # The student must learn the GSM8K marker, even when the teacher used \boxed{}.
            if extract_strict_answer(solution) is None:
                solution += f"\n#### {int(answer) if answer.is_integer() else answer}"
            kept.append({"question": example["question"], "answer": solution})

        print(f"{split}: {start + len(batch)}/{len(examples)} processed, {len(kept)} kept")

    save_jsonl(kept, DATA_DIR / "distill" / f"{split}.jsonl")
    print(f"{split}: kept {len(kept)}/{len(examples)} ({len(kept) / len(examples):.1%})\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Only process the first N questions per split.")
    args = parser.parse_args()

    mx.random.seed(0)
    model, tokenizer = load(TEACHER)
    (DATA_DIR / "distill").mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid"]:
        distill_split(model, tokenizer, split, args.limit)


if __name__ == "__main__":
    main()
