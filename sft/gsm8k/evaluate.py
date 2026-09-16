"""Evaluate GSM8K accuracy of the base model or the LoRA adapter."""

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from prepare_data import DATA_DIR

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
ADAPTER_PATH = Path(__file__).parent / "adapters"
OUTPUT_DIR = Path(__file__).parent / "outputs"

BATCH_SIZE = 128
MAX_TOKENS = 1024

NUMBER_PATTERN = r"[-+]?\d[\d,]*(?:\.\d+)?"


def to_number(text):
    return float(text.replace(",", ""))


def extract_strict_answer(text):
    """The number after '####', the GSM8K convention. None if absent."""
    match = re.search(rf"####\s*({NUMBER_PATTERN})", text)
    return to_number(match.group(1)) if match else None


def extract_flexible_answer(text):
    """Best-effort answer: '####', then 'the answer is N', then the last number."""
    answer = extract_strict_answer(text)
    if answer is not None:
        return answer

    matches = re.findall(rf"answer\s*(?:is|:|=)\s*\$?\s*({NUMBER_PATTERN})", text, re.IGNORECASE)
    if not matches:
        matches = re.findall(NUMBER_PATTERN, text)
    return to_number(matches[-1]) if matches else None


def load_jsonl(path, limit=None):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f][:limit]


def build_prompt(question, shots):
    """Plain text, no chat template: the base model has never seen one.

    Few-shot demonstrations show a base model the '#### <number>' convention
    and when to stop, which it otherwise has no reason to follow.
    """
    parts = [f"Question: {shot['question']}\nAnswer: {shot['answer']}" for shot in shots]
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def clean_prediction(text):
    """Drop everything from the first invented 'Question:' onward.

    Without a chat template nothing marks the end of the turn, so the model
    often continues by making up further problems.
    """
    return text.split("Question:")[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", action="store_true", help="Evaluate the LoRA adapter.")
    parser.add_argument("--shots", type=int, default=0, help="Few-shot examples from the training set.")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N test examples.")
    args = parser.parse_args()

    model, tokenizer = load(MODEL_NAME, adapter_path=str(ADAPTER_PATH) if args.adapter else None)
    examples = load_jsonl(DATA_DIR / "test.jsonl", args.limit)
    shots = load_jsonl(DATA_DIR / "train.jsonl", args.shots)

    predictions = []
    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start : start + BATCH_SIZE]
        prompts = [tokenizer.encode(build_prompt(e["question"], shots)) for e in batch]
        result = batch_generate(
            model,
            tokenizer,
            prompts=prompts,
            max_tokens=MAX_TOKENS,
            sampler=make_sampler(temp=0.0),
            completion_batch_size=BATCH_SIZE,
            prefill_batch_size=8,
        )
        predictions.extend(result.texts)
        mx.clear_cache()
        print(f"Generated {len(predictions)}/{len(examples)}")

    results = []
    for example, raw in zip(examples, predictions):
        prediction = clean_prediction(raw)
        gold = extract_strict_answer(example["answer"])
        strict = extract_strict_answer(prediction)
        flexible = extract_flexible_answer(prediction)
        results.append(
            {
                **example,
                "prediction": prediction,
                "gold": gold,
                "strict": strict,
                "flexible": flexible,
                "strict_correct": strict == gold,
                "flexible_correct": flexible == gold,
                "has_format": strict is not None,
                # batch_generate reports no finish reason; hitting the limit means cut off.
                "truncated": len(tokenizer.encode(raw)) >= MAX_TOKENS,
            }
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{'lora' if args.adapter else 'base'}-{args.shots}shot.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(f"{json.dumps(r, ensure_ascii=False)}\n" for r in results)

    total = len(results)
    print(f"\nModel: {'lora' if args.adapter else 'base'}, {args.shots}-shot, {total} examples")
    for label, key in [
        ("Strict accuracy", "strict_correct"),
        ("Flexible accuracy", "flexible_correct"),
        ("Format compliance", "has_format"),
        ("Truncated", "truncated"),
    ]:
        count = sum(r[key] for r in results)
        print(f"{label + ':':<19}{count}/{total} ({count / total:.2%})")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
