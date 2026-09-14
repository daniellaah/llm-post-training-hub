"""Evaluate GSM8K accuracy using the final answer after ####."""

import argparse
import json
import re
from pathlib import Path

from mlx_lm import batch_generate, load

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
BATCH_SIZE = 128
ADAPTER_PATH = Path(__file__).parent / "adapters"
TEST_PATH = Path(__file__).parent / "data" / "test.jsonl"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def extract_answer(text):
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    return None


def load_test_data(limit=None):
    with open(TEST_PATH, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    return examples[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    adapter_path = ADAPTER_PATH if args.adapter else None

    model, tokenizer = load(MODEL_NAME, adapter_path=adapter_path)
    tokenizer.add_eos_token("<|im_end|>")
    examples = load_test_data(args.limit)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            add_generation_prompt=True,
        )
        for example in examples
    ]
    predictions = batch_generate(
        model,
        tokenizer,
        prompts=prompts,
        max_tokens=256,
        completion_batch_size=BATCH_SIZE,
        prefill_batch_size=BATCH_SIZE,
        verbose=True,
    ).texts

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / ("lora.jsonl" if adapter_path else "base.jsonl")
    correct = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, (example, prediction) in enumerate(zip(examples, predictions), start=1):
            predicted_answer = extract_answer(prediction)
            gold_answer = extract_answer(example["completion"])
            is_correct = (
                predicted_answer is not None and predicted_answer == gold_answer
            )
            correct += int(is_correct)

            result = {
                **example,
                "prediction": prediction,
                "predicted_answer": predicted_answer,
                "gold_answer": gold_answer,
                "correct": is_correct,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(
                f"{i}/{len(examples)} "
                f"pred={predicted_answer} "
                f"gold={gold_answer} "
                f"correct={is_correct}"
            )

    accuracy = correct / len(examples)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
