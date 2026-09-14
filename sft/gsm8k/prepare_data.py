"""Prepare GSM8K splits in MLX-LM prompt/completion format."""

import json
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent / "data"


def convert_example(example):
    return {
        "prompt": (
            "Solve the following math problem step by step. "
            "End your response with '#### <number>'.\n\n"
            f"{example['question'].strip()}"
        ),
        "completion": example["answer"].strip(),
    }


def save_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(
            f"{json.dumps(example, ensure_ascii=False)}\n" for example in examples
        )


def main():
    dataset = load_dataset("openai/gsm8k", "main")

    train_valid = dataset["train"].train_test_split(test_size=0.1, seed=42)

    splits = {
        "train": train_valid["train"],
        "valid": train_valid["test"],
        "test": dataset["test"],
    }

    DATA_DIR.mkdir(exist_ok=True)

    for name, split in splits.items():
        examples = (convert_example(example) for example in split)
        save_jsonl(examples, DATA_DIR / f"{name}.jsonl")
        print(f"{name}: {len(split)}")

    print(f"\nSample example:\n{convert_example(splits['train'][0])}")


if __name__ == "__main__":
    main()
