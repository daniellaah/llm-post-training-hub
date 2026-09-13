"""Download GSM8K and create reproducible train/validation/test splits."""

from pathlib import Path

from datasets import load_dataset


def main():
    dataset = load_dataset("openai/gsm8k", "main")
    train_valid = dataset["train"].train_test_split(test_size=0.1, seed=42)
    splits = {
        "train": train_valid["train"],
        "valid": train_valid["test"],
        "test": dataset["test"],
    }

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    for name, split in splits.items():
        output_path = data_dir / f"{name}.jsonl"
        split.to_json(output_path, force_ascii=False)
        print(f"{name}: {len(split):,} examples -> {output_path}")


if __name__ == "__main__":
    main()
