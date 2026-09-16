"""LoRA fine-tuning for one epoch, with the loss on the answer only.

`mlx_lm.lora` can mask the prompt only for chat-template data, so this script
builds the plain-text examples itself and reuses MLX-LM's training loop.
"""

import types
from pathlib import Path

import numpy as np
import yaml
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model, yaml_loader

from evaluate import ADAPTER_PATH, load_jsonl

HERE = Path(__file__).parent


class PromptMaskedDataset:
    """MLX-LM masks every position before the returned offset."""

    def __init__(self, path, tokenizer):
        self.examples = load_jsonl(path)
        self.tokenizer = tokenizer

    def process(self, example):
        # Tokenized separately so the offset is exact and the prompt tokens
        # match what evaluate.py feeds the model.
        prompt = self.tokenizer.encode(f"Question: {example['question']}\nAnswer:")
        answer = self.tokenizer.encode(f" {example['answer']}") + [self.tokenizer.eos_token_id]
        return prompt + answer, len(prompt)

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)


def main():
    with open(HERE / "config.yaml") as f:
        config = yaml.load(f, yaml_loader)
    args = types.SimpleNamespace(**{**CONFIG_DEFAULTS, **config})
    args.adapter_path = str(ADAPTER_PATH)

    np.random.seed(args.seed)
    model, tokenizer = load(args.model)

    train_set = PromptMaskedDataset(HERE / args.data / "train.jsonl", tokenizer)
    valid_set = PromptMaskedDataset(HERE / args.data / "valid.jsonl", tokenizer)

    # One pass over the training data; only the final weights are saved.
    args.iters = len(train_set) // args.batch_size
    args.save_every = args.iters
    print(f"Training for one epoch: {args.iters} steps of {args.batch_size} examples")

    train_model(args, model, train_set, valid_set)


if __name__ == "__main__":
    main()
