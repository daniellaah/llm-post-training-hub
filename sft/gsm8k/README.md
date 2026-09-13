# GSM8K SFT with LoRA

Fine-tune Qwen3-0.6B-Base on GSM8K using LoRA and MLX-LM.

## Setup

From the repository root:

```bash
uv sync --locked
cd sft/gsm8k
```

## Usage

### 1. Prepare data

```bash
uv run python prepare_data.py
```

[GSM8K](https://huggingface.co/datasets/openai/gsm8k) (`main`): split the official training set 90/10 for training and validation (seed `42`), and keep the official test set.

| File | Examples |
| --- | ---: |
| `data/train.jsonl` | 6,725 |
| `data/valid.jsonl` | 748 |
| `data/test.jsonl` | 1,319 |

### 2. Evaluate base model

TODO.

### 3. Train

TODO.

### 4. Evaluate fine-tuned model

TODO.

### 5. Demo

TODO.
