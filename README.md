# LLM Post-Training Hub

Small, self-contained experiments that train a post-training stage from scratch on a
Mac and measure its effect against a fair baseline. One directory per method, one
shared benchmark, the same evaluation protocol throughout, so results are comparable
across stages.

| Method                 | Directory               | Status  | Result (GSM8K, Qwen3-0.6B-Base)              |
| ---------------------- | ----------------------- | ------- | -------------------------------------------- |
| SFT + LoRA             | [`sft/`](sft/README.md) | Done    | 48.9% (base, 5-shot) → **68.0%** (zero-shot) |
| GRPO                   | `grpo/`                 | Planned |                                              |
| DPO                    | `dpo/`                  | Planned |                                              |
| PPO                    | `ppo/`                  | Planned |                                              |
| On-policy distillation | `opd/`                  | Planned |                                              |

## Principles

- **Train it yourself.** Every directory produces its own weights from a base model;
  nothing is downloaded pre-tuned.
- **Fair baseline.** A base model is evaluated with few-shot prompting; zero-shot it
  does not know the answer format, and its score would measure format, not ability.
  Trained models are evaluated zero-shot and compared against that number.
- **Same ruler everywhere.** Full test set, greedy decoding, one answer-extraction
  rule, and paired significance tests. Later stages (GRPO, DPO, ...) start from the
  SFT model and reuse `sft/evaluate.py`.
- **Small enough to read.** Each method is a handful of short scripts and one config;
  MLX-LM's training loop is reused rather than reimplemented.

## Environment

- Apple Silicon Mac (native arm64), macOS 14 or newer.
- Python 3.14, pinned in `.python-version`; managed with [uv](https://docs.astral.sh/uv/).
- [MLX](https://github.com/ml-explore/mlx) and [MLX-LM](https://github.com/ml-explore/mlx-lm)
  for inference and LoRA training. `uv.lock` pins the resolved versions.

```bash
uv sync --locked
uv run python -c 'import mlx.core as mx; print("Metal available:", mx.metal.is_available())'
```

`uv run` uses the project virtual environment; there is no need to activate it.

## Layout

```
sft/            supervised fine-tuning (data prep, distillation, training, evaluation, demo)
pyproject.toml  shared dependencies for every method
```
