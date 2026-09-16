"""Compare base and LoRA responses to a random GSM8K test example."""

import argparse
import random

from evaluate import ADAPTER_PATH, DATA_DIR, MODEL_NAME, build_prompt, clean_prediction, load_jsonl
from huggingface_hub.utils import disable_progress_bars
from mlx_lm import load, stream_generate
from rich.console import Console
from rich.panel import Panel

MAX_TOKENS = 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="Also save the output as an SVG image, e.g. demo.svg.")
    args = parser.parse_args()

    disable_progress_bars()  # hide the "Fetching 7 files" bars printed on model load

    example = random.choice(load_jsonl(DATA_DIR / "test.jsonl"))
    shots = load_jsonl(DATA_DIR / "train.jsonl", 5)
    console = Console(markup=False, highlight=False, record=bool(args.save), width=100)
    console.rule("GSM8K · Base (5-shot) vs LoRA (0-shot)")
    console.print(Panel(example["question"], title="Question", border_style="cyan"))
    console.print()

    # Same setup as the evaluation: the base model gets five worked examples
    # from the training set, the fine-tuned model gets the question alone.
    for name, adapter_path, prompt_shots, color in [
        ("Base model, 5-shot", None, shots, "blue"),
        ("LoRA model, 0-shot", str(ADAPTER_PATH), [], "magenta"),
    ]:
        prompt = build_prompt(example["question"], prompt_shots)
        with console.status(f"Generating {name} response..."):
            model, tokenizer = load(MODEL_NAME, adapter_path=adapter_path)
            answer = ""
            truncated = False
            for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS):
                answer += response.text
                truncated = response.finish_reason == "length"

        console.print(
            Panel(
                clean_prediction(answer),
                title=name,
                border_style=color,
                subtitle=f"Reached {MAX_TOKENS} token limit" if truncated else None,
            )
        )
        console.print()

    console.print(Panel(example["answer"], title="Reference answer", border_style="green"))

    if args.save:
        console.save_svg(args.save, title="uv run python demo.py")


if __name__ == "__main__":
    main()
