"""
Example 07: hyperparameters and reading the loss curve. OFFLINE, FREE.

You don't need many knobs to fine-tune, and the defaults are usually fine. But
three matter enough to understand, and reading the training loss curve is how you
tell whether they were right.

  * n_epochs: how many times the model sees the whole dataset. Too few and it
    hasn't learned the behavior; too many and it OVERFITS (memorizes the training
    examples, generalizes worse). The tell: validation loss stops dropping, or
    rises, while training loss keeps falling.
  * learning_rate_multiplier: how big each update step is. Higher descends faster
    but can overshoot and get noisy/unstable; lower is steadier but slower.
  * batch_size: how many examples per update. Mostly a speed/stability tradeoff;
    leave it on auto unless you have a reason.

This script renders the (simulated) loss curve the mock fabricates from the
hyperparameters, for a few settings, so you can SEE these tradeoffs offline. The
curve is illustrative, but the shapes and what they mean are real.

    python examples/07_hyperparameters.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune import load_jsonl, mock_tuner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "datasets", "support_train.jsonl")


def sparkline(values: list[float]) -> str:
    """A tiny inline chart of the loss curve using block characters."""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(blocks[min(len(blocks) - 1, int((v - lo) / span * (len(blocks) - 1)))] for v in values)


def run(hp: dict) -> None:
    file_id = mock_tuner.upload_training_file(TRAIN)
    job = mock_tuner.create_job(file_id, hyperparameters=hp)
    while not job.is_done():
        job.poll()
    curve = job.loss_curve
    label = f"epochs={hp.get('n_epochs', 3)}, lr_mult={hp.get('learning_rate_multiplier', 1.0)}"
    print(f"{label:<32}  {sparkline(curve)}  final loss {curve[-1]:.3f}  ({len(curve)} steps)")


def main() -> int:
    n = len(load_jsonl(TRAIN))
    print(f"Training set: {n} examples. Simulated loss curves (the SHAPE is the lesson):\n")

    run({"n_epochs": 1, "learning_rate_multiplier": 1.0})
    run({"n_epochs": 3, "learning_rate_multiplier": 1.0})
    run({"n_epochs": 8, "learning_rate_multiplier": 1.0})
    print()
    run({"n_epochs": 3, "learning_rate_multiplier": 0.3})
    run({"n_epochs": 3, "learning_rate_multiplier": 2.0})

    print(
        "\nHow to read it:\n"
        "  * More epochs -> more steps and a lower final TRAINING loss, but the\n"
        "    real question is the VALIDATION loss. When val loss flattens or rises\n"
        "    while train loss keeps dropping, you're overfitting: stop earlier.\n"
        "  * A higher learning-rate multiplier descends faster but gets noisier;\n"
        "    a lower one is smoother but slower to converge.\n"
        "  * The curve is diagnostic, not the goal. The goal is the held-out result\n"
        "    in Section 7. A gorgeous loss curve that doesn't win the eval is a\n"
        "    model that overfit your training set."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
