"""
hands_on/finetune_run.py — the capstone: the whole fine-tune, end to end.
=========================================================================

Everything in the repo, wired into one command that does the real workflow:

    validate  ->  tune  ->  eval-gate vs. baseline  ->  ship ONLY if it wins

That last step is the discipline the whole repo is about: a fine-tune ships only
when it provably beats the base model on a held-out set. If it doesn't, the gate
says so and exits non-zero — the same shape as a CI eval gate.

It runs entirely offline on PROVIDER=mock (the default): validate the hand-made
set, simulate the tune, evaluate base vs. tuned on the held-out set, and gate. To
run the REAL paid OpenAI job instead, set PROVIDER=openai and pass --real (you'll
get a cost warning and a confirmation prompt).

    # Offline, free, the full arc:
    python hands_on/finetune_run.py

    # Point at a different training file (e.g. the distilled one from Section 9):
    python hands_on/finetune_run.py --train datasets/support_distilled.jsonl

    # Require the tuned model to clear a minimum win-rate to "ship":
    python hands_on/finetune_run.py --min-winrate 0.6

    # The real, PAID path (opt-in, confirmed):
    PROVIDER=openai python hands_on/finetune_run.py --real
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

import finetune
from finetune import (
    SUPPORT_SYSTEM,
    accuracy_on,
    load_jsonl,
    mock_tuner,
    providers,
    validate_dataset,
    win_rate,
)

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN = os.path.join(ROOT, "datasets", "support_train.jsonl")
DEFAULT_EVAL = os.path.join(ROOT, "datasets", "support_eval.jsonl")

HYPERPARAMS = {"n_epochs": 3, "learning_rate_multiplier": 1.0, "suffix": "support"}


def step(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")
    print("-" * 60)


def tune_mock(train_path: str) -> str:
    file_id = mock_tuner.upload_training_file(train_path)
    job = mock_tuner.create_job(file_id, hyperparameters=HYPERPARAMS)
    print(f"job {job.id} created")
    while not job.is_done():
        print(f"  ... {job.poll()}")
    print(f"succeeded — trained_tokens={job.trained_tokens}")
    return job.fine_tuned_model or ""


def tune_real(train_path: str) -> str:
    import time

    model = providers.tunable_model()
    examples = load_jsonl(train_path)
    est = finetune.estimate_cost(examples, model=model, epochs=HYPERPARAMS["n_epochs"])
    print("!" * 64)
    print("!! REAL OPENAI FINE-TUNE — COSTS REAL MONEY AND TAKES A WHILE.")
    print(f"!! base={model}  rough estimate ${est:.4f}  (verify on OpenAI's pricing page)")
    print("!" * 64)
    if input("Type 'yes, charge me' to proceed: ").strip() != "yes, charge me":
        sys.exit("Aborted — nothing uploaded, nothing charged.")
    file_id = providers.openai_upload_training_file(train_path)
    job_id = providers.openai_create_job(file_id, model=model, hyperparameters={"n_epochs": HYPERPARAMS["n_epochs"]})
    print(f"job {job_id} created; polling...")
    while True:
        info = providers.openai_poll_job(job_id)
        print(f"  ... {info['status']}")
        if info["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(30)
    if info["status"] != "succeeded":
        sys.exit(f"Job ended as {info['status']}. Nothing to ship.")
    return info["fine_tuned_model"] or ""


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end fine-tune with an eval gate.")
    parser.add_argument("--train", default=DEFAULT_TRAIN, help="training JSONL file")
    parser.add_argument("--eval", default=DEFAULT_EVAL, help="held-out eval JSONL file")
    parser.add_argument("--min-winrate", type=float, default=0.5,
                        help="tuned must win at least this fraction to ship (default 0.5)")
    parser.add_argument("--real", action="store_true", help="run the REAL paid OpenAI job (opt-in)")
    args = parser.parse_args()

    train_path = args.train if os.path.isabs(args.train) else os.path.join(ROOT, args.train)
    eval_path = args.eval if os.path.isabs(args.eval) else os.path.join(ROOT, args.eval)

    providers.ensure_ready(for_tuning=True)
    print(f"Provider: {providers.describe()}")

    # 1. Validate — never train on a dataset you haven't checked.
    step(1, "Validate the training data (offline, free)")
    examples = load_jsonl(train_path)
    report = validate_dataset(examples, model=providers.base_model(), epochs=HYPERPARAMS["n_epochs"])
    print(report.summary())
    for w in report.warnings:
        print(f"  ! {w}")
    if not report.ok:
        for e in report.errors:
            print(f"  x {e}")
        print("\nGATE: dataset has errors — fix them before training. Exiting.")
        return 1

    # 2. Tune — mock by default, real only with --real on PROVIDER=openai.
    step(2, "Fine-tune")
    if args.real:
        if providers.provider_name() != "openai":
            sys.exit("--real requires PROVIDER=openai.")
        tuned = tune_real(train_path)
    else:
        if providers.provider_name() != "mock":
            print(f"(PROVIDER={providers.provider_name()} but no --real — using the free mock.)")
        tuned = tune_mock(train_path)
    base = providers.base_model()
    print(f"base model:  {base}")
    print(f"tuned model: {tuned}")

    # 3. Eval-gate — base vs. tuned on the held-out set.
    step(3, "Evaluate base vs. tuned on the held-out set")
    held_out = load_jsonl(eval_path)
    base_acc = accuracy_on(held_out, model=base, system=SUPPORT_SYSTEM)
    tuned_acc = accuracy_on(held_out, model=tuned, system=SUPPORT_SYSTEM)
    wins = win_rate(held_out, model_a=base, model_b=tuned, system=SUPPORT_SYSTEM)
    n = len(held_out)
    winrate = wins["B"] / n if n else 0.0
    print(f"accuracy   base {base_acc.accuracy:.0%}  ->  tuned {tuned_acc.accuracy:.0%}  ({tuned_acc.accuracy - base_acc.accuracy:+.0%})")
    print(f"win-rate   tuned wins {wins['B']}/{n} = {winrate:.0%}  (ties {wins['TIE']}, base {wins['A']})")

    # 4. Ship only if it wins.
    step(4, "Decision")
    improved_acc = tuned_acc.accuracy > base_acc.accuracy
    cleared_winrate = winrate >= args.min_winrate
    if improved_acc and cleared_winrate:
        print(f"SHIP IT: tuned improved accuracy AND cleared the {args.min_winrate:.0%} "
              f"win-rate bar. This is a defensible improvement, not a vibe.")
        return 0
    print(f"DO NOT SHIP: improved_accuracy={improved_acc}, "
          f"win-rate {winrate:.0%} vs bar {args.min_winrate:.0%}.")
    print("The base model is as good or better on the held-out set. Improve the "
          "dataset (more/better examples, fix imbalance) and run again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
