"""
Example 06: did it actually help? Base vs. fine-tuned on a held-out set.

THE PUNCHLINE. A fine-tune you *think* is better is worth nothing; the only thing
worth shipping is one you can *prove* beat your baseline on data the training
never saw. This is the evals deep dive's method (#5 in the series) pointed at one
decision: base model vs. fine-tuned model, on datasets/support_eval.jsonl, a
HELD-OUT set, none of which was in training.

Two comparisons, both from finetune/evaluate.py:

  1. accuracy: % of held-out examples where the predicted category matches gold.
                 The headline "is it right?" number.
  2. win-rate: pairwise, the way evals/07_pairwise.py does it. For each example
                 show a judge both answers and tally which is better. Here the
                 'judge' is an offline format rubric so this runs free; in
                 production you'd use an LLM-as-judge (judging both orderings to
                 dodge position bias; see evals #9).

    python examples/06_did_it_help.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from finetune import SUPPORT_SYSTEM, accuracy_on, load_jsonl, mock_tuner, providers, win_rate

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "datasets", "support_train.jsonl")
EVAL = os.path.join(ROOT, "datasets", "support_eval.jsonl")


def get_finetuned_model() -> str:
    if providers.provider_name() == "mock":
        file_id = mock_tuner.upload_training_file(TRAIN)
        job = mock_tuner.create_job(file_id, hyperparameters={"suffix": "support"})
        while not job.is_done():
            job.poll()
        return job.fine_tuned_model or ""
    ft = os.getenv("FT_MODEL", "").strip()
    if not ft:
        sys.exit("Set FT_MODEL to your fine-tuned model id, or use PROVIDER=mock.")
    return ft


def main() -> int:
    providers.ensure_ready()
    print(f"Provider: {providers.describe()}\n")

    held_out = load_jsonl(EVAL)
    base = providers.base_model()
    tuned = get_finetuned_model()
    print(f"Held-out eval set: {len(held_out)} examples (none seen in training)\n")

    # 1. Accuracy: the headline number.
    base_acc = accuracy_on(held_out, model=base, system=SUPPORT_SYSTEM)
    tuned_acc = accuracy_on(held_out, model=tuned, system=SUPPORT_SYSTEM)
    print("Category accuracy on the held-out set:")
    print(f"  base:  {base_acc.correct}/{base_acc.n} = {base_acc.accuracy:.0%}")
    print(f"  tuned: {tuned_acc.correct}/{tuned_acc.n} = {tuned_acc.accuracy:.0%}")
    delta = tuned_acc.accuracy - base_acc.accuracy
    print(f"  delta: {delta:+.0%}\n")

    # 2. Win-rate: pairwise, B (tuned) vs A (base).
    wins = win_rate(held_out, model_a=base, model_b=tuned, system=SUPPORT_SYSTEM)
    n = len(held_out)
    print("Pairwise win-rate (A=base, B=fine-tuned), rubric = follows house format:")
    print(f"  base wins:  {wins['A']}/{n}")
    print(f"  tuned wins: {wins['B']}/{n} = {wins['B'] / n:.0%}")
    print(f"  ties:       {wins['TIE']}/{n}\n")

    # 3. The gate. This is the decision, not a vibe.
    helped = tuned_acc.accuracy > base_acc.accuracy or wins["B"] > wins["A"]
    if helped:
        print("VERDICT: the fine-tune beat the baseline -> worth shipping.")
    else:
        print("VERDICT: no improvement over baseline -> do NOT ship it. "
              "Fix the dataset and try again.")
    print(
        f"\nThat verdict is the whole point. 'It looks better' ships regressions; a "
        f"concrete '{delta:+.0%} accuracy, {wins['B'] / n:.0%} win-rate on the held-out "
        f"set' is a result you can defend. Same method as the evals dive; only the "
        f"thing compared changed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
