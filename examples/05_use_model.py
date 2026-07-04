"""
Example 05 — using the fine-tuned model. OFFLINE on the mock.
============================================================

Once a job succeeds, the provider hosts your model under a new id (e.g.
'ft:gpt-4o-mini:...'). Using it is just a normal chat call with that id as the
model — no special API. This script tunes on the mock (fast and free), then asks
both the BASE model and the FINE-TUNED model the same questions, side by side, so
you can SEE the behavior change.

Watch what changes: the base model handles the one or two categories it happens to
know (the password one) but rambles and ignores the house format on the rest; the
fine-tuned model answers EVERY one in the trained 'category: ... | reply: ...'
shape. That's the whole idea — fine-tuning changed HOW it behaves, taught only by
examples.

    python examples/05_use_model.py

(On a real provider you'd skip the tune step and pass your saved model id via the
FT_MODEL env var. This script uses that if it's set and PROVIDER != mock.)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from finetune import SUPPORT_SYSTEM, generate, mock_tuner, providers

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "datasets", "support_train.jsonl")

QUESTIONS = [
    "I can't sign in, I think I forgot my password",  # a category the base knows -> it formats this one
    "the app keeps crashing when I open it",           # the base rambles; the tuned model gets it
    "how do I turn on two-factor login?",              # a paraphrase the base fumbles, the tuned model nails
]


def get_finetuned_model() -> str:
    """On the mock, tune now (free) to get a model id. On a real provider, read
    the id you saved from Example 04 out of the FT_MODEL env var."""
    if providers.provider_name() == "mock":
        file_id = mock_tuner.upload_training_file(TRAIN)
        job = mock_tuner.create_job(file_id, hyperparameters={"suffix": "support"})
        while not job.is_done():
            job.poll()
        return job.fine_tuned_model or ""
    ft = os.getenv("FT_MODEL", "").strip()
    if not ft:
        sys.exit("Set FT_MODEL to your fine-tuned model id (from Example 04 --real), "
                 "or use PROVIDER=mock to run offline.")
    return ft


def main() -> int:
    providers.ensure_ready()
    print(f"Provider: {providers.describe()}\n")

    base = providers.base_model()
    tuned = get_finetuned_model()
    print(f"Base model:       {base}")
    print(f"Fine-tuned model: {tuned}\n")

    for q in QUESTIONS:
        base_ans = generate(SUPPORT_SYSTEM, q, model=base).text
        tuned_ans = generate(SUPPORT_SYSTEM, q, model=tuned).text
        print(f"Q: {q}")
        print(f"  base:  {base_ans}")
        print(f"  tuned: {tuned_ans}\n")

    print(
        "Same prompt, same model family — only the training changed the behavior. "
        "Seeing the difference is reassuring, but it's not proof. Section 7 turns "
        "this into a number on a held-out set: did it ACTUALLY help?"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
