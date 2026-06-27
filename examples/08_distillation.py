"""
Example 08 — distillation: train a small model on a strong model's outputs.
==========================================================================

The most common production shape of fine-tuning isn't hand-labeling data — it's
DISTILLATION: take a big, expensive, smart model (the TEACHER) that already does
your task well, run it over a pile of inputs, and use its answers as the training
data for a small, cheap, fast model (the STUDENT). The student learns to imitate
the teacher on your task — at a fraction of the per-call cost and latency.

Why it's so common: the labels write themselves. You don't sit there writing
assistant turns; the teacher generates them. That's what makes a distillation set
of hundreds or thousands of examples cheap to build.

This script BUILDS a distillation dataset (offline on the mock: the 'teacher' is
just the mock answering in the house format; on a real provider, point the teacher
at a strong model like gpt-4o or claude). It then validates the result so you can
see it's a normal training file — ready to feed into Section 5's tune step.

    python examples/08_distillation.py
    PROVIDER=openai python examples/08_distillation.py   # teacher = a real model
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from finetune import SUPPORT_SYSTEM, distillation_example, generate, providers, validate_dataset, write_jsonl

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "datasets", "support_distilled.jsonl")

# Unlabeled inputs — in real life, a sample of your production traffic. Notice
# there are no answers here; the teacher will produce them.
UNLABELED = [
    "I can't get into my account, the password reset email never came",
    "can I get my money back for the plan I just bought?",
    "the site throws an error 503 when I try to save",
    "how do I add my colleague to the workspace?",
    "is there a way to see past invoices?",
    "the desktop app freezes on the loading screen",
    "do you have a referral program?",
    "I need to switch my plan to annual billing",
]

# In real distillation the TEACHER is a strong model; the STUDENT is a small one.
# A perfect teacher prompt and a known-good format is what the student inherits.
TEACHER_MODEL = providers.base_model()  # mock: the same tiny model, but answering well below


def teacher_answer(user: str) -> str:
    """Get a teacher answer. On the mock we synthesize an ideal house-format answer
    (the mock's base model can't), so the demo is self-contained offline. On a real
    provider, this is a single generate() call to your strong teacher model."""
    if providers.provider_name() == "mock":
        # A stand-in "teacher": rule-based ideal labels so the offline demo has a
        # high-quality teacher to distill FROM. On openai/claude, delete this branch.
        u = user.lower()
        if "password" in u or "account" in u or "workspace" in u or "colleague" in u:
            cat = "account"
        elif "money" in u or "invoice" in u or "plan" in u or "billing" in u or "refund" in u:
            cat = "billing"
        elif "error" in u or "freezes" in u or "503" in u or "app" in u or "save" in u:
            cat = "technical"
        else:
            cat = "other"
        return f"category: {cat} | reply: Happy to help with that — here's the quickest path."
    return generate(SUPPORT_SYSTEM, user, model=TEACHER_MODEL).text


def main() -> int:
    providers.ensure_ready()
    print(f"Provider: {providers.describe()}")
    print(f"Teacher model: {TEACHER_MODEL}\n")

    print("Harvesting teacher answers into training examples...")
    examples = []
    for user in UNLABELED:
        ans = teacher_answer(user)
        examples.append(distillation_example(user, ans))
        print(f"  {user[:40]:<42} -> {ans}")

    write_jsonl(OUT, examples)
    print(f"\nWrote {len(examples)} distilled examples to {os.path.relpath(OUT, ROOT)}")

    report = validate_dataset(examples, model=providers.base_model())
    print("\nValidating the distilled set (it's a normal training file now):")
    print("  " + report.summary().replace("\n", "\n  "))

    print(
        "\nThe lesson: a distillation set is just a training set whose assistant\n"
        "turns came from a model instead of a human. Feed support_distilled.jsonl\n"
        "into Section 5 to tune the small STUDENT, then Section 7 to check it kept\n"
        "the teacher's quality at the student's cost. Caveat: the student can only\n"
        "be as good as the teacher, and it inherits the teacher's mistakes — so you\n"
        "still gate on a held-out eval, not on 'it matched the teacher.'"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
