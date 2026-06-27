"""
Example 03 — validate your data before you pay to train on it. OFFLINE, FREE.
============================================================================

A fine-tune job is slow and (on a real provider) costs money. The cheapest way to
not waste either is to check the dataset first. This runs every offline validation
check on the hand-made set: schema, duplicates, class balance, and a token + cost
estimate — then does it again on a deliberately BROKEN set so you can see the
checks catch real problems.

    python examples/03_validate_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune import load_jsonl, validate_dataset
from finetune.dataset import ChatExample, Message

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "datasets", "support_train.jsonl")


def show(report) -> None:
    print(report.summary())
    for w in report.warnings:
        print(f"  ! warning: {w}")
    for e in report.errors:
        print(f"  x error:   {e}")
    print(f"  => {'OK to train' if report.ok else 'BLOCKED: fix the errors first'}\n")


# 1. The real dataset — should pass clean.
print("=== Validating the hand-made training set ===")
examples = load_jsonl(TRAIN)
report = validate_dataset(examples, model="gpt-4o-mini", epochs=3)
show(report)

cost = __import__("finetune").estimate_cost(examples, model="gpt-4o-mini", epochs=3)
print(f"Estimated OpenAI training cost (gpt-4o-mini, 3 epochs): ${cost:.4f}")
print("(On PROVIDER=mock the cost is $0 — that's the whole point of practicing here.)\n")

# 2. A deliberately broken dataset — watch the checks fire.
print("=== Validating a deliberately broken set ===")
SYS = "You are a triage bot."
broken = [
    # missing assistant target (last turn is the user)
    ChatExample([Message("system", SYS), Message("user", "help me")]),
    # empty assistant content
    ChatExample([Message("system", SYS), Message("user", "refund?"), Message("assistant", "  ")]),
    # bad role
    ChatExample([Message("system", SYS), Message("bot", "hi"), Message("assistant", "category: other | reply: hi")]),
    # an exact duplicate of a good line, three times, to trip dedup + imbalance
]
good = ChatExample([Message("system", SYS), Message("user", "cancel plan"),
                    Message("assistant", "category: billing | reply: Cancel under Billing > Plan.")])
broken += [good, good, good]
report = validate_dataset(broken, model="gpt-4o-mini", epochs=3)
show(report)

print(
    "The lesson: almost every 'my fine-tune did nothing' story is a dataset problem "
    "a ten-line check would have caught. Validate first; it's free and it saves the "
    "slow, paid round-trip of training on bad data."
)
