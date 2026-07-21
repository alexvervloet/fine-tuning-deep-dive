"""
Example 02: the dataset is the product: the chat JSONL format. OFFLINE, FREE.

A fine-tune learns the behavior your examples demonstrate, so the examples ARE
the product. This script shows the format those examples live in (the same one
OpenAI's fine-tuning API expects), builds one from scratch, loads the hand-made
training set, and splits it into train/val.

The format is JSON Lines: one conversation per line, each ending in the assistant
turn the model should learn to produce.

    python examples/02_dataset_format.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from finetune import load_jsonl, make_example, train_val_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "datasets", "support_train.jsonl")

# 1. Build ONE example from its parts, to see the shape.
print("One training example, built from scratch:")
ex = make_example(
    user="i can't log in, forgot my password",
    category="account",
    reply="Reset it from Settings > Security > Reset password.",
)
print(json.dumps(ex.to_dict(), indent=2))
print(
    "\nNote: the model learns to produce the *assistant* turn given the system + "
    "user turns. Each line is one worked example of 'input like this -> behave "
    "like this'. The system prompt should match the one you'll use at inference."
)

# 2. Load the real hand-made dataset.
examples = load_jsonl(TRAIN)
print(f"\nLoaded {len(examples)} examples from {os.path.relpath(TRAIN, ROOT)}.")
print("First user message: ", examples[0].last_user())
print("Its taught reply:   ", examples[0].last_assistant())

# 3. Split into train / validation (reproducible with a fixed seed).
train, val = train_val_split(examples, val_fraction=0.2, seed=0)
print(f"\nSplit: {len(train)} train, {len(val)} val (val_fraction=0.2, seed=0).")
print(
    "The VAL set is what the provider scores while training (the loss curve in "
    "Section 8). It is NOT the held-out EVAL set from Section 7; keep a third set "
    "the training never touches, so 'did it help?' stays an honest out-of-sample test."
)

print(
    "\nThe lesson: there's no model magic here yet, just well-formed examples. "
    "Quality and consistency of these examples is ~all that determines your result."
)
