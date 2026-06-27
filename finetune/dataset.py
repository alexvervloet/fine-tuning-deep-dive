"""
finetune/dataset.py — the chat JSONL format, and reading/splitting it.
======================================================================

THE DATASET IS THE PRODUCT. A fine-tune is only ever as good as the examples you
show it — the model learns the *behavior demonstrated*, format quirks and all.
So this file is about one thing: the file format your examples live in, and the
basic operations on it (load, split, write).

The format is the same one OpenAI's fine-tuning API expects: JSON Lines, one
training example per line, each example a chat conversation:

    {"messages": [
       {"role": "system",    "content": "You are Acme's support classifier."},
       {"role": "user",      "content": "i can't log in, forgot my password"},
       {"role": "assistant", "content": "category: account | reply: ..."}
    ]}

Why this shape? Because fine-tuning teaches the model to produce the *assistant*
turn given the *system + user* turns. Each line is one worked example of "when
you see input like this, behave like this." The system prompt should match the
one you'll use at inference time — the model learns to associate that instruction
with that behavior.

We model a training example as a `ChatExample` (a list of `Message`s) so the rest
of the repo can reason about it without re-parsing JSON everywhere.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass


@dataclass
class Message:
    """One turn in a conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatExample:
    """One training example: a conversation ending in the assistant turn the
    model should learn to produce."""

    messages: list[Message]

    @classmethod
    def from_dict(cls, row: dict) -> "ChatExample":
        msgs = [Message(m["role"], m["content"]) for m in row["messages"]]
        return cls(messages=msgs)

    def to_dict(self) -> dict:
        return {"messages": [m.to_dict() for m in self.messages]}

    # Convenience accessors the validator and eval lean on.
    def system(self) -> str | None:
        for m in self.messages:
            if m.role == "system":
                return m.content
        return None

    def last_user(self) -> str | None:
        users = [m.content for m in self.messages if m.role == "user"]
        return users[-1] if users else None

    def last_assistant(self) -> str | None:
        assts = [m.content for m in self.messages if m.role == "assistant"]
        return assts[-1] if assts else None


def load_jsonl(path: str) -> list[ChatExample]:
    """Load training examples from a chat JSONL file (one object per line)."""
    examples: list[ChatExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {line_no} is not valid JSON: {e}") from e
            examples.append(ChatExample.from_dict(row))
    return examples


def write_jsonl(path: str, examples: list[ChatExample]) -> None:
    """Write examples back out as chat JSONL (used by the data-build helpers)."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def train_val_split(
    examples: list[ChatExample], *, val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[ChatExample], list[ChatExample]]:
    """Split examples into (train, val). Shuffles with a fixed seed so the split
    is reproducible — you want the same held-out set every run so the eval number
    is comparable across experiments.

    The validation set is what the provider scores during training (the val loss
    curve in Section 8). It is NOT the same as your held-out *eval* set in
    Section 7 — keep an eval set the model and the training process never touch,
    so "did it help?" is an honest, out-of-sample question.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1 (exclusive).")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val
