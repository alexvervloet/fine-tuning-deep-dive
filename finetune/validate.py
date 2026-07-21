"""
finetune/validate.py: check your data BEFORE you pay to train on it.

A fine-tune job is slow and (on a real provider) costs money. The single
cheapest way to not waste either is to validate the dataset first. Almost every
"my fine-tune did nothing / got worse" story traces back to a dataset problem a
ten-line check would have caught: a malformed line, the same example 40 times, a
class the model never sees, or a system prompt that drifts between examples.

So everything here runs OFFLINE and for free, and answers four questions:

  1. Is each line well-formed?   -> validate_schema()
  2. Are there duplicates?       -> find_duplicates()
  3. Is the dataset balanced?    -> class_balance()
  4. How big / how much?         -> estimate_tokens() + estimate_cost()

`validate_dataset()` runs all of them and returns a `ValidationReport` you can
print and gate on. The format rules mirror what OpenAI's fine-tuning endpoint
enforces, so a dataset that passes here is one the real API will accept.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .dataset import ChatExample

_VALID_ROLES = {"system", "user", "assistant"}

# Rough USD per 1M *training* tokens for OpenAI fine-tuning, by base model.
# Training cost ~= (tokens per example) x (number of examples) x (epochs).
# These are illustrative ballparks for teaching the *shape* of the cost: always
# check the provider's current pricing page before running a real job.
_TRAIN_USD_PER_1M = {
    "gpt-4o-mini": 3.00,
    "gpt-4o-mini-2024-07-18": 3.00,
    "mock-1": 0.0,  # the mock is free, on purpose
}


@dataclass
class ValidationReport:
    """The verdict on a dataset. `ok` is the one-line gate for the capstone."""

    n_examples: int
    errors: list[str] = field(default_factory=list)      # must-fix; block training
    warnings: list[str] = field(default_factory=list)    # smells; review them
    duplicates: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    total_tokens: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [f"{self.n_examples} examples, ~{self.total_tokens} tokens"]
        if self.label_counts:
            balance = ", ".join(f"{k}={v}" for k, v in sorted(self.label_counts.items()))
            lines.append(f"label balance: {balance}")
        if self.duplicates:
            lines.append(f"duplicates: {self.duplicates}")
        lines.append(f"errors: {len(self.errors)}  warnings: {len(self.warnings)}")
        return "\n".join(lines)


def approx_tokens(text: str) -> int:
    """A rough token count (~4 chars/token). Good enough to estimate cost offline;
    a real cost check would use the provider's tokenizer (e.g. tiktoken)."""
    return max(1, len(text) // 4)


def example_tokens(ex: ChatExample) -> int:
    """Approximate tokens for one example: every message counts (you pay to train
    on the whole conversation, not just the assistant turn)."""
    return sum(approx_tokens(m.content) for m in ex.messages)


def validate_schema(examples: list[ChatExample]) -> list[str]:
    """Return a list of format errors. Empty list == every example is well-formed.

    The rules a chat fine-tuning example must satisfy:
      - at least one user and exactly the conversation ending in an assistant turn
      - every role is system/user/assistant
      - no empty content
      - the LAST message is the assistant turn (that's the target to learn)
    """
    errors: list[str] = []
    for i, ex in enumerate(examples):
        where = f"example {i}"
        if not ex.messages:
            errors.append(f"{where}: has no messages")
            continue
        for j, m in enumerate(ex.messages):
            if m.role not in _VALID_ROLES:
                errors.append(f"{where}: message {j} has invalid role {m.role!r}")
            if not m.content or not m.content.strip():
                errors.append(f"{where}: message {j} ({m.role}) has empty content")
        if ex.messages[-1].role != "assistant":
            errors.append(f"{where}: last message must be the assistant turn (the target)")
        if not any(m.role == "user" for m in ex.messages):
            errors.append(f"{where}: needs at least one user message")
    return errors


def check_system_consistency(examples: list[ChatExample]) -> list[str]:
    """Warn if the system prompt isn't consistent across examples.

    Not an error, but if the system prompt drifts, the model can't reliably learn
    to associate *one* instruction with the behavior you want. Pick the system
    prompt you'll use at inference time and use it in every training example.
    """
    systems = {ex.system() for ex in examples}
    systems.discard(None)
    if len(systems) > 1:
        return [
            f"system prompt varies across examples ({len(systems)} distinct). "
            f"Use the same system prompt you'll use at inference time, everywhere."
        ]
    return []


def find_duplicates(examples: list[ChatExample]) -> int:
    """Count exact-duplicate examples (same full conversation).

    Duplicates aren't illegal, but they silently re-weight training toward the
    repeated example and inflate your token bill. A handful is fine; dozens of
    copies of one line usually means a data-building bug.
    """
    seen: Counter[str] = Counter()
    for ex in examples:
        key = "\n".join(f"{m.role}:{m.content}" for m in ex.messages)
        seen[key] += 1
    return sum(c - 1 for c in seen.values() if c > 1)


def class_balance(examples: list[ChatExample], *, label_prefix: str = "category:") -> dict[str, int]:
    """Count examples per class, for classification-style datasets.

    Our example dataset puts the label in the assistant turn as
    'category: <label> | reply: ...'. This pulls that label out and counts it, so
    you can see a class imbalance (e.g. 30 'account' vs 1 'billing') that would
    teach the model to over-predict the common class. Returns {} for free-form
    datasets where no label prefix is present.
    """
    counts: Counter[str] = Counter()
    for ex in examples:
        ans = ex.last_assistant() or ""
        if label_prefix in ans:
            after = ans.split(label_prefix, 1)[1].strip()
            label = after.split("|", 1)[0].strip()
            if label:
                counts[label] += 1
    return dict(counts)


def estimate_tokens(examples: list[ChatExample]) -> int:
    """Total approximate tokens across the dataset (one epoch)."""
    return sum(example_tokens(ex) for ex in examples)


def estimate_cost(examples: list[ChatExample], *, model: str, epochs: int = 3) -> float:
    """Estimate USD to train, before you commit. The mock is always $0.

    cost ~= total_tokens x epochs x price_per_token. This is the number that
    should make you pause and ask "is this fine-tune worth it, vs. a better
    prompt?", which is exactly Section 2's question.
    """
    per_1m = _TRAIN_USD_PER_1M.get(model, 0.0)
    total = estimate_tokens(examples) * epochs
    return total / 1_000_000 * per_1m


def validate_dataset(
    examples: list[ChatExample], *, model: str = "gpt-4o-mini", epochs: int = 3
) -> ValidationReport:
    """Run every check and bundle the result into one ValidationReport."""
    report = ValidationReport(n_examples=len(examples))
    if not examples:
        report.errors.append("dataset is empty")
        return report

    report.errors.extend(validate_schema(examples))
    report.warnings.extend(check_system_consistency(examples))
    report.duplicates = find_duplicates(examples)
    if report.duplicates:
        report.warnings.append(f"{report.duplicates} duplicate example(s); dedupe to save tokens")

    report.label_counts = class_balance(examples)
    if report.label_counts:
        counts = list(report.label_counts.values())
        if max(counts) > 3 * max(min(counts), 1):
            report.warnings.append(
                "class imbalance: the largest class is 3x+ the smallest. "
                "the model will over-predict the common class"
            )

    # OpenAI requires at least 10 training examples; warn well before that.
    if len(examples) < 10:
        report.warnings.append(
            f"only {len(examples)} examples; most providers want 10+ to even "
            f"accept the job, and dozens to hundreds to actually learn a behavior"
        )

    report.total_tokens = estimate_tokens(examples)
    return report
