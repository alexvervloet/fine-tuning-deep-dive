"""
finetune/evaluate.py: did the fine-tune actually help? Prove it.

This is the punchline of the whole repo. A fine-tune that you *think* is better
is worth nothing; the only thing worth shipping is one you can *show* beat your
baseline on data the training never saw. This module is the from-scratch eval
that does that, the same method as the evals deep dive (#5 in the series),
pointed at a single decision: base model vs. fine-tuned model.

It offers the two comparisons that matter for fine-tuning:

  1. accuracy on a held-out set, when there's a right answer (e.g. the correct
     category), run both models and count exact matches. The headline number.
  2. win-rate vs. baseline, when "better" is a matter of quality/format (no
     single right string), show a judge both answers and tally which wins. This
     is the pairwise win-rate from evals/07_pairwise.py. Here the judge is a
     simple offline rubric scorer so the punchline runs free; in production you'd
     use an LLM-as-judge (and judge both orderings to dodge position bias).

Keep the eval set HELD OUT: examples the model was never trained on. Re-using
training data here is the classic way to fool yourself into shipping a regression.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import providers
from .dataset import ChatExample


@dataclass
class EvalResult:
    """One model's score on the held-out set."""

    model: str
    n: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


def _predicted_category(text: str) -> str:
    """Pull the category label out of an assistant reply, if present.

    Our house format is 'category: <label> | reply: ...'. The base model doesn't
    follow it; the fine-tuned model (should), so even extracting the label is a
    signal that the fine-tune taught the format.
    """
    if "category:" in text:
        after = text.split("category:", 1)[1].strip()
        return after.split("|", 1)[0].strip().lower()
    return ""


def accuracy_on(examples: list[ChatExample], *, model: str, system: str) -> EvalResult:
    """Run `model` over the held-out set and score category accuracy.

    For each example we generate from (system, user) and compare the predicted
    category to the gold category in the example's assistant turn. Exact-label
    accuracy is the simplest honest "is it right?" for a classification task.
    """
    correct = 0
    for ex in examples:
        gold = _predicted_category(ex.last_assistant() or "")
        resp = providers.generate(system, ex.last_user() or "", model=model)
        pred = _predicted_category(resp.text)
        if gold and pred == gold:
            correct += 1
    return EvalResult(model=model, n=len(examples), correct=correct)


def _format_score(text: str) -> int:
    """A tiny offline 'judge': how well does a reply follow the house format?

    Rewards the structured 'category: ... | reply: ...' shape and a non-empty
    reply; penalizes the base model's rambling. Stands in for an LLM-as-judge so
    the win-rate runs offline. In production, replace with providers.generate()
    asked to pick the better answer (and judge both orderings; see evals #9).
    """
    score = 0
    if "category:" in text:
        score += 2
    if "| reply:" in text or "reply:" in text:
        score += 1
    if 5 <= len(text) <= 200:
        score += 1
    return score


def win_rate(
    examples: list[ChatExample], *, model_a: str, model_b: str, system: str
) -> dict[str, int]:
    """Pairwise win-rate of model_b over model_a on the held-out set.

    Returns {"A": .., "B": .., "TIE": ..}. By convention A is the baseline (base
    model) and B is the challenger (fine-tuned), so a high B count is the result
    you're hoping for. This mirrors evals/07_pairwise.py; the rubric (here, the
    format scorer) is the most important part: it defines what "better" means.
    """
    wins = {"A": 0, "B": 0, "TIE": 0}
    for ex in examples:
        user = ex.last_user() or ""
        ans_a = providers.generate(system, user, model=model_a).text
        ans_b = providers.generate(system, user, model=model_b).text
        sa, sb = _format_score(ans_a), _format_score(ans_b)
        if sb > sa:
            wins["B"] += 1
        elif sa > sb:
            wins["A"] += 1
        else:
            wins["TIE"] += 1
    return wins
