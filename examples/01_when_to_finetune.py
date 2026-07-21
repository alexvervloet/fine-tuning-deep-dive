"""
Example 01: when to fine-tune (vs. prompt, few-shot, RAG). OFFLINE, FREE.

Before any code, the most valuable skill in fine-tuning is knowing when NOT to.
Fine-tuning is the slow, expensive, provider-specific option, and reaching for it
first is the most common and costly mistake. This is the hands-on version of the
RAG deep dive's "RAG, fine-tuning, or something else?" section.

The one rule that resolves most cases:

    RAG / long context change WHAT'S IN THE CONTEXT (knowledge).
    Fine-tuning changes HOW THE MODEL BEHAVES BY DEFAULT (format, tone, a skill).
    Tools/agents change WHAT THE MODEL CAN DO (capability).

So: if you need facts that change or must be cited -> RAG. If you need the same
behavior/format every time, or lower cost/latency on a fixed task -> fine-tune.
If the model needs to act or fetch live data -> tools.

This script is a tiny decision function. It takes a short description of your
problem and recommends an approach, with the reasoning. No model, no key, no cost.

    python examples/01_when_to_finetune.py
    python examples/01_when_to_finetune.py "answer questions about our internal wiki, with citations"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keyword signals for each approach. This is deliberately simple: the point is the
# *reasoning*, which it prints, not a clever classifier.
_SIGNALS = {
    "RAG": (
        ["cite", "citation", "source", "document", "wiki", "knowledge base", "changes",
         "up to date", "private docs", "look up", "facts"],
        "You need KNOWLEDGE that changes / is private / must be cited. Retrieve it at "
        "request time; update the corpus, not the model. (See the RAG deep dive.)",
    ),
    "Fine-tuning": (
        ["format", "tone", "style", "consistent", "every time", "house style", "voice",
         "classify", "structured output", "cheaper", "latency", "faster", "smaller model"],
        "You need a consistent BEHAVIOR/format, or lower cost/latency on a fixed, "
        "high-volume task. That's taught by examples, which is what training adjusts.",
    ),
    "Long context": (
        ["small", "fits in the prompt", "one document", "short", "paste"],
        "If the material fits in the prompt, just include it; retrieval and training "
        "are machinery you don't need yet.",
    ),
    "Tools / agents": (
        ["act", "action", "live data", "real-time", "call an api", "book", "send", "fetch"],
        "The gap is CAPABILITY, not knowledge or behavior, so give the model tools. "
        "(See the agents deep dive.)",
    ),
    "Better prompt / few-shot": (
        ["prompt", "instructions", "few-shot", "examples in the prompt", "not sure", "try"],
        "Start here. A sharper prompt or a few in-context examples is free, instant, "
        "and fixes a surprising amount. Exhaust this BEFORE you fine-tune.",
    ),
}


def recommend(problem: str) -> tuple[str, str]:
    p = problem.lower()
    best, best_score = None, 0
    for approach, (keywords, _why) in _SIGNALS.items():
        score = sum(1 for kw in keywords if kw in p)
        if score > best_score:
            best, best_score = approach, score
    if best is None:
        best = "Better prompt / few-shot"  # the safe default: always try this first
    return best, _SIGNALS[best][1]


CASES = [
    "answer questions about our internal wiki, with citations to the source",
    "always reply in our strict JSON format, the same way every time",
    "classify support tickets cheaply at high volume",
    "the model should be able to book a meeting on my calendar",
    "summarize this one short contract I'll paste in",
]


def main() -> int:
    if len(sys.argv) > 1:
        cases = [" ".join(sys.argv[1:])]
    else:
        cases = CASES
        print("No problem given, so running the built-in examples.\n"
              "Try:  python examples/01_when_to_finetune.py \"your problem here\"\n")

    for problem in cases:
        approach, why = recommend(problem)
        print(f"Problem:  {problem}")
        print(f"  -> {approach}")
        print(f"     {why}\n")

    print("Two rules of thumb that survive every case:")
    print("  1. Don't fine-tune FIRST. Try a better prompt, few-shot, then RAG. "
          "Fine-tuning can't add knowledge that changes, and it costs real money.")
    print("  2. Don't decide by vibes. The ONLY way to know fine-tuning beat your "
          "baseline is to measure both on the same held-out set (Section 7).")
    print("\nA common production shape is BOTH: fine-tune for format, RAG for facts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
