"""
Example 10: preference tuning (DPO/RLHF): learning from comparisons. CONCEPTUAL.

Every example so far taught by DEMONSTRATION: show the model the one right answer
(`messages` ending in the assistant turn you want) and have it imitate. That's
supervised fine-tuning (SFT), and it's perfect when there *is* a single correct
output: a fixed format, a known label.

But some qualities don't have one right answer. "Be more concise." "Sound warmer."
"Refuse this kind of request more firmly." You can't write THE correct response 
but you can reliably say which of two responses is *better*. **Preference tuning**
learns from exactly that: pairs of (better, worse) answers to the same prompt.

  - RLHF (reinforcement learning from human feedback): the classic, train a
    separate "reward model" on human rankings, then optimize the model against it.
    Powerful, but a fiddly multi-stage pipeline.
  - DPO (direct preference optimization): the modern shortcut, skip the reward
    model and train directly on the preference pairs, nudging the model to make the
    "chosen" answer more likely than the "rejected" one. Same data, far simpler.

This script doesn't train anything (DPO on open weights needs a GPU + Hugging Face
`trl`; some providers offer hosted preference tuning). It shows the DATA SHAPE that
makes preference tuning different, and where the pairs come from.

    python examples/10_preference_tuning.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# A preference example isn't (input -> the one right output). It's a prompt plus
# TWO responses, labeled which is better: here, "warmer and more concise" wins.
PREFERENCE_PAIRS = [
    {
        "prompt": "A customer writes: 'I was double-charged this month.'",
        "chosen": "So sorry about that! I see the duplicate charge — I've refunded it, and it'll "
                  "post in 5-10 days. Anything else I can fix?",
        "rejected": "Per our billing policy, duplicate charges may occur and are reviewed within "
                    "5-10 business days in accordance with the terms of service section 4.2.",
    },
    {
        "prompt": "A customer writes: 'How do I export my notes?'",
        "chosen": "Happy to help! Go to Settings → Data → Export and we'll email you a download "
                  "link, usually within the hour.",
        "rejected": "Exporting is a feature. You can find it in the settings of the application "
                    "under the data-related options if it is enabled for your plan tier.",
    },
]


def main() -> int:
    print("Demonstration (SFT) vs. preference (DPO/RLHF)")
    print("=" * 64)
    print(
        "SFT learns from ONE right answer:   {prompt} -> {ideal reply}\n"
        "DPO learns from a COMPARISON:        {prompt} -> chosen ≻ rejected\n\n"
        "Reach for preferences when there's no single correct output but you can\n"
        "still say which of two is better: tone, conciseness, helpfulness, how\n"
        "firmly to refuse. That's most 'make it feel more like us' goals.\n"
    )

    print("The data shape")
    print("=" * 64)
    for i, pair in enumerate(PREFERENCE_PAIRS, 1):
        print(f"[pair {i}] {pair['prompt']}")
        print(f"   chosen  ≻  {pair['chosen'][:70]}...")
        print(f"   rejected   {pair['rejected'][:70]}...\n")
    print(
        "DPO's training signal, in one sentence: increase the model's likelihood of\n"
        "the CHOSEN response relative to the REJECTED one for the same prompt, widening\n"
        "the margin, while staying close to the original model so it doesn't drift.\n"
    )

    print("Where the pairs come from")
    print("=" * 64)
    print(
        "You rarely hand-write them. The cheapest source is your own traffic: the\n"
        "👍/👎 feedback flywheel from the Production dive turns real answers into\n"
        "preference pairs (thumbs-up = chosen, a fixed/edited version = rejected, or\n"
        "vice-versa). You can also have a strong model rank two candidates (RLAIF,\n"
        "AI feedback instead of human). Either way the asset is the same: pairs.\n"
    )

    print("How to actually run it")
    print("=" * 64)
    print(
        "Open weights: Hugging Face `trl` has a DPOTrainer that takes exactly the\n"
        "  {prompt, chosen, rejected} rows above, pairing with the Local Models dive\n"
        "  and the LoRA section (example 09); DPO is usually done as a LoRA.\n"
        "Hosted: some providers offer preference/RLHF tuning; the data shape is the\n"
        "  same pairs.\n\n"
        "The discipline doesn't change: you still PROVE it helped on a held-out set\n"
        "(Section 7's eval gate) before shipping; preferences are easier to collect\n"
        "than to get right."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
