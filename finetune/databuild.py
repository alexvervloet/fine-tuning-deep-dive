"""
finetune/databuild.py: small helpers for *building* training examples.

Section 3's lesson is "the dataset is the product," and most of the work of a
fine-tune is producing good examples. These helpers make that concrete:

  - make_example()      assemble a (system, user, assistant) into a ChatExample
  - format_target()     write the assistant turn in the house format, so every
                          example demonstrates the SAME output shape (consistency
                          is what the model actually learns)
  - distillation_example() turn a strong "teacher" model's answer into a
                          training example for a smaller "student" model. This is
                          the shape of Section 9 (distillation): you don't
                          hand-write the assistant turns, you harvest them from a
                          model that already does the task well.

Nothing here calls a provider; the teacher call lives in the example script, and
distillation_example() just wraps whatever text you pass it. That keeps this file
offline and the I/O explicit.
"""

from __future__ import annotations

from .dataset import ChatExample, Message

# The system prompt we use everywhere: in training AND at inference. Keeping it
# identical is what lets the model associate this instruction with the behavior.
SUPPORT_SYSTEM = (
    "You are Acme Cloud's support triage assistant. Classify the user's message "
    "and reply in EXACTLY this format: 'category: <one of account|billing|technical|other> "
    "| reply: <one short, friendly sentence>'."
)


def format_target(category: str, reply: str) -> str:
    """Render the assistant turn in the house format. One place defines the shape,
    so every example is consistent (a validator-friendly, model-friendly habit)."""
    return f"category: {category} | reply: {reply}"


def make_example(user: str, category: str, reply: str, *, system: str = SUPPORT_SYSTEM) -> ChatExample:
    """Build one training example from its parts."""
    return ChatExample(
        messages=[
            Message("system", system),
            Message("user", user),
            Message("assistant", format_target(category, reply)),
        ]
    )


def distillation_example(user: str, teacher_answer: str, *, system: str = SUPPORT_SYSTEM) -> ChatExample:
    """Wrap a teacher model's answer as a student training example.

    In distillation you call a strong, expensive model (the teacher) on a pile of
    inputs, then train a small, cheap model (the student) to reproduce those
    answers. Each (input, teacher_answer) pair becomes one training example, and this
    is the only step that differs from hand-built data, and it's why distillation
    scales: the labels write themselves.
    """
    return ChatExample(
        messages=[
            Message("system", system),
            Message("user", user),
            Message("assistant", teacher_answer.strip()),
        ]
    )
