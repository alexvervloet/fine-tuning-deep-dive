"""
finetune: a small, from-scratch library for learning to fine-tune models.

Everything here is built to be *read*, not just used. The big idea: fine-tuning
changes HOW a model behaves (its default format, tone, narrow skill), taught by
EXAMPLES, and then you must PROVE it beat your baseline. The files map to that
arc:

  providers.py   the ONLY provider-specific file: chat generate() + the tuning
                   lifecycle (mock | openai | claude). The mock makes it free.
  dataset.py     the chat JSONL format: ChatExample, load/write, train/val split
  databuild.py   helpers for *building* examples (incl. distillation)
  validate.py    offline checks: schema, dedup, balance, token + cost estimate
  mock_tuner.py  the offline fine-tuning lifecycle (upload->job->poll->use)
  evaluate.py    did it help? accuracy + win-rate vs. baseline (held-out set)

Import what you need, e.g.:

    from finetune import load_jsonl, validate_dataset, mock_tuner
"""

from . import mock_tuner, providers
from .databuild import (
    SUPPORT_SYSTEM,
    distillation_example,
    format_target,
    make_example,
)
from .dataset import (
    ChatExample,
    Message,
    load_jsonl,
    train_val_split,
    write_jsonl,
)
from .evaluate import EvalResult, accuracy_on, win_rate
from .providers import (
    base_model,
    can_tune,
    describe,
    ensure_ready,
    generate,
    provider_name,
)
from .validate import (
    ValidationReport,
    class_balance,
    estimate_cost,
    estimate_tokens,
    find_duplicates,
    validate_dataset,
    validate_schema,
)

__all__ = [
    # providers
    "providers",
    "generate",
    "provider_name",
    "describe",
    "ensure_ready",
    "base_model",
    "can_tune",
    # dataset
    "ChatExample",
    "Message",
    "load_jsonl",
    "write_jsonl",
    "train_val_split",
    # databuild
    "SUPPORT_SYSTEM",
    "make_example",
    "format_target",
    "distillation_example",
    # validate
    "ValidationReport",
    "validate_dataset",
    "validate_schema",
    "find_duplicates",
    "class_balance",
    "estimate_tokens",
    "estimate_cost",
    # mock_tuner
    "mock_tuner",
    # evaluate
    "EvalResult",
    "accuracy_on",
    "win_rate",
]
