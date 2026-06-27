"""
finetune/providers.py — the ONLY file that talks to a fine-tuning backend.
==========================================================================

Same keystone idea as every sibling repo: hide the provider-specific calls
behind a tiny, uniform interface so the rest of the code is provider-agnostic.
Fine-tuning has *two* kinds of provider call, not one:

  1. a CHAT call  — using a model (base or fine-tuned) to generate text. This is
     what the eval step needs to compare "before" and "after."
  2. a TUNING lifecycle — upload a training file, create a job, poll it to done,
     and learn the resulting model's name. This is what the rest of this repo is
     actually about.

There are three stacks, selected by the PROVIDER env var:

  PROVIDER=mock    ->  a deterministic, offline, in-process backend. No key, no
                       network, no cost. It both (a) "fine-tunes" by simulating
                       the whole upload->job->poll->use lifecycle and (b) serves
                       a tiny chat model so the eval works offline. THE DEFAULT,
                       and what makes the entire learning arc free. See
                       mock_tuner.py for the simulated lifecycle.
  PROVIDER=openai  ->  real OpenAI fine-tuning + chat   (needs OPENAI_API_KEY,
                       and the tuning job COSTS REAL MONEY — opt-in only).
  PROVIDER=claude  ->  chat only. Anthropic does not offer self-serve fine-tuning
                       (it's a limited/enterprise program), so the tuning
                       lifecycle is unavailable on this stack. You can still use
                       PROVIDER=claude as the *base* model to compare against, or
                       as the strong "teacher" in the distillation example.

Honesty matters here: hosted, self-serve fine-tuning in this repo is mainly an
OpenAI path. That's not a limitation of the teaching — the *concepts* (the
dataset is the product, validate before you spend, prove it beat the baseline)
are provider-independent, and the mock lets you practice all of them for $0.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache

# Default models per stack. Match the sibling repos' model IDs exactly.
_OPENAI_CHAT = "gpt-4o-mini"           # the base model we fine-tune (and the default chat model)
_CLAUDE_CHAT = "claude-haiku-4-5"
_MOCK_MODEL = "mock-1"

# The model families that OpenAI currently supports for self-serve fine-tuning.
# (gpt-4o-mini is the cheap, sensible default to learn on.)
_OPENAI_TUNABLE = "gpt-4o-mini-2024-07-18"

_KEYS = {
    "mock": [],  # the whole point: no key required
    "openai": ["OPENAI_API_KEY"],
    "claude": ["ANTHROPIC_API_KEY"],
}

# Which stacks can actually run a fine-tuning job (vs. chat-only).
_CAN_TUNE = {"mock": True, "openai": True, "claude": False}


@dataclass
class LLMResponse:
    """One chat call's result, plus the metadata the eval layer wants."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def provider_name() -> str:
    """The active stack: 'mock' (default), 'openai', or 'claude'."""
    return os.getenv("PROVIDER", "mock").strip().lower()


def required_keys() -> list[str]:
    return _KEYS.get(provider_name(), [])


def can_tune() -> bool:
    """True if the active stack supports running a fine-tuning job."""
    return _CAN_TUNE.get(provider_name(), False)


def base_model() -> str:
    """The base (un-tuned) model id for the active stack — what you fine-tune
    *from* and compare *against*."""
    return {"mock": _MOCK_MODEL, "openai": _OPENAI_CHAT, "claude": _CLAUDE_CHAT}.get(
        provider_name(), _MOCK_MODEL
    )


def tunable_model() -> str:
    """The exact model snapshot to pass to a fine-tune job (OpenAI pins a date)."""
    return {"mock": _MOCK_MODEL, "openai": _OPENAI_TUNABLE}.get(provider_name(), _MOCK_MODEL)


def describe() -> str:
    p = provider_name()
    if p == "mock":
        return f"mock  (offline, deterministic, model={_MOCK_MODEL}, no key, tuning simulated)"
    if p == "openai":
        return f"openai  (chat={_OPENAI_CHAT}, tunable={_OPENAI_TUNABLE})"
    if p == "claude":
        return f"claude  (chat={_CLAUDE_CHAT}, tuning NOT available on this stack)"
    return f"unknown provider {p!r}"


def ensure_ready(*, for_tuning: bool = False) -> None:
    """Fail fast with a friendly message if the stack isn't configured.

    For PROVIDER=mock this never fails — that's the point. Pass for_tuning=True
    from the tune scripts so we can warn early when a stack can't fine-tune.
    """
    import sys

    p = provider_name()
    if p not in _KEYS:
        sys.exit(
            f"PROVIDER={p!r} is not recognized. Set PROVIDER=mock (default), "
            f"openai, or claude in .env."
        )
    missing = [k for k in required_keys() if not os.getenv(k)]
    if missing:
        sys.exit(
            f"PROVIDER={p} needs {', '.join(missing)} in .env. "
            f"See .env.example, or run `python check_setup.py`. "
            f"(Tip: PROVIDER=mock needs no key and runs everything offline.)"
        )
    if for_tuning and not can_tune():
        sys.exit(
            f"PROVIDER={p} can't run a fine-tuning job (Anthropic fine-tuning is "
            f"a limited/enterprise program, not self-serve). Use PROVIDER=mock to "
            f"practice the whole lifecycle offline for free, or PROVIDER=openai for "
            f"the real (paid) path."
        )


# ---------------------------------------------------------------------------
# The mock chat model — a deterministic, offline "model"
# ---------------------------------------------------------------------------
#
# It classifies short support messages into a category and a one-line reply,
# from a tiny built-in rulebook. The point isn't the rules — it's that the mock
# is deterministic, so the eval in Section 7 produces a stable, reproducible
# number with no key. The mock can be handed a "fine-tuned" behavior table by
# the mock tuner (see apply_mock_finetune below); a base call uses the weaker
# default behavior, so fine-tuning visibly *changes how it behaves*.

# The base model's behavior: it knows a couple of categories but is sloppy about
# format (it rambles instead of answering in the house style). Fine-tuning will
# fix the format and add the categories it's missing.
_MOCK_BASE_RULES = {
    "password": "You can probably reset your password somewhere in the settings, I think.",
    "refund": "Refunds might be possible, you'd have to check the billing page maybe.",
}
_MOCK_BASE_FALLBACK = "Hmm, I'm not totally sure about that one, sorry!"

# A fine-tuned model is just the base model plus a learned behavior table that the
# mock tuner derives from the training data. Set by apply_mock_finetune().
_mock_finetuned_table: dict[str, str] | None = None
_mock_finetuned_name: str | None = None


def apply_mock_finetune(name: str, behavior: dict[str, str]) -> None:
    """Register a mock fine-tuned model so generate() can serve it.

    `behavior` maps a keyword to the exact house-style reply the training data
    taught. This is the offline stand-in for "the provider now hosts your tuned
    model under this name." Called by mock_tuner.MockJob when a job completes.
    """
    global _mock_finetuned_table, _mock_finetuned_name
    _mock_finetuned_name = name
    _mock_finetuned_table = dict(behavior)


def _approx_tokens(text: str) -> int:
    """A rough token count (~4 chars/token), good enough for cost demos offline."""
    return max(1, len(text) // 4)


# Keywords -> category, used by the fine-tuned mock to follow the house FORMAT
# even on unseen inputs. A real fine-tune learns the format robustly (it saw it on
# every example), so the tuned model generalizes the 'category: ... | reply: ...'
# shape to paraphrases the base model fumbles. This is what makes the held-out
# eval in Section 7 show a real, generalizing win rather than memorization.
_MOCK_CATEGORY_HINTS = {
    "account": ["password", "log in", "login", "sign in", "account", "2fa", "two-factor",
                "multi-factor", "email", "member", "teammate", "colleague", "workspace", "team"],
    "billing": ["refund", "money", "charge", "charged", "invoice", "receipt", "billing",
                "plan", "subscription", "cancel", "upgrade", "downgrade", "discount", "annual", "pay"],
    "technical": ["error", "500", "503", "401", "crash", "crashes", "freeze", "freezes",
                  "load", "loading", "spin", "export", "download", "api", "save", "outage", "bug"],
}


def _classify_mock(q: str) -> str:
    best, best_score = "other", 0
    for cat, words in _MOCK_CATEGORY_HINTS.items():
        score = sum(1 for w in words if w in q)
        if score > best_score:
            best, best_score = cat, score
    return best


def _mock_generate(system: str, user: str, model: str) -> LLMResponse:
    q = user.lower()

    # A fine-tuned model id routes to the learned behavior table; anything else is
    # the (weaker) base model. This is what makes "did it help?" land offline.
    if _mock_finetuned_table is not None and model == _mock_finetuned_name:
        answer = _MOCK_BASE_FALLBACK
        for keyword, reply in _mock_finetuned_table.items():
            if keyword in q:
                answer = reply
                break
        # The fine-tune also taught the house FORMAT even on unseen inputs: emit a
        # tidy 'category: ... | reply: ...' (best-effort category) instead of the
        # base model's category-less ramble. This is the generalization a real
        # fine-tune buys you, and it's what lets the tuned model win on held-out
        # paraphrases it never saw verbatim.
        if answer is _MOCK_BASE_FALLBACK:
            cat = _classify_mock(q)
            answer = f"category: {cat} | reply: Happy to help — here's the quickest way to sort this out."
    else:
        answer = _MOCK_BASE_FALLBACK
        for keyword, reply in _MOCK_BASE_RULES.items():
            if keyword in q:
                answer = reply
                break

    return LLMResponse(
        text=answer,
        model=model,
        prompt_tokens=_approx_tokens(system + user),
        completion_tokens=_approx_tokens(answer),
        latency_ms=2.0,
    )


# --- Real providers: created lazily, so importing this module never forces an
#     SDK import or a network call. ---


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI()


@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic

    return anthropic.Anthropic()


def generate(system: str, user: str, *, model: str | None = None, max_tokens: int = 256) -> LLMResponse:
    """Turn a (system, user) prompt into an `LLMResponse`, from a chosen model.

    `model` defaults to the active stack's base model. Pass a fine-tuned model id
    to use the tuned model instead — that's how the eval compares base vs tuned.
    """
    p = provider_name()
    model = model or base_model()

    if p == "mock":
        return _mock_generate(system, user, model)

    start = time.perf_counter()
    if p == "openai":
        resp = _openai_client().chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=model,
            prompt_tokens=usage.prompt_tokens if usage else _approx_tokens(system + user),
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
        )
    if p == "claude":
        resp = _anthropic_client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = (time.perf_counter() - start) * 1000
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(
            text=text,
            model=model,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            latency_ms=latency_ms,
        )
    raise ValueError(f"Unknown PROVIDER={p!r} (expected 'mock', 'openai', or 'claude').")


# ---------------------------------------------------------------------------
# The real OpenAI tuning lifecycle (used by examples/05 behind an opt-in flag).
# ---------------------------------------------------------------------------
#
# These are thin wrappers so the real path reads the same as the mock path. They
# are only reached on PROVIDER=openai AND with the explicit opt-in flag in the
# tune script — never by default, never by the tests, never for free.


def openai_upload_training_file(path: str) -> str:
    """Upload a JSONL training file to OpenAI; returns the file id."""
    with open(path, "rb") as f:
        result = _openai_client().files.create(file=f, purpose="fine-tune")
    return result.id


def openai_create_job(training_file_id: str, *, model: str | None = None, hyperparameters: dict | None = None) -> str:
    """Create a fine-tuning job; returns the job id. THIS STARTS A PAID JOB."""
    kwargs: dict = {"training_file": training_file_id, "model": model or tunable_model()}
    if hyperparameters:
        kwargs["hyperparameters"] = hyperparameters
    job = _openai_client().fine_tuning.jobs.create(**kwargs)
    return job.id


def openai_poll_job(job_id: str) -> dict:
    """Fetch a job's current status. Returns a small dict of the fields we show."""
    job = _openai_client().fine_tuning.jobs.retrieve(job_id)
    return {
        "id": job.id,
        "status": job.status,  # validating_files | queued | running | succeeded | failed | cancelled
        "fine_tuned_model": job.fine_tuned_model,  # populated when succeeded
        "trained_tokens": getattr(job, "trained_tokens", None),
    }
