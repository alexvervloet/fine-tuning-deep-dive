"""
finetune/mock_tuner.py — the offline fine-tuning lifecycle, simulated.
======================================================================

This is the file that makes the whole repo free. A real fine-tune is: upload a
file, create a job, poll it from `queued` -> `running` -> `succeeded` (minutes to
hours), then use the new model id. Every step costs money or waits on a server.

The mock reproduces that *exact lifecycle* in-process, deterministically, in a
fraction of a second, for $0:

    file_id  = upload_training_file("datasets/train.jsonl")   # like files.create
    job      = create_job(file_id, hyperparameters=...)       # like jobs.create
    while not job.is_done():                                  # like jobs.retrieve
        job.poll()
    model_id = job.fine_tuned_model                           # the tuned model

The shape is intentionally identical to the real OpenAI calls wrapped in
providers.py, so the example scripts read the same whether you run the mock or
the real thing — only a flag changes.

What does the mock actually "learn"? It reads the training file and builds a
small behavior table: for each example, it maps a keyword from the user turn to
the assistant reply the data demonstrated. On `succeeded`, it registers that
table with providers.apply_mock_finetune(), so calling generate(model=<tuned id>)
now produces the taught behavior. That's a toy of the real thing — but it's
enough to make Section 7 ("did it help?") give a real, measurable win offline:
the base model rambles, the "fine-tuned" model answers in the trained format.

The mock also fabricates a plausible loss curve from the hyperparameters, so
Section 8 (reading the loss curve) has something to plot offline.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field

from . import providers
from .dataset import ChatExample, load_jsonl

# The simulated lifecycle states, in order. Real providers add validating_files
# and can end in failed/cancelled; we keep the happy path for teaching.
_STATES = ["queued", "running", "running", "succeeded"]


@dataclass
class MockFile:
    """An 'uploaded' training file. Holds the parsed examples in memory."""

    id: str
    path: str
    examples: list[ChatExample]
    n_bytes: int


_files: dict[str, MockFile] = {}


def upload_training_file(path: str) -> str:
    """Simulate files.create(purpose='fine-tune'). Returns a deterministic file id.

    Deterministic so re-running an example is reproducible. The id is derived from
    the file path, the way `file-abc123` would identify your upload on the server.
    """
    examples = load_jsonl(path)
    with open(path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha1(path.encode()).hexdigest()[:12]
    file_id = f"file-mock-{digest}"
    _files[file_id] = MockFile(id=file_id, path=path, examples=examples, n_bytes=len(raw))
    return file_id


def _learn_behavior(examples: list[ChatExample]) -> dict[str, str]:
    """Derive the 'fine-tuned' behavior table from the training data.

    For each example we take a salient keyword from the user message and map it to
    the assistant reply. This is a deliberate toy of what training does — a real
    fine-tune adjusts weights, not a lookup table — but it captures the essential
    truth we want learners to feel: the model's new behavior comes *entirely* from
    the examples you showed it. Garbage examples -> garbage behavior.
    """
    table: dict[str, str] = {}
    # A few content words we treat as the example's "topic" keyword, longest first
    # so 'password' beats 'pass'. In a real dataset you'd never do this; here it
    # just gives the mock something stable to key on.
    topic_words = sorted(
        ["password", "refund", "cancel", "billing", "login", "export", "upgrade", "invoice"],
        key=len,
        reverse=True,
    )
    for ex in examples:
        user = (ex.last_user() or "").lower()
        reply = ex.last_assistant()
        if not reply:
            continue
        for word in topic_words:
            if re.search(rf"\b{re.escape(word)}\b", user):
                table.setdefault(word, reply)
                break
    return table


@dataclass
class MockJob:
    """A simulated fine-tuning job — the same surface as the real one.

    Drive it with poll() until is_done(); then read fine_tuned_model. The loss
    curve is fabricated from the hyperparameters so Section 8 has data to plot.
    """

    id: str
    training_file: str
    model: str
    hyperparameters: dict
    status: str = "queued"
    fine_tuned_model: str | None = None
    trained_tokens: int = 0
    loss_curve: list[float] = field(default_factory=list)
    _step: int = 0

    def is_done(self) -> bool:
        return self.status in ("succeeded", "failed", "cancelled")

    def poll(self) -> str:
        """Advance the job one step and return the new status, like a retrieve()
        call would on a server that's been working in the background."""
        time.sleep(0.05)  # a token pause so the lifecycle is visible, not free-real-time
        if self.is_done():
            return self.status
        self._step += 1
        self.status = _STATES[min(self._step, len(_STATES) - 1)]
        if self.status == "succeeded":
            self._finish()
        return self.status

    def _finish(self) -> None:
        mock_file = _files[self.training_file]
        behavior = _learn_behavior(mock_file.examples)
        # A tuned model id mirrors OpenAI's shape: ft:<base>:<org>:<suffix>:<id>.
        suffix = self.hyperparameters.get("suffix", "support")
        self.fine_tuned_model = f"ft:{self.model}:deepdive:{suffix}:{self.id[-6:]}"
        epochs = int(self.hyperparameters.get("n_epochs", 3))
        self.trained_tokens = sum(
            sum(max(1, len(m.content) // 4) for m in ex.messages) for ex in mock_file.examples
        ) * epochs
        self.loss_curve = _fake_loss_curve(self.hyperparameters, n_examples=len(mock_file.examples))
        # Register the tuned behavior so providers.generate(model=...) serves it.
        providers.apply_mock_finetune(self.fine_tuned_model, behavior)


def create_job(training_file: str, *, model: str | None = None, hyperparameters: dict | None = None) -> MockJob:
    """Simulate fine_tuning.jobs.create(). Returns a MockJob in 'queued' state."""
    if training_file not in _files:
        raise ValueError(f"unknown training file {training_file!r} — upload it first")
    digest = hashlib.sha1((training_file + str(hyperparameters)).encode()).hexdigest()[:6]
    return MockJob(
        id=f"ftjob-mock-{digest}",
        training_file=training_file,
        model=model or providers.base_model(),
        hyperparameters=hyperparameters or {},
    )


def _fake_loss_curve(hyperparameters: dict, *, n_examples: int) -> list[float]:
    """Fabricate a plausible training-loss curve from the hyperparameters.

    The point is pedagogical (Section 8): more epochs -> more steps and a lower
    final loss, but with diminishing returns and rising overfitting risk; a higher
    learning-rate multiplier descends faster but gets noisier. None of this is a
    real optimizer — it's a shape that lets you *read* a loss curve offline.
    """
    epochs = int(hyperparameters.get("n_epochs", 3))
    lr_mult = float(hyperparameters.get("learning_rate_multiplier", 1.0))
    steps = max(4, n_examples) * epochs
    curve = []
    for s in range(steps):
        # Exponential descent toward a floor, faster with a bigger LR multiplier.
        base = 0.2 + 1.8 * math.exp(-3.0 * lr_mult * s / steps)
        # A little deterministic jitter so it looks like a real (noisy) curve.
        jitter = 0.05 * lr_mult * math.sin(s * 1.7)
        curve.append(round(max(0.05, base + jitter), 4))
    return curve
