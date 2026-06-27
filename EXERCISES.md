# Exercises — make the learning stick

Reading code teaches you less than *predicting* what it will do and then checking.
This file turns each section of the [README](README.md) into a few quick
active-recall prompts.

How to use it: work the section first, then come back. **Commit to an answer
before you run or reveal** — the prediction is where the learning happens. Answers
are hidden behind ▸ toggles.

> **Every section here runs offline and free** on the default `PROVIDER=mock` —
> including the simulated fine-tune. Nothing in these exercises costs a cent.

---

## Section 2 — When to fine-tune **(offline)**

**Recall.** Three tools change three different things about a model's answer. Match
each — RAG/long context, fine-tuning, tools/agents — to *what it changes*.

<details><summary>▸ Answer</summary>

- **RAG / long context** → changes *what's in the context* (knowledge).
- **Fine-tuning** → changes *how the model behaves by default* (format, tone, a skill).
- **Tools / agents** → changes *what the model can do* (capability).

Most "we need to fine-tune" instincts are actually a knowledge problem (RAG) or a
prompt problem (few-shot) in disguise.
</details>

**Predict, then run.** In `examples/01_when_to_finetune.py`, a scenario needs the
model to answer support questions using this week's pricing, with citations. Which
tool does it land on, and why *not* fine-tuning?

<details><summary>▸ Answer</summary>

**RAG**, not fine-tuning. The pricing *changes* and must be *cited* — that's
knowledge, and baking it into weights would be stale the moment prices change (and
can't cite a source). Fine-tuning is for behavior that stays the same, not facts
that move.
</details>

---

## Section 3 — The dataset is the product **(offline)**

**Recall.** What exact format does a hosted fine-tune expect, and which turn does
each training example *end* on?

<details><summary>▸ Answer</summary>

**JSON Lines (JSONL)** — one conversation per line, each a `{"messages": [...]}`
object. Each example ends on the **assistant** turn: that's the behavior the model
is being taught to reproduce. The system+user turns are the setup; the assistant
turn is the lesson.
</details>

**Do.** Run `examples/02_dataset_format.py`. Why does it split the data into *train*
and *validation* instead of training on all of it?

<details><summary>▸ Answer</summary>

So you have data the model **didn't train on** to measure against (Section 7). If
you evaluate on the training rows, a model that simply memorized them looks
perfect while generalizing terribly — the validation/held-out split is the only
thing that catches overfitting.
</details>

---

## Section 4 — Validate before you train **(offline)**

**Predict, then run.** Before running `examples/03_validate_data.py`: name two
problems an offline check can catch *before* you spend a cent training.

<details><summary>▸ Answer</summary>

Any of: malformed/duplicate examples, **class imbalance** (e.g. 90% of rows are one
category, so the model just learns to guess it), a missing assistant turn, a system
prompt that drifts between rows, or a token/cost estimate that's bigger than you
expected. The script runs all of these on the good set and then on a deliberately
**broken** set so you watch each check fire.
</details>

---

## Section 5 — Run a fine-tune job

**Recall.** List the four steps of the hosted fine-tune lifecycle in order.

<details><summary>▸ Answer</summary>

1. **upload** the training file (`files.create`)
2. **create** a job from it (`fine_tuning.jobs.create`) — this is what trains
3. **poll** the job until done (`fine_tuning.jobs.retrieve`)
4. **use** the returned model id

`examples/04_run_finetune.py` runs this on the mock (instant, free) with nearly
identical code to the real OpenAI path.
</details>

**Predict.** You run `examples/04_run_finetune.py` with `PROVIDER=openai` but
*without* `--real`. What happens, and why is it built that way?

<details><summary>▸ Answer</summary>

It does **not** start a paid job — the real path is opt-in behind `--real` plus a
confirmation, precisely because a fine-tune costs real money and takes real time.
You have to *mean* it. Without `--real` it runs the safe simulated path.
</details>

---

## Section 6 — Use the fine-tuned model

**Recall.** After a job succeeds, what's different about *calling* the fine-tuned
model versus the base model?

<details><summary>▸ Answer</summary>

Almost nothing — it's a normal chat call; you just pass the **new model id** (e.g.
`ft:gpt-4o-mini:...`) instead of the base id. There's no special "use a fine-tune"
endpoint. The behavior change lives in the weights, not the API shape.
</details>

**Do.** Run `examples/05_use_model.py` and watch base vs. tuned on the same
questions. What concretely changed, and what did *not*?

<details><summary>▸ Answer</summary>

**Changed:** the tuned model now answers in the rigid `category: ... | reply: ...`
house format the training demonstrated — reliably, with no reminder in the prompt.
**Did not change:** its underlying knowledge. Fine-tuning taught a *format/behavior*,
not new facts.
</details>

---

## Section 7 — Did it actually help?

**Recall.** Why is a held-out set non-negotiable when measuring a fine-tune?

<details><summary>▸ Answer</summary>

Because a model can score perfectly on rows it *trained on* simply by memorizing
them — that number is meaningless. Only data the training never saw
(`datasets/support_eval.jsonl`) tells you whether the behavior **generalizes**. No
held-out score, no claim.
</details>

**Predict, then run.** `examples/06_did_it_help.py` reports both accuracy and a
pairwise win-rate. If the tuned model *loses* to the base model, what's the correct
move?

<details><summary>▸ Answer</summary>

**Don't ship it.** A fine-tune that doesn't beat the baseline is a regression, no
matter how much effort went in. Go back to the dataset — more, cleaner, better-
balanced examples — and re-measure. "We already trained it" is the sunk-cost
trap.
</details>

---

## Section 8 — Hyperparameters **(offline)**

**Recall.** What does the *training loss falling while validation loss rises* tell
you, and which knob do you turn?

<details><summary>▸ Answer</summary>

**Overfitting** — the model is memorizing the training set instead of learning the
general behavior. The first knob to turn down is **n_epochs** (fewer passes over
the data); more/cleaner data helps too. `examples/07_hyperparameters.py` draws this
exact divergence in the simulated loss curve.
</details>

---

## Section 9 — Distillation **(offline)**

**Recall.** In distillation, who is the *teacher*, who is the *student*, and why is
the dataset so cheap to build?

<details><summary>▸ Answer</summary>

The **teacher** is a big, strong, expensive model that already does the task; the
**student** is a small, cheap, fast model you fine-tune to imitate it. The dataset
is cheap because **the labels write themselves** — you run the teacher over your
inputs and use its outputs as the assistant turns. No hand-labeling.
</details>

**Do.** Run `examples/08_distillation.py`. After it builds the distilled set, what
can you immediately do with the file — and which earlier section does that prove
the point of?

<details><summary>▸ Answer</summary>

Feed it straight into the Section 5 tune step — it's a normal training file. The
example **validates** it (Section 4) to prove exactly that: a distillation set is
just a training set whose labels came from a model instead of a human.
</details>

---

## Section 10 — Open weights & LoRA **(offline, conceptual)**

**Recall.** What's the difference between *full* fine-tuning and *LoRA/PEFT*, and
what stays identical between hosted and open-weight training?

<details><summary>▸ Answer</summary>

**Full** fine-tuning updates *all* the model's weights (heavy: lots of GPU memory).
**LoRA/PEFT** freezes the base weights and trains a small number of *added* weights
— far cheaper, almost as good for most tasks. What's identical either way: **your
dataset**. The JSONL you built in Sections 3–4 is exactly what an open-weight
trainer consumes too.
</details>

---

## Section 11 — Preference tuning (DPO/RLHF) **(offline, conceptual)**

**Recall.** SFT (every earlier section) trains on the one right answer. When would
you reach for preference tuning instead, and how is its training data shaped
differently?

<details><summary>▸ Answer</summary>

Reach for it when there's **no single correct output** but you can still say which of
two is better — tone, conciseness, helpfulness, how firmly to refuse. Its data isn't
`{prompt → ideal answer}`; it's `{prompt, chosen, rejected}` pairs. DPO nudges the
model to make the *chosen* response more likely than the *rejected* one. The cheapest
source of pairs is your own 👍/👎 traffic (the Production feedback flywheel).
</details>

---

## Capstone — `finetune_run.py`

**Do.** Run `python hands_on/finetune_run.py`. It chains validate → tune →
eval-gate → ship-if-wins. What is the gate actually deciding, and what does a
*non-zero exit* mean?

<details><summary>▸ Answer</summary>

The gate decides whether the tuned model **provably beat the base model** on the
held-out set (optionally above a `--min-winrate` you set). A non-zero exit means it
**did not** clear the bar — the same signal a CI eval gate gives, so a bad fine-tune
can't silently ship.
</details>

**Stretch.** Run it again with `--train datasets/support_distilled.jsonl` (build it
first via Section 9). Does the distilled set clear the gate? Then write **five new
training rows** of your own, re-run, and watch the numbers move. The first time you
change the *data* and the held-out score changes, the "the dataset is the product"
idea has clicked.

---

### Where to take it next

Pick a tiny behavior you want a model to do *the same way every time* — a fixed
JSON shape, a house tone, a one-line classification — write 30–50 honest examples,
validate them, tune on the mock, and gate. If it wins on held-out data, you've done
the entire real workflow; only then is it worth spending money to run it for real.
