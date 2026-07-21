# Fine-tuning: A Guided Deep Dive

A hands-on playground for learning **fine-tuning**, teaching a model new
*behavior* from examples, from the ground up. You'll build a training set from
scratch, validate it, run a fine-tune job, and then do the part everyone skips:
*prove* the fine-tuned model actually beat the base model on data it never saw.
No framework magic, just enough code to see how each step works.

The twist that makes this repo work: it runs **completely offline on a mock
provider**, with no API key. Real fine-tuning costs money and takes minutes-to-hours,
which is a terrible way to learn the *shape* of it. So the default `PROVIDER=mock`
ships a tiny deterministic "model" and a **simulated fine-tune lifecycle**
(upload → job → poll → use) that runs in-process, in under a second, for $0. Flip
one env var and the exact same code runs a real, paid OpenAI fine-tune.

This repo is **standalone**: it teaches everything it needs on its own. It is the
hands-on version of the [RAG deep dive](https://github.com/alexvervloet/rag-deep-dive)'s
"RAG, fine-tuning, or something else?" section, and Section 7 borrows the win-rate
method from the [Evals deep dive](https://github.com/alexvervloet/evals-deep-dive), but
its code depends on neither.

Like its siblings, it's meant to be *walked through*. Each section ends with
something to run, and **every section runs offline and free** on the mock.
[EXERCISES.md](EXERCISES.md) has a predict-then-run prompt for each one.

---

## 0. The one big idea

> **Fine-tuning changes how the model *behaves*, not what it *knows*. You teach
> a default behavior with examples, and then you must *prove* it beat your
> baseline.**

That's the whole repo. RAG and long context change *what's in the context window*
(knowledge); fine-tuning changes *how the model responds by default* (format,
tone, a narrow skill), taught only by showing it input→output examples. So the
training set **is** the product; most of the work is building and validating it.
And because "it feels better" is worth nothing, the discipline that makes
fine-tuning real is the last step: measure the tuned model against the base model
on a held-out set, and ship only if it wins. Hold onto that and none of this feels
complicated.

---

## 1. Setup (5 minutes)

```bash
# 1. Create an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies (the default mock stack needs only python-dotenv)
pip install -r requirements.txt

# 3. Copy the env file: the default runs keyless (no API key needed)
cp .env.example .env
#    (Real provider instead of the mock? Its key goes in your OS keychain,
#     not .env: see ../SECRETS.md, then run scripts as `secrun python ...`.)

# 4. Confirm everything is wired up (makes no API call, costs nothing)
python check_setup.py
```

The default stack is **offline and free**; the whole learning arc runs on a mock
provider with no key. Pick a real provider only when you want to run an actual
(paid) fine-tune:

| `PROVIDER` | What runs | Key needed |
|------------|-----------|------------|
| `mock` (default) | Deterministic in-process model **+ a simulated fine-tune lifecycle**. The entire repo, free. | none |
| `openai` | Real chat **+ the real fine-tuning API**. Running a job **costs real money** and is opt-in (`--real`). | `OPENAI_API_KEY` |
| `claude` | Chat only. Anthropic fine-tuning is a limited/enterprise program, not self-serve. Usable as a base/teacher model. | `ANTHROPIC_API_KEY` |

> **You can complete every section for $0.** The mock simulates uploading,
> training, polling, and serving a fine-tuned model. The only thing it can't do is
> spend your money. The real OpenAI path exists so you can see the identical code
> hit a real provider, but you never *need* it to learn the ideas.

---

## 2. When to fine-tune: vs. prompt, few-shot, RAG

```bash
python examples/01_when_to_finetune.py
```

The most valuable fine-tuning skill is knowing when **not** to. Fine-tuning is the
slow, expensive, provider-specific option, and reaching for it first is the most
common and costly mistake. One rule resolves most cases:

- **RAG / long context** change *what's in the context* → reach for it when you
  need facts that change or must be cited.
- **Fine-tuning** changes *how the model behaves by default* → reach for it when
  you need the same format/tone/skill every time, or lower cost/latency on a
  fixed, high-volume task.
- **Tools / agents** change *what the model can do* → reach for them when it must
  act or fetch live data.

The example walks a handful of real scenarios and lands each on the right tool.
The headline rule of thumb, **don't fine-tune first**, is here for a reason:
a better prompt or a few examples solves most of what people *think* needs
training.

---

## 3. The dataset is the product

```bash
python examples/02_dataset_format.py
```

A fine-tune learns the behavior its examples demonstrate, so the examples *are*
the product. Hosted fine-tuning uses **JSON Lines**: one conversation per line,
each ending in the assistant turn you want the model to learn to produce.

```json
{"messages": [
  {"role": "system", "content": "You are Acme's support triage assistant. Reply EXACTLY as 'category: <...> | reply: <...>'."},
  {"role": "user", "content": "i forgot my password"},
  {"role": "assistant", "content": "category: account | reply: Use Settings > Security > Reset password."}
]}
```

The example builds one from scratch, loads the hand-made
[`datasets/support_train.jsonl`](datasets/support_train.jsonl), and splits it into
train/validation. The whole repo's running example is a **support-triage**
assistant taught to answer in one rigid house format: exactly the kind of
narrow, repeated behavior fine-tuning is good at.

---

## 4. Validate your data before you pay to train on it

```bash
python examples/03_validate_data.py
```

A fine-tune job is slow and (on a real provider) costs money. The cheapest way to
waste neither is to check the dataset *first*. This runs every offline check on
the hand-made set (**schema**, **duplicates**, **class balance**, and a **token +
cost estimate**) then runs them again on a deliberately *broken* set so you can
watch the checks catch real problems. Garbage in, garbage model: most bad
fine-tunes are bad datasets that nobody validated.

---

## 5. Run a fine-tune job: upload → create → poll → use

```bash
python examples/04_run_finetune.py
```

Every hosted fine-tune follows the same lifecycle:

1. **upload** the training file (`files.create`)
2. **create** a job from it (`fine_tuning.jobs.create`), *this is what trains*
3. **poll** the job until it's done (`fine_tuning.jobs.retrieve`)
4. **use** the new model id

The example runs this **two ways with nearly identical code**, which is the
point:

- **Default (`PROVIDER=mock`)** simulates the whole lifecycle in-process,
  deterministically, in under a second, for $0.
- **Opt-in real run** (`PROVIDER=openai` *and* `--real`) uploads your file and
  starts a genuine job. It prints a cost warning and asks for confirmation first,
  and it can take a while.

---

## 6. Use the fine-tuned model

```bash
python examples/05_use_model.py
```

Once a job succeeds, the provider hosts your model under a new id (e.g.
`ft:gpt-4o-mini:...`). Using it is just a normal chat call with that id; there's
no special API. The example asks the **base** and the **fine-tuned** model the same
questions side by side: the base handles the one or two categories it happens to
know but rambles and ignores the house format on the rest; the tuned model snaps
every one into the trained `category: ... | reply: ...` shape. That behavior
change, taught only by examples, is the whole idea.

---

## 7. Did it actually help?

```bash
python examples/06_did_it_help.py
```

**The punchline.** A fine-tune you *think* is better is worth nothing; the only
thing worth shipping is one you can *prove* beat your baseline on data the training
never saw. This points the [Evals deep dive](https://github.com/alexvervloet/evals-deep-dive)'s
method at one decision, base vs. fine-tuned on the held-out
[`datasets/support_eval.jsonl`](datasets/support_eval.jsonl), with two numbers:

- **accuracy**: % of held-out examples where the predicted category matches gold.
- **win-rate**: pairwise, the way `evals/07_pairwise.py` does it: show a judge both
  answers and tally which is better. (Here the "judge" is an offline format rubric
  so it runs free; in production you'd use an LLM-as-judge.)

If the tuned model doesn't beat the baseline, the honest move is to *not ship it* 
and go back to the dataset.

---

## 8. Hyperparameters & reading the loss curve

```bash
python examples/07_hyperparameters.py
```

You don't need many knobs, and the defaults are usually fine, but three matter:

- **n_epochs**: how many times the model sees the whole dataset. Too few and it
  hasn't learned; too many and it **overfits**. The tell: validation loss stops
  dropping (or rises) while training loss keeps falling.
- **learning_rate_multiplier**: how big each step is. Higher is faster but can
  overshoot; lower is steadier but slower.
- **batch_size**: examples per update; mostly a speed/stability tradeoff. Leave
  it on auto unless you have a reason.

The example renders the (simulated) loss curve and shows what overfitting looks
like, so the abstract knobs become a picture.

---

## 9. Distillation: train a small model on a strong model's outputs

```bash
python examples/08_distillation.py
```

The most common production shape of fine-tuning isn't hand-labeling. It's
**distillation**: take a big, expensive, smart model (the **teacher**) that already
does your task well, run it over a pile of inputs, and use its answers as training
data for a small, cheap, fast model (the **student**). The labels write themselves,
which is what makes a set of hundreds or thousands of examples cheap to build. The
example *builds* a distillation dataset (on the mock, the "teacher" is just the mock
in the house format; on a real provider you'd point it at `gpt-4o`/`claude`), then
validates it, proving it's a normal training file you can feed straight into
Section 5.

---

## 10. Beyond hosted: open-weight fine-tuning & LoRA/PEFT

```bash
python examples/09_open_weights_lora.py
```

Everything so far was **hosted** fine-tuning: hand a provider a JSONL file, they
train and host the result. The other world is **open-weight** fine-tuning: download
a model whose weights are public (Llama, Mistral, Qwen, Gemma) and train it
yourself on your own (or rented) GPU. This section is conceptual; running open
weights needs a GPU and a different stack (PyTorch + Hugging Face
`transformers`/`peft`), its own deep dive, but it explains the two ideas you need
(**full fine-tuning vs. LoRA/PEFT**) and shows that **your dataset is the same
asset either way**: the file you built in Sections 3–4 is exactly what an
open-weight trainer consumes. To actually *run* open weights locally, see the
**[Local Models deep dive](https://github.com/alexvervloet/local-models-deep-dive)**.

---

## 11. Preference tuning: learning from comparisons (DPO/RLHF)

```bash
python examples/10_preference_tuning.py
```

Every example so far taught by **demonstration**: show the one right answer and
imitate (that's SFT). But some goals have no single right answer: "be warmer," "be
more concise," "refuse this more firmly." You can't write THE correct reply, but you
can say which of two replies is *better*. **Preference tuning** learns from exactly
that, pairs of `{prompt, chosen, rejected}`. **RLHF** trains a reward model from the
rankings; **DPO** (the modern shortcut) trains directly on the pairs. This section is
conceptual: it shows the data shape, where the pairs come from (the Production dive's
👍/👎 flywheel is the cheapest source), and how it's run (Hugging Face `trl`'s
DPOTrainer, usually as a LoRA). The discipline is unchanged: still gate on a held-out
eval before shipping.

---

## 12. Reinforcement fine-tuning (RFT): learning from a grader

SFT learns from **demonstrations** (one right answer to imitate). Preference tuning
learns from **comparisons** (A is better than B). **Reinforcement fine-tuning** learns
from a **grader**: the model generates an answer, a scoring function rates it, and
training pushes the model toward higher-scoring answers. There's no labeled target and
no pair, just a way to *score* an attempt. This section is conceptual (no runnable
example): graders are expensive and fiddly, and it's the most complex rung here.

The whole game is the grader: a function `score(prompt, answer) -> number`. It can be
a hard programmatic check (do the unit tests pass? does the JSON validate against the
schema? does the math answer match?) or a model-as-judge scoring against a rubric. This
is the "reinforcement learning from *verifiable* rewards" that trains modern reasoning
models: when correctness is *checkable* but you can't write down THE one right output,
a grader beats labeled data.

**When a grader beats labeled pairs**: reach for RFT when (a) success is easy to
*verify* but hard to *demonstrate* (there are many correct programs, proofs, or plans,
so you can check one but not enumerate them), (b) writing thousands of gold answers or
preference pairs is more expensive than writing one scoring function, or (c) you're
optimizing a multi-step behavior where only the *outcome* is gradeable. Stick with SFT
when you *can* cheaply demonstrate the target, and preference tuning when quality is a
matter of taste a judge can rank but not score objectively.

The catch, beyond cost: a model optimizing a score will **hack** a weak grader, passing
the letter of the check while missing the point (the eval-gaming failure the
[Evals dive](https://github.com/alexvervloet/evals-deep-dive) warns about, now inside the
training loop). So the grader itself needs the same scrutiny as an eval, and the
shipping discipline is unchanged: gate on a held-out set the grader never saw.

---

## The capstone: `finetune_run.py`

Everything assembled into one command that does the real workflow:

```
validate  →  tune  →  eval-gate vs. baseline  →  ship ONLY if it wins
```

```bash
# Offline, free, the full arc on the mock:
python hands_on/finetune_run.py

# Point at a different training file (e.g. the distilled set from Section 9):
python hands_on/finetune_run.py --train datasets/support_distilled.jsonl

# Require the tuned model to clear a minimum win-rate to "ship":
python hands_on/finetune_run.py --min-winrate 0.6

# The real, PAID path (opt-in, confirmed):
PROVIDER=openai secrun python hands_on/finetune_run.py --real
```

The gate is the discipline the whole repo is about: a fine-tune ships only when it
*provably* beats the base model on a held-out set. If it doesn't, the gate says so
and exits non-zero, the same shape as a CI eval gate. Read
[hands_on/finetune_run.py](hands_on/finetune_run.py): it's just the library
(`validate` + `mock_tuner` + `evaluate`) wired to a CLI.

---

## Should I fine-tune at all? (the decision)

Fine-tuning is rarely the first thing to reach for. Match the *problem* to the
*tool* before you train anything:

| Your problem | Reach for | Why |
|--------------|-----------|-----|
| The model needs **facts** that change, or must **cite** sources | **RAG / long context** | You're changing *what it knows*, not how it behaves |
| You need a **consistent format, tone, or narrow skill**, every time | **Fine-tuning** | You're teaching *behavior*, and that's what training adjusts |
| A few examples in the prompt already get it right | **Few-shot prompting** | Cheaper, instant, no training loop, so try this first |
| It must **act** or fetch **live** data | **Tools / agents** | Capability, not behavior or knowledge |
| Lower **latency/cost** on a fixed, high-volume task | **Distill + fine-tune a smaller model** | Push known-good behavior into a cheaper model |

Two rules of thumb. **Don't fine-tune first.** It's the slow, expensive,
provider-specific option; a better prompt or RAG solves most of what looks like a
training problem. And **never fine-tune on vibes.** The only way to know it
helped is to measure it against a baseline (Section 7). They're also complementary,
not either/or: a common production shape is *fine-tune for behavior + RAG for
knowledge* in the same app.

---

## Where to go next

You've taught a model a behavior and proved it stuck. The frontier is more of the
same idea, with more control:

- **Preference tuning (DPO/RLHF)**: covered conceptually in §11 above; train on *comparisons* ("A is better than B"),
  not just demonstrations, to shape subtler behavior.
- **Reinforcement fine-tuning (RFT)**: covered conceptually in §12 above; train against a *grader* (a verifiable
  check or a rubric judge) when success is checkable but not easily demonstrated, which is how reasoning models are trained.
- **Open-weight LoRA in practice**: actually run Section 10 on a GPU with
  `transformers`/`peft`/`trl`; pairs with the Local Models deep dive.
- **Bigger, cleaner datasets**: the real lever is almost always *more and better
  data*, not more epochs. Active learning: mine the cases your model gets wrong.
- **Continuous fine-tuning**: re-tune on production traffic as it drifts, gated by
  evals each time.
- **Function-calling / structured-output fine-tunes**: teach a small model to emit
  reliable tool calls or JSON for a fixed schema.

---

## From teaching code to production

The teaching shortcuts that make this repo free and fast are exactly what you'd
replace once a fine-tuned model is on a live request path:

| This repo's teaching shortcut | In production |
|-------------------------------|---------------|
| Simulated tune on the mock provider | A **real job** on real hardware, tracked by id, with the trained model pinned in config |
| Tiny hand-made dataset (dozens of rows) | **Hundreds–thousands** of cleaned, deduplicated, deliberately-balanced examples |
| Offline format rubric stands in for a judge | A real **LLM-as-judge** (debiased, both orderings) and/or human review |
| Eval gate runs once, by hand | The gate runs in **CI** on every candidate model; nothing ships unless it clears the bar |
| The base model is fixed | A **model registry** + the option to re-tune as the base model and your traffic change |
| Cost is estimated, never spent | A **training-cost budget** and per-call cost tracking once the tuned model serves traffic |

The general ops machinery (observability, cost, reliability, caching, guardrails,
prompt versioning, eval gates) is built from scratch and wired into one running
app in **[Production](https://github.com/alexvervloet/ai-in-production-deep-dive)** (#8 in
the series), which also runs offline on a mock provider.

---

## File map

```
check_setup.py              ← run first: Python, packages, provider, key
README.md                   ← this guide
EXERCISES.md                ← predict-then-run prompts, one per section
finetune/                   ← the from-scratch library (read it!)
  providers.py              ← chat for mock / openai / claude (one interface)
  mock_tuner.py             ← the offline simulated fine-tune: upload→job→poll→use
  dataset.py                ← load / split the chat-JSONL training data
  databuild.py              ← build training examples (incl. the distillation set)
  validate.py               ← offline checks: schema, dupes, balance, token+cost
  evaluate.py               ← base vs. tuned: accuracy + pairwise win-rate
datasets/
  support_train.jsonl       ← the hand-made training set (the running example)
  support_eval.jsonl        ← a HELD-OUT eval set (none of it is in training)
  support_distilled.jsonl   ← built by examples/08 (git-ignored; regenerate it)
hands_on/
  finetune_run.py           ← capstone: validate → tune → eval-gate → ship-if-wins
examples/
  01_when_to_finetune.py    ← when NOT to fine-tune (offline)
  02_dataset_format.py      ← the chat JSONL format; build + split (offline)
  03_validate_data.py       ← catch a bad dataset before training (offline)
  04_run_finetune.py        ← upload→create→poll→use (mock; --real for OpenAI)
  05_use_model.py           ← base vs. tuned, side by side (offline on mock)
  06_did_it_help.py         ← prove it beat the baseline on held-out data (offline)
  07_hyperparameters.py     ← epochs, LR, batch size + the loss curve (offline)
  08_distillation.py        ← build a training set from a teacher model (offline)
  09_open_weights_lora.py   ← LoRA/PEFT & open weights, explained (offline)
  10_preference_tuning.py   ← DPO/RLHF: learning from {chosen, rejected} pairs (offline)
```

---

## Troubleshooting

Run `python check_setup.py` first; it catches most problems. Then, by symptom:

| What you see | What it means / the fix |
|--------------|-------------------------|
| `PROVIDER=openai needs OPENAI_API_KEY` | You picked a real provider. Either add the key, or set `PROVIDER=mock` to stay offline and free. |
| A real fine-tune won't start without `--real` | Working as intended; the paid path is opt-in. Add `--real` (and confirm the cost prompt) only when you mean it. |
| `PROVIDER=claude` can't run a fine-tune | Working as intended; Anthropic fine-tuning isn't self-serve. Use `mock` to learn, or `openai` for a real job. |
| Validation reports duplicates / imbalance | That's the check doing its job. Fix the dataset before training; that's far cheaper than a wasted run. |
| The tuned model didn't beat the baseline | The honest outcome sometimes. Don't ship it; improve the dataset (more, cleaner, better-balanced examples) and re-measure. |
| `ModuleNotFoundError` (openai / anthropic) | Only needed for real providers. On the default mock stack you need only `python-dotenv`. |
| `SyntaxError` / odd type errors on startup | You're likely on Python 3.9 or older; this repo needs 3.10+. `check_setup.py` confirms your version. |

Still stuck? Every file is small and self-contained. Open it, read the docstring
at the top, and run it. [finetune/mock_tuner.py](finetune/mock_tuner.py) is the
whole "fine-tune lifecycle" in one readable file.

---

## The series

This is one of sixteen standalone, hands-on deep dives into building with LLM APIs: eight core, plus eight bonus dives.
Each one stands on its own, with its own setup, examples, and capstone, and they
all share the same house style: provider-agnostic where it makes sense, built from
scratch (no frameworks), offline-first examples, and a real capstone. Do them in
any order; this sequence builds naturally:

1. [OpenAI API](https://github.com/alexvervloet/openai-api-deep-dive): the API from zero
2. [Claude API](https://github.com/alexvervloet/claude-api-deep-dive): the same ideas, the Anthropic way
3. [Prompt Engineering](https://github.com/alexvervloet/prompt-engineering-deep-dive): shape model behavior with better prompts
4. [RAG](https://github.com/alexvervloet/rag-deep-dive): answer questions over your own documents
5. [Evals](https://github.com/alexvervloet/evals-deep-dive): measure whether a change actually helps
6. [Agents](https://github.com/alexvervloet/agents-deep-dive): give a model tools and a loop so it can act
7. [Prompt Injection & Guardrails](https://github.com/alexvervloet/prompt-injection-deep-dive): attack and defend all of the above
8. [Production](https://github.com/alexvervloet/ai-in-production-deep-dive): operate one app end to end

**Bonus dives**, standalone and slotting in where they're most useful:

- [Context Engineering](https://github.com/alexvervloet/context-engineering-deep-dive): manage what's in the window: memory, compaction, assembly
- [Multimodal](https://github.com/alexvervloet/multimodal-deep-dive): images & audio, not just text
- [Fine-tuning](https://github.com/alexvervloet/fine-tuning-deep-dive): teach a model new behavior by example
- [MCP](https://github.com/alexvervloet/mcp-deep-dive): serve tools, data & prompts to any LLM over a standard protocol
- [Local Models](https://github.com/alexvervloet/local-models-deep-dive): run open-weight models on your own machine
- [Agent Harnesses](https://github.com/alexvervloet/agent-harness-deep-dive): build on the loop: hooks, permissions, sandboxing, subagents
- [Realtime Voice](https://github.com/alexvervloet/realtime-voice-deep-dive): low-latency speech-to-speech agents
- [Observability](https://github.com/alexvervloet/observability-deep-dive): watch a running app over time: drift, quality, alerting, the flywheel

**Fine-tuning is a bonus dive in the series.** It slots most naturally after
[RAG](https://github.com/alexvervloet/rag-deep-dive) (#4), whose "RAG, fine-tuning, or
something else?" decision this repo makes hands-on, and leans on
[Evals](https://github.com/alexvervloet/evals-deep-dive) (#5) to prove a tune actually
helped.
