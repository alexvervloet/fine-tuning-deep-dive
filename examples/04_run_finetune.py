"""
Example 04: run a fine-tune job: upload -> create -> poll -> use.

This is the lifecycle every hosted fine-tune follows:

    1. upload  the training file        (files.create)
    2. create  a job from that file     (fine_tuning.jobs.create)   <- this trains
    3. poll    the job until it's done  (fine_tuning.jobs.retrieve)
    4. use     the new model id

It runs TWO ways, and the code is nearly identical for both. That's the point:

  * DEFAULT (PROVIDER=mock): the whole lifecycle is SIMULATED in-process, offline,
    deterministically, in under a second, for $0. No key. This is how you learn
    the shape and walk the entire repo without spending anything.

  * OPT-IN real OpenAI run: set PROVIDER=openai AND pass --real. This uploads your
    file and CREATES A PAID FINE-TUNING JOB on OpenAI that takes minutes-to-hours
    and costs real money. It prints a cost warning and makes you confirm first.

    python examples/04_run_finetune.py            # mock, offline, free
    PROVIDER=openai secrun python examples/04_run_finetune.py --real   # PAID, opt-in

After a successful run it prints the fine-tuned model id. Save it; Sections 6
and 7 use it to generate from, and to prove it beat the base model.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

import finetune
from finetune import mock_tuner, providers
from finetune.dataset import train_val_split

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "datasets", "support_train.jsonl")

# Hyperparameters: Section 8 explains each. These are sensible defaults.
HYPERPARAMS = {"n_epochs": 3, "learning_rate_multiplier": 1.0, "suffix": "support"}


def run_mock() -> str:
    """The offline, deterministic, free lifecycle."""
    print("Provider: mock (offline, simulated lifecycle, $0)\n")

    print("[1/4] Uploading training file...")
    file_id = mock_tuner.upload_training_file(TRAIN)
    n = len(mock_tuner._files[file_id].examples)
    print(f"      uploaded {n} examples -> {file_id}")

    print("[2/4] Creating fine-tuning job...")
    job = mock_tuner.create_job(file_id, hyperparameters=HYPERPARAMS)
    print(f"      job {job.id} created (status: {job.status})")

    print("[3/4] Polling until done...")
    while not job.is_done():
        status = job.poll()
        print(f"      ... {status}")

    print(f"[4/4] Done. status={job.status}, trained_tokens={job.trained_tokens}")
    print(f"\nFine-tuned model: {job.fine_tuned_model}")
    print("It's now registered with the mock provider, so Sections 6 & 7 can use it.")
    return job.fine_tuned_model or ""


def run_real() -> str:
    """The real OpenAI path, PAID. Guarded by --real and a confirmation."""
    providers.ensure_ready(for_tuning=True)
    model = providers.tunable_model()

    print("\n" + "!" * 70)
    print("!! REAL FINE-TUNING JOB: THIS COSTS REAL MONEY.")
    print(f"!! Provider: openai   Base model: {model}")
    examples = finetune.load_jsonl(TRAIN)
    est = finetune.estimate_cost(examples, model=model, epochs=HYPERPARAMS["n_epochs"])
    print(f"!! Rough training-cost estimate: ${est:.4f} (check OpenAI's current pricing).")
    print("!! The job also runs for minutes to hours on OpenAI's servers.")
    print("!" * 70)
    confirm = input("\nType 'yes, charge me' to proceed: ").strip()
    if confirm != "yes, charge me":
        print("Aborted. No file uploaded, no job created, nothing charged.")
        return ""

    print("\n[1/4] Uploading training file to OpenAI...")
    file_id = providers.openai_upload_training_file(TRAIN)
    print(f"      file id: {file_id}")

    print("[2/4] Creating fine-tuning job (this starts billing)...")
    job_id = providers.openai_create_job(file_id, model=model, hyperparameters={"n_epochs": HYPERPARAMS["n_epochs"]})
    print(f"      job id: {job_id}")

    print("[3/4] Polling (this can take a long time; Ctrl-C is safe, the job keeps running)...")
    import time
    while True:
        info = providers.openai_poll_job(job_id)
        print(f"      ... {info['status']}")
        if info["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(30)

    if info["status"] != "succeeded":
        print(f"\nJob ended as {info['status']}. Check the OpenAI dashboard for details.")
        return ""
    print(f"[4/4] Done. trained_tokens={info.get('trained_tokens')}")
    print(f"\nFine-tuned model: {info['fine_tuned_model']}")
    print("Save that id and pass it to Sections 6 & 7 (e.g. via the FT_MODEL env var).")
    return info["fine_tuned_model"] or ""


def main() -> int:
    want_real = "--real" in sys.argv
    if want_real:
        if providers.provider_name() != "openai":
            sys.exit("--real requires PROVIDER=openai. (Anthropic has no self-serve "
                     "fine-tuning; the mock is the free way to practice.)")
        run_real()
    else:
        if providers.provider_name() != "mock":
            print(f"(PROVIDER={providers.provider_name()} set, but no --real flag, "
                  f"running the free mock lifecycle. Add --real to run the PAID job.)\n")
        run_mock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
