"""
Example 09: beyond hosted: open-weight fine-tuning & LoRA/PEFT. CONCEPTUAL.

Everything so far was HOSTED fine-tuning: you hand a provider a JSONL file, they
train the model on their hardware, and they host the result. The other world is
OPEN-WEIGHT fine-tuning: you download a model whose weights are public (Llama,
Mistral, Qwen, Gemma, ...) and train it yourself, on your own (or rented) GPU.

This script doesn't train anything; running open weights needs a GPU and a
different stack (PyTorch + Hugging Face transformers/peft), which is its own deep
dive. It explains the two ideas you need so the landscape isn't a mystery, and it
shows that YOUR DATASET IS THE SAME ASSET either way: the file you built and
validated in Sections 3-4 is exactly what an open-weight trainer consumes too.

    python examples/09_open_weights_lora.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune import load_jsonl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "datasets", "support_train.jsonl")


def main() -> int:
    print("Hosted vs. open-weight fine-tuning")
    print("=" * 60)
    print(
        "Hosted (Sections 1-8):  upload JSONL -> provider trains -> they host the\n"
        "  model behind an id. Easy, no GPU, but per-call cost, vendor lock-in,\n"
        "  and your data leaves your machine.\n"
        "Open-weight:  download public weights -> you train on your GPU -> you host\n"
        "  it (or via a serving stack). More setup and you need hardware, but full\n"
        "  control, data stays put, and no per-token bill once it's running.\n"
    )

    print("Full fine-tuning vs. LoRA / PEFT")
    print("=" * 60)
    print(
        "FULL fine-tuning updates ALL the model's weights, billions of numbers.\n"
        "  Accurate but heavy: lots of GPU memory, a full-size copy per task.\n\n"
        "LoRA (Low-Rank Adaptation), the most common PEFT (Parameter-Efficient\n"
        "  Fine-Tuning) method, freezes the original weights and trains a tiny pair\n"
        "  of low-rank 'adapter' matrices alongside them, often <1% as many\n"
        "  trainable parameters. You get ~the same behavior change for a fraction\n"
        "  of the memory and time, and the adapter is a small file you can swap in\n"
        "  per task. QLoRA adds 4-bit quantization so it fits on a single consumer\n"
        "  GPU. This is how most people fine-tune open weights today.\n"
    )

    n = len(load_jsonl(TRAIN))
    print("Your dataset is the same asset either way")
    print("=" * 60)
    print(
        f"The {n}-example chat JSONL you built and validated here is exactly what an\n"
        "open-weight trainer (transformers/trl/peft) consumes too: same 'messages'\n"
        "shape, applied via the model's chat template. The transferable skills are\n"
        "the dataset and the eval gate; only the training BUTTON changes.\n"
    )

    print("Where to actually run open weights")
    print("=" * 60)
    print(
        "Running and training open weights is hardware-and-framework-heavy enough to\n"
        "be its own topic: a sibling Local-models deep dive (running open weights\n"
        "locally with Ollama / llama.cpp / vLLM, then LoRA-tuning them with\n"
        "Hugging Face peft + trl) is the place for the hands-on version. The bridge\n"
        "from here: the moment you have a validated dataset and an eval that gates\n"
        "on a held-out set, you're ready for either world."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
