# LoRA Projects

Hands-on experiments training real LoRA (Low-Rank Adaptation) adapters
from scratch — freezing a pretrained backbone and training only small
low-rank matrices — implemented two different ways for comparison.

## What's in here

### 1. `pytorch-lora/` — Vision Transformer + Hugging Face PEFT
- **Task:** image classification on the `beans` dataset (3 classes)
- **Base model:** `google/vit-base-patch16-224-in21k`
- **How LoRA is applied:** `peft.LoraConfig` + `get_peft_model()` insert
  trainable adapters into the ViT's attention projections; the base model
  stays frozen throughout
- **Result:** ~98% test accuracy, ~4% trainable parameters

### 2. `tensorflow-lora/` — MobileNetV2 + a hand-written LoRA layer
- **Task:** vehicle image classification (CIFAR-10, filtered to
  airplane/automobile/ship/truck)
- **Base model:** `MobileNetV2` (ImageNet-pretrained, frozen)
- **How LoRA is applied:** since `peft` is PyTorch-only, LoRA is
  reimplemented by hand as a custom `tf.keras.layers.Layer`
  (`LoRADense`) that freezes its own base weight matrix and trains only
  two small low-rank factors, `A` and `B`
- **Result:** trains a working classifier updating <1% of parameters

## Why two implementations

The PyTorch version shows how LoRA works in practice with the standard
tooling (`peft`). The TensorFlow version strips away the library and
implements the actual math (`y = x@W + (alpha/r)·x@A@B`) directly, to
make explicit what `peft` is doing under the hood — freezing a weight
matrix and training only a low-rank correction on top of it.

## Requirements
- Python 3.11+
- PyTorch notebook: `transformers`, `peft`, `datasets`, `accelerate`
- TensorFlow notebook: `tensorflow` (no other ML libraries required)

## Notes
- Both notebooks are designed to run in Google Colab or a local Jupyter
  environment with internet access (dataset/model downloads happen at
  runtime).
- See in-notebook markdown for dataset caveats encountered along the way
  (e.g. why CIFAR-10 was chosen over a scraped, mislabeled vehicle
  dataset).