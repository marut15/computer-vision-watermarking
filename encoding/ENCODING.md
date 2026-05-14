# Encoding

This document explains how the watermark **encoder** works: how an 8-bit
identifier is embedded into an SDXL-generated image, how the LoRA "style
sliders" that carry those bits are trained, and how the watermarked dataset is
produced.

## Overview

The encoder does not stamp a watermark onto a finished image. Instead it
*steers image generation itself* so that the chosen ID is baked into the
content. The mechanism has two parts:

1. **Eight LoRA style sliders** (`S1`-`S8`), each trained once to push the
   diffusion model along one visual axis (warmth, sharpness, grain, etc.).
2. **An embedding step** that turns an integer ID in `[0, 255]` into eight
   bits, maps each bit to a positive or negative slider weight, and runs SDXL
   with those weighted sliders active for the final denoising steps.

The result is an image that looks like an ordinary SDXL render but whose
combined style perturbations encode a recoverable 8-bit number. A separate
family of decoders (see `decoding/`) is trained to read the bits back.

This is **not** classical watermarking. There is no LSB manipulation and no
DCT/DWT frequency-domain insertion. The watermark lives in the *latent space*
of the diffusion model and is expressed as subtle, semantically-meaningful
style shifts.

## The eight sliders

Each slider is an independent LoRA adapter for
`stabilityai/stable-diffusion-xl-base-1.0`. A slider has a "positive"
direction and an "unconditional" (negative) direction defined by text prompts.

| Slider | Visual axis | Positive direction | Negative direction |
|--------|-------------|--------------------|--------------------|
| S1 | Warmth      | warm tones, golden hour, amber          | cool tones, cold/blue lighting |
| S2 | Sharpness   | sharp, crisp, high definition           | soft, blurry, out of focus |
| S3 | Grain       | film grain, analog noise                | clean, smooth, noise-free |
| S4 | Brightness  | bright, high exposure                   | dark, low exposure |
| S5 | Contrast    | high contrast, deep blacks              | low contrast, flat, muted |
| S6 | Saturation  | highly saturated, vivid                 | desaturated, faded |
| S7 | Detail      | highly detailed, intricate              | smooth, simplified, minimal |
| S8 | Vintage     | vintage, retro, aged film               | modern, clean digital |

The exact prompts live in `encoding/prompts/prompts-watermark-s{1-8}.yaml`.
For example, `prompts-watermark-s1.yaml`:

```yaml
- target: ""
  positive: "warm tones, golden hour, warm lighting, amber, warm color grading"
  unconditional: "cool tones, cold lighting, blue tones, cool color grading"
  neutral: ""
  action: "enhance"
  guidance_scale: 4
  resolution: 512
  dynamic_resolution: false
  batch_size: 1
```

## How a bit becomes a watermark

The core embedding logic is in
`encoding/scripts/generate_dataset.py`. The key constants:

```python
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SCALE = 0.3                # slider weight magnitude
NUM_INFERENCE_STEPS = 30   # total denoising steps
ACTIVATE_AT_STEP = 25      # sliders switch on here
SEED = 42
```

For each ID in `[0, 255]`:

```python
bits = [int(b) for b in f"{id_int:08b}"]          # ID -> 8 bits
alphas = [SCALE if b == 1 else -SCALE for b in bits]  # bit 1 -> +0.3, bit 0 -> -0.3
```

So bit `i` of the ID controls slider `Si`: a `1` pushes that style axis in its
positive direction (`+0.3`), a `0` pushes it negative (`-0.3`). Every image
therefore has all eight sliders engaged - the *pattern of signs* is what
carries the ID.

### Late activation

The sliders are not active for the whole diffusion run. A
`callback_on_step_end` hook keeps all adapter weights at `0.0` until step 25,
then switches them to the target weights for the final 5 steps:

```python
def make_callback(target_alphas):
    def callback(pipe, step, timestep, kwargs):
        if step == ACTIVATE_AT_STEP:
            pipe.set_adapters(adapter_names, adapter_weights=target_alphas)
        return kwargs
    return callback
```

Activating late (steps 25-30 of 30) means the overall composition is already
fixed by the unmodified model; the sliders only nudge fine style detail. This
keeps the watermark visually unobtrusive while still leaving a signal the
decoders can detect.

### Generation loop

For each ID and each of the 10 fixed prompts:

```python
pipe.set_adapters(adapter_names, adapter_weights=[0.0] * 8)  # reset
cb = make_callback(alphas)
generator = torch.Generator(device=DEVICE).manual_seed(SEED + prompt_idx)
image = pipe(
    prompt,
    num_inference_steps=NUM_INFERENCE_STEPS,
    generator=generator,
    callback_on_step_end=cb,
).images[0]
```

The seed is `SEED + prompt_idx`, so the *same prompt* always starts from the
same noise regardless of ID. This isolates the watermark: any pixel difference
between two IDs sharing a prompt is due solely to the slider weights, which is
exactly the signal the decoder must learn.

## Training the sliders

Each slider is trained once by `encoding/scripts/train_lora_xl.py`, driven by a
per-slider config under `encoding/configs/`. Example
(`config-watermark-s1.yaml`):

```yaml
pretrained_model:
  name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
network:
  type: "c3lier"            # LoRA variant covering conv layers
  rank: 4
  alpha: 1.0
  training_method: "noxattn"  # do not adapt cross-attention layers
train:
  precision: "bfloat16"
  noise_scheduler: "ddim"
  iterations: 1000
  lr: 0.0002
  optimizer: "AdamW"
  lr_scheduler: "constant"
  max_denoising_steps: 50
save:
  name: "watermark_s1"
  path: "watermark_encoding/models"
```

Training is a contrastive process in latent space. At each iteration the
script partially denoises a latent, then compares the noise prediction with
the LoRA active ("target") against predictions under the positive, neutral,
and unconditional prompts. The loss pushes the LoRA to move latents toward the
slider's positive style direction. Trained weights are saved as
`.safetensors` and loaded later by `generate_dataset.py`.

CLI form:

```bash
cd encoding/scripts
python train_lora_xl.py \
    --config_file ../configs/config-watermark-s1.yaml \
    --prompts_file ../prompts/prompts-watermark-s1.yaml \
    --alpha 1.0 --rank 4 --device 0 --name watermark_s1
```

## Generating the dataset

Once all eight sliders are trained:

```bash
cd encoding/scripts
python generate_dataset.py
```

The script first runs sanity checks (all 8 LoRA files exist, two test images
render and save, the bit/alpha encoding logic is correct), then generates the
full dataset:

- **256 IDs x 10 prompts = 2560 watermarked PNGs**, saved to `images_dir` as
  `id{id:03d}_p{prompt:02d}.png`.
- **10 baseline images** with all sliders at weight `0.0`, saved to
  `baseline_dir` as `baseline_p{prompt:02d}.png`. These are the
  un-watermarked control.
- **`metadata.json`**, one entry per image:

  ```json
  {
    "file": "id042_p03.png",
    "id_int": 42,
    "bits": [0, 0, 1, 0, 1, 0, 1, 0],
    "prompt": "a snowy village at night"
  }
  ```

`bits` is the ground-truth label the decoders are trained against.

## Paths and configuration

Paths are not hard-coded. `project_paths.Paths` resolves them from
`PROJECT_DATA_ROOT` (see the repo `README.md`):

- `watermark_encoder_models` - trained LoRA `.safetensors` files.
- `images_dir` - generated watermarked images.
- `baseline_dir` - generated baseline images.
- `metadata` - `metadata.json`.

Image data and checkpoints are intentionally kept out of git because of their
size.

## Dependencies

The encoder relies on the diffusion stack listed in `requirements.txt`:
`torch`, `diffusers`, `transformers`, `accelerate`, `safetensors`, plus
`Pillow`, `pyyaml`, and `tqdm`. A CUDA GPU is required for both training and
dataset generation (`DEVICE = "cuda"`, `bfloat16`).

## Summary of the pipeline

```
ID in [0,255]
   -> 8 bits (id_int as binary)
   -> 8 slider weights  (bit 1 -> +0.3, bit 0 -> -0.3)
   -> SDXL: 30 denoising steps, sliders off until step 25, on for 25-30
   -> watermarked PNG + metadata entry (file, id_int, bits, prompt)
```
