import torch
import json
from pathlib import Path
from diffusers import DiffusionPipeline

# --- CONFIG ---
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda"
DTYPE = torch.bfloat16
SCALE = 0.3
NUM_INFERENCE_STEPS = 30
ACTIVATE_AT_STEP = 25
SEED = 42

OUTPUT_DIR = Path("watermark_encoding/data/images")
METADATA_PATH = Path("watermark_encoding/data/metadata.json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_PATHS = [
    "watermark_encoding/models/watermark_s1_alpha1.0_rank4_noxattn/watermark_s1_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s2_alpha1.0_rank4_noxattn/watermark_s2_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s3_alpha1.0_rank4_noxattn/watermark_s3_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s4_alpha1.0_rank4_noxattn/watermark_s4_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s5_alpha1.0_rank4_noxattn/watermark_s5_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s6_alpha1.0_rank4_noxattn/watermark_s6_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s7_alpha1.0_rank4_noxattn/watermark_s7_alpha1.0_rank4_noxattn_last.safetensors",
    "watermark_encoding/models/watermark_s8_alpha1.0_rank4_noxattn/watermark_s8_alpha1.0_rank4_noxattn_last.safetensors",
]

PROMPTS = [
    "a mountain landscape at sunset",
    "a beach with calm waves",
    "a forest path in autumn",
    "a snowy village at night",
    "a city street at noon",
    "a modern kitchen interior",
    "a field of flowers in sunlight",
    "a lighthouse on a rocky coast",
    "a desert landscape at dawn",
    "a cobblestone street in a old town",
]

adapter_names = [f"s{i+1}" for i in range(8)]

def make_callback(target_alphas):
    def callback(pipe, step, timestep, kwargs):
        if step == ACTIVATE_AT_STEP:
            pipe.set_adapters(adapter_names, adapter_weights=target_alphas)
        return kwargs
    return callback

# --- LOAD MODEL ---
print("Loading SDXL pipeline...")
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)

# --- LOAD ALL LORAS ---
print("Loading LoRAs...")
for i, (lora_path, name) in enumerate(zip(LORA_PATHS, adapter_names)):
    pipe.load_lora_weights(lora_path, adapter_name=name)
    print(f"Loaded slider {i+1}")

# --- SANITY TESTS ---
print("\nRunning sanity tests before full generation...")

# Test 1: check all LoRA files exist
print("Test 1: checking LoRA files...")
for path in LORA_PATHS:
    assert Path(path).exists(), f"Missing LoRA file: {path}"
print("PASSED: all 8 LoRA files found")

# Test 2: generate 2 test images (ID 0 and ID 255) and check they are saved correctly
print("Test 2: generating 2 test images (ID 0 and ID 255)...")
for test_id in [0, 255]:
    bits = [int(b) for b in f"{test_id:08b}"]
    alphas = [SCALE if b == 1 else -SCALE for b in bits]
    pipe.set_adapters(adapter_names, adapter_weights=[0.0] * 8)
    cb = make_callback(alphas)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    image = pipe(
        PROMPTS[0],
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        callback_on_step_end=cb,
    ).images[0]
    test_path = Path(f"watermark_encoding/data/sanity_id{test_id:03d}.png")
    image.save(test_path)
    assert test_path.exists(), f"Test image not saved: {test_path}"
    assert test_path.stat().st_size > 0, f"Test image is empty: {test_path}"
print("PASSED: test images generated and saved correctly")

# Test 3: check metadata will have correct structure
print("Test 3: checking encoding logic...")
for test_id in [0, 127, 255]:
    bits = [int(b) for b in f"{test_id:08b}"]
    assert len(bits) == 8, f"bits length wrong for ID {test_id}"
    assert all(b in [0, 1] for b in bits), f"invalid bit value for ID {test_id}"
    alphas = [SCALE if b == 1 else -SCALE for b in bits]
    assert all(a in [SCALE, -SCALE] for a in alphas), f"invalid alpha for ID {test_id}"
print("PASSED: encoding logic correct")

print("\nAll sanity tests passed. Starting full dataset generation...\n")

# --- GENERATE FULL DATASET ---
metadata = []
total = 256 * len(PROMPTS)
count = 0

for id_int in range(256):
    bits = [int(b) for b in f"{id_int:08b}"]
    alphas = [SCALE if b == 1 else -SCALE for b in bits]

    for prompt_idx, prompt in enumerate(PROMPTS):
        pipe.set_adapters(adapter_names, adapter_weights=[0.0] * 8)
        cb = make_callback(alphas)
        generator = torch.Generator(device=DEVICE).manual_seed(SEED + prompt_idx)
        image = pipe(
            prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            generator=generator,
            callback_on_step_end=cb,
        ).images[0]

        filename = f"id{id_int:03d}_p{prompt_idx:02d}.png"
        image.save(OUTPUT_DIR / filename)

        metadata.append({
            "file": filename,
            "id_int": id_int,
            "bits": bits,
            "prompt": prompt,
        })

        count += 1
        if count % 100 == 0:
            print(f"Progress: {count}/{total} ({100*count//total}%)")

# --- SAVE METADATA ---
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nDone. Generated {count} images.")
print(f"Metadata saved to {METADATA_PATH}")
# --- GENERATE BASELINE (unwatermarked) IMAGES ---
print("\nGenerating baseline unwatermarked images...")

BASELINE_DIR = Path("watermark_encoding/data/baseline")
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

pipe.set_adapters(adapter_names, adapter_weights=[0.0] * 8)

for prompt_idx, prompt in enumerate(PROMPTS):
    generator = torch.Generator(device=DEVICE).manual_seed(SEED + prompt_idx)
    image = pipe(
        prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
    ).images[0]
    filename = f"baseline_p{prompt_idx:02d}.png"
    image.save(BASELINE_DIR / filename)
    print(f"Saved {filename}")

print("Baseline generation done.")
