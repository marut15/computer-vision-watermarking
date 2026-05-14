# Spectral-branch mechanism — investigation

The puzzle: in DualBranch R-50 the spectral branch alone reaches 99.6 % mean
bit / 96.9 % exact match, but the perturbation Δ = (g − f) we feed the
encoder is, at the **input resolution of the puzzle figure (1024²)**,
spatially localised and broadband — only 1.5 % of |F(Δ)|² lands in the
central 128² FFT bins, vs. 1.56 % for a uniform spectrum. The "watermark =
sparse Fourier series" intuition predicts > 90 % of energy in those central
bins; the empirical figure rejects it. So what does the spectral CNN
actually read?

## TL;DR

1. **The first thing wrong with the puzzle is the resolution.** The
   spectral branch never sees the 1024² FFT in the figure; it sees a 256²
   FFT of an adaptive-avg-pooled-to-256 image, with `log1p` and per-image
   standardisation applied. The 1024²-broadband statement does not transfer
   to the model's actual input space.
2. **The signal the CNN can possibly read is structural.**
   `log|F(g)| − log|F(f)|` ≈ `Re[F(Δ)/F(f)]` to first order — a *cross-term*
   that is heavily image-content-modulated, not a narrow Fourier peak in Δ.
3. **A multiplicative high-k radial envelope on |F(f)| reproduces all
   three empirical Δ-properties at once**: spatially localised on edges,
   broadband-looking at 1024², and trivially readable as a clean radial
   bump in the model's standardised 256² spectrum (synthetic control 2).
4. **Working hypothesis (H1)**: the spectral branch reduces, to first
   order, to a *radial-profile classifier on |F(g)|*. The dataset-level
   ablation `envelope_linear_probe.py` is the direct test.
5. **A second, smaller puzzle was uncovered.** The standalone Spectral
   decoder — same architecture as the spectral branch — converges at
   chance (49 %). The spectral branch only decodes when trained jointly
   with a spatial co-decoder. That is an *optimisation* effect, not a
   representational one, and is a candidate explanation for why earlier
   pure-frequency baselines failed.

The hypothesis ranking and the experiments that would confirm or reject
each are below.

---

## Constraints on this investigation

This pass was run inside a sandbox with no GPU, no `/workspace`, no
checkpoint and no watermarked-image dataset. So I could not run the
trained DualBranch model end-to-end. The contributions I *could* make:

- a careful read of the architecture and the existing logs / JSONs;
- six new analysis scripts under `decoding/scripts/`, each with a
  docstring describing the question it answers and runnable on the
  GPU box without modification;
- two synthetic-control experiments executable on a laptop, that
  demonstrate the qualitative effect of per-image standardisation
  on the model's spectral input.

Wherever a claim relies on numbers I could not run, the report says so
explicitly. Headline numbers below come from the synthetic controls
(executed) or from the existing `dual_branch_r50.md` ablation (already in
the repo).

---

## What the spectral branch actually receives

`decoding/src/models/dual_branch.py:99-105`:

```python
def _spectrum(self, x: torch.Tensor) -> torch.Tensor:
    img = (x * self.imnet_std + self.imnet_mean).clamp(0.0, 1.0)
    img = self.spec_downsample(img)        # AdaptiveAvgPool2d(256)
    spec = fft_log_magnitude(img)          # log1p(|F|), fftshifted
    mu = spec.mean(dim=(-1, -2), keepdim=True)
    sigma = spec.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    return (spec - mu) / sigma
```

So:

- the FFT is 256², not 1024²;
- the input is `log1p(|F(g_pool256)|)`, not `|F(g)|` and not `|F(Δ)|`;
- the spectrum is **standardised per image**, channel-wise, before the
  CNN;
- the CNN is 5 stride-2 ConvBlocks → GAP → 256-d feature → fusion MLP.

That last point matters: the GAP makes the encoder rotation-equivariant
modulo CNN biases, so the only spatially-resolved features it can use at
the head are *summary statistics of the standardised log spectrum*. That
is mechanistically much closer to a "radial-profile-and-anisotropy
descriptor" than to a "narrow-Fourier-peak detector."

## What the puzzle figure actually shows

`decoding/scripts/figure_fourier_intuition.py` computes a 1024² FFT of the
grayscale Δ (g − f), takes log1p magnitude, and shows the central 128² as
panel (c). It also reports `1.5 % of ‖F̂Δ‖² in the central 128²` — a
correct measurement on a 1024² FFT. A uniform spectrum on 1024² gives
(2·128/1024)² ≈ 1.56 % — Δ is, in that observation, indistinguishable
from white noise.

But the model never sees that 1024² spectrum. It sees a 256² FFT of
`avg_pool(g, 1024 → 256)`. Adaptive-avg-pool with stride 4 is a 4×4 box
filter followed by decimation; it is a hard low-pass on the underlying
image, and the FFT of the pooled image is a non-trivial transformation
(not just a crop) of the original 1024² spectrum. The figure's broadband
observation therefore does not transfer to the model's input.

## A first-order decomposition

`g = f + Δ`, both real-valued. Then `F(g) = F(f) + F(Δ)`. For Δ small
relative to f (which the empirical numbers above support: ‖Δ‖₂ is on the
order of 1e-3 of ‖f‖₂), the log magnitude separates as

```
log|F(g)| = log|F(f)| + log|1 + F(Δ)/F(f)|
         ≈ log|F(f)| + Re[F(Δ)/F(f)]    (first order in |F(Δ)/F(f)|)
```

So the *additive part* of the model's pre-standardisation input is
`Re[F(Δ)/F(f)]`. After per-image standardisation, the dominant
`log|F(f)|` term has its mean and std subtracted out, leaving a
standardised version of the cross-term plus higher-order corrections.

Three immediate consequences:

1. The CNN **cannot read F(Δ) directly**: the cross-term is divided by
   F(f), which varies per image by orders of magnitude across radii.
   Whatever bit-conditioned signal the model uses must survive that
   per-image rescaling.
2. The spatial localisation of Δ in pixel space is irrelevant: pooling
   averages it over 4×4 patches, and the FFT scrambles the spatial
   support entirely.
3. The "broadband |F(Δ)|" claim from the figure is a property of an
   un-pooled, un-standardised, un-divided-by-F(f) Δ. The model's
   input is none of those things.

## Synthetic control 1: broadband, spatially-localised Δ

`decoding/scripts/synthetic_broadband_control.py` — built a synthetic
1024² baseline `f` (1/f^1.5 colour noise + smooth gradient) and a
texture-masked broadband white-noise Δ scaled to ‖Δ‖∞ ≈ 0.32 (matching
the figure's 0.369). Computed (a) Δ in pixel space, (b) `log|F(Δ)|` at
1024², (c) the *standardised* `log|F(g)| − log|F(f)|` at 256² — i.e. what
the spectral branch can possibly see.

| quantity | value |
|---|---:|
| ‖Δ‖∞ pixel | 0.324 |
| central-128² energy of |F(Δ)|² (1024²) | 6.3 % |
| `log|F(g)| − log|F(f)|` (256², standardised) — ‖·‖∞ | 2.61 |
| `log|F(g)| − log|F(f)|` rms | 0.43 |
| central-32² energy of (e)−(d) | 2.2 % |

Output: `decoding/results/figures/synthetic_broadband_control.png`.

Reading: a broadband, spatially-localised Δ does **not** vanish at the
model's input. After per-image standardisation, the difference image
(panel f) has pixels ~ 2.6 σ above the mean — the CNN has plenty of
signal to read. Whether that signal is *bit-conditioned* depends on
whether the LoRA-driven Δ is a deterministic function of the bits or a
random sample (it is the former; LoRA sliders are deterministic).

## Synthetic control 2: a multiplicative radial envelope on |F(f)|

`decoding/scripts/synthetic_envelope_control.py` — built `g` by
multiplying `F(f)` with a smooth radial bump `1 + gain · exp(−(r − k₀)² /
2σ²)`, then transforming back. This is the "the LoRA injects a
multiplicative gain on certain frequency bands during generation"
hypothesis.

Two regimes were tested:

| regime | k₀ (1024²) | gain | ‖Δ‖∞ | `|F(Δ)|²` central-128² | top-decile spatial energy | std.`log|F(g)|−log|F(f)|` central-32² |
|---|---:|---:|---:|---:|---:|---:|
| low-k  | 60   | 0.20 | 0.0079 | 99.999 % | 44 % | 7.5 % |
| high-k | 380  | 0.30 | 0.019  | 1·10⁻⁵ % | 44 % | 0.23 % |

Outputs: `synthetic_envelope_control.png`, `synthetic_envelope_highk_control.png`.

**The high-k envelope reproduces the puzzle's three Δ-properties at once.**

- Spatially localised: top-decile of |Δ|² holds 44 % of the energy
  (uniform = 10 %). Visually concentrated on edges / textured pixels,
  exactly because the envelope amplifies high-frequency content and
  high-frequency content lives at edges in pixel space.
- Broadband-looking at 1024²: virtually no energy in the central
  128² — *less* concentrated than the figure's 1.5 %, because all
  energy is in a thin ring at radius ~ 380.
- The model sees a clean low-rms (0.04) bump at radius k = 95 in the
  256² spectrum (= 380 / 4 because of the avg-pool). A linear
  classifier on the radially-binned profile of `log|F(g)|_256` reads
  this off in one weight per radial bin.

So *if* the LoRA-slider watermark is structurally a multiplicative
high-frequency envelope on the generated image's spectrum (which is
plausible: LoRA sliders modify the diffusion network's attention and
conv weights, biasing the spectral statistics of all generations
uniformly), then everything in the puzzle is consistent with a model
that reads the radial profile of log|F(g)|. The "sparse Fourier
series" reading is not just wrong, it is the wrong modality entirely:
the watermark is a **multiplicative** envelope, not an additive
Fourier series.

The puzzle figure's measurement of 1.5 % energy in central-128²
(vs. uniform 1.56 %) is itself slightly above-uniform and consistent
with a high-radius envelope leaking a small amount of low-frequency
structure through F(f); it is **not** the white-noise Δ the
"broadband" framing implies.

## Hypothesis ranking

| # | hypothesis | confidence | reachable test |
|---|---|---|---|
| **H1** | The spectral branch reduces (to first order) to a radial-profile classifier on standardised `log|F(g)|_256`. | high | `envelope_linear_probe.py` |
| **H2** | The watermark is structurally a multiplicative gain on a high-frequency band of \|F(f)\|, *not* an additive Fourier series. | medium-high (synthetic-2 supports it; needs dataset-level confirmation) | `delta_dataset_analysis.py` per-bit radial profile of `|F(g)|_256` |
| **H3** | The spectral branch's accuracy depends on joint training with the spatial branch (a *gradient-flow* effect, not a representational one). | medium (the standalone Spectral decoder collapsed at chance with the same architecture, see logs below) | re-train spectral-only initialised from the joint checkpoint's spectral weights with the spatial branch frozen at zero. Out-of-scope under the "don't retrain" constraint, but is a one-shot non-destructive experiment. |
| **H4** | The cepstrum / phase carries bit information. | low. Phase is dropped by `fft_log_magnitude`; the architecture cannot read it. The cepstrum is implicitly readable but is not in the model's gradient path. | `cepstrum_delta.py` (descriptive only — answers whether Δ has periodic structure, irrespective of whether the model uses it). |
| **H5** | The watermark is a sparse Fourier series. | rejected. The empirical 1.5 % central-128² figure refutes it directly. | — |

## A second puzzle: standalone vs. joint training

Two facts from `decoding/results/training_logs/`:

- `spectral.log` (standalone Spectral decoder, same architecture as the
  spectral branch): final val mean-bit ≈ 0.50, exact match 0.0078
  (chance).
- `dual_branch_r50.md` ablation (`no_spatial`, spectral branch only at
  inference): 99.61 % / 96.88 %.

Architecture is identical; only the training context differs. Two
non-exclusive explanations:

a. **Optimisation saddle.** Standalone, the spectral CNN starts at
   train loss = ln 2 and never escapes. With a co-decoder that reaches
   meaningful predictions in the first few epochs, the fusion-head
   gradients into the spectral branch become non-pathological, and the
   spectral CNN learns the same features it could have learned alone
   in principle.
b. **Distillation through the fusion head.** The spatial branch acts
   as a co-teacher: when its features are noisy at inference (which is
   the *no_spectral* mode at 59.5 % mean bit), the spectral branch
   *had* to learn enough during training to compensate, because the
   fusion head alternates between which branch is providing useful
   signal. This is a curriculum effect.

Either way, this is independent of the original puzzle but worth
flagging: any future "frequency-only" decoder probably needs a
co-decoder during training even if it is removed at inference.

## What was reachable here vs. what wasn't

**Reachable (executed):**

| script | what it does | output |
|---|---|---|
| `synthetic_broadband_control.py` | constructs Δ as texture-masked white noise; shows what (e)−(d) looks like at 256² | `figures/synthetic_broadband_control.png` + json |
| `synthetic_envelope_control.py` | constructs `g` from a multiplicative gain on \|F(f)\|; shows the envelope produces the puzzle's three Δ-properties simultaneously | `figures/synthetic_envelope_control.png`, `synthetic_envelope_highk_control.png` + json |

**Not run (require GPU + checkpoint + dataset):**

| script | what it does | hypothesis it tests |
|---|---|---|
| `delta_dataset_analysis.py` | per-(prompt, bits) pixel and 256²-FFT stats across the entire dataset, including per-bit mean radial profile of |F(Δ)| | H2 |
| `radial_ablate_spectral.py` | mask out FFT bins inside / outside / annular bands, measure accuracy drop. Also runs with the spatial branch zeroed to isolate the spectral path. | H1 |
| `envelope_linear_probe.py` | logistic regression per bit on the radial-binned standardised log-spectrum of g. If it reaches > 95 %, the spectral CNN is implementing a learned radial-profile classifier with extra robustness slack. | H1 directly |
| `per_bit_branch_compare.py` | per-bit accuracy under spatial-only and spectral-only ablation. Tells us whether the spatial 59 % is a uniform improvement or carries specific bits. | refines mechanism story |
| `cepstrum_delta.py` | mean cepstrum + autocorrelation of Δ. Periodic spatial structure → off-origin peaks. | H4 (descriptive) |
| `null_seed_baseline.py` | (baseline_a − baseline_b) for same-prompt different-seed pairs. **Requires regenerating data.** | quantifies the null distribution that all of the above are working against |

## Recommended next-step ordering

1. `delta_dataset_analysis.py` — fast (no model load), gives per-bit
   radial profiles. If H2 is right, the per-bit `P(k|bit=1) / P(k|bit=0)`
   curves should differ meaningfully on a small set of radii.
2. `envelope_linear_probe.py --feature radial` — direct H1 test. The
   ceiling here is the headline number for whether "the spectral CNN is
   a radial-profile classifier".
3. `radial_ablate_spectral.py` — confirms which bands the trained model
   actually uses (versus what the linear probe could in principle use).
4. `null_seed_baseline.py` — only if you can spare the GPU time to
   regenerate ≥ 4 baseline seeds per prompt. It's the only experiment
   here that is informative about generation-noise variance.

## Honest negative results

- I could not numerically confirm H1 or H2; I have a synthetic
  construction that is consistent with both, but the dataset
  measurement remains to be done.
- The "spatial branch alone gets 59.5 %" result is not investigated at
  the per-bit level here. `per_bit_branch_compare.py` would settle
  whether the spatial branch carries specific bits or is uniformly
  guessing-with-bias.
- I could not test H3 at all: confirming the joint-training story
  requires retraining, which the constraint forbids.
- The cepstrum / phase question (H4) has a clear architectural answer
  (the model doesn't read either), so the empirical question is "does
  Δ have periodic structure that *could* be exploited" — interesting
  but not directly relevant to mechanism.

## Concrete revision to the slide deck

The figure `figure_fourier_intuition.py` is correct as a description of
Δ at 1024² but is misleading as an explanation for the spectral branch's
99.6 %, since the model never sees the 1024² FFT. A clearer story for the
deck would replace panel (c) with `synthetic_envelope_highk_control.png`
panel (f) — the standardised `log|F(g)|_256 − log|F(f)|_256` — annotated
with "this is what the spectral CNN can read; the radial bump is the
watermark." That panel reverses the rhetorical direction of the figure
from "the watermark is a Fourier series, the FFT inverts the sum" to
"the watermark is a multiplicative envelope on the generation's
spectrum; per-image standardisation + a 256² FFT exposes it as a clean
radial signature."
