# Method Details: PhiZero

## Problem Formulation (Sec. 3.1)

Let **V** denote the future video, I⁰ the first frame (current world state), c the textual action intent, and **z** the discrete physical language describing the transition from I⁰ to **V**.

Future prediction factorizes into:
1. **Physical-language reasoning:** P(**z** | I⁰, c) — infer how the current state should evolve
2. **Video rendering:** P(**V** | I⁰, **z**) — render the inferred transition into video

This reason-then-render decomposition separates state-transition reasoning from pixel-level synthesis.

---

## Physical Language Tokenizer (Sec. 3.2)

### Encoder with Transition-level Q-Former

**Input:** Video V ∈ ℝ^(B×3×T×H×W)

1. **Spatiotemporal encoding:** Wan2.2 VAE encoder → latent features x ∈ ℝ^(B×C×t×h×w)
   - A 33-frame video produces 9 temporal latent states
2. **Transition extraction:** For each adjacent pair (x^i, x^(i+1)), a shared Q-Former with M=32 learnable transition queries Q ∈ ℝ^(32×D_q) extracts transition features q_i ∈ ℝ^(B×32×D_q)
3. **Concatenation:** All adjacent transition features concatenated in temporal order → q ∈ ℝ^(B×N×D_q) where N = (t-1)×M = (9-1)×32 = 256

### Finite Scalar Quantization (FSQ)

- Quantization levels: (8, 5, 5, 5, 5, 5) → vocabulary size = 8×5×5×5×5×5 = **25,000 discrete symbols**
- No separately learned codebook — vocabulary is the Cartesian product of scalar quantization levels
- Quantized representation z projected to diffusion transformer hidden dimension d → physical-language context P_c ∈ ℝ^(B×256×d)

### Diffusion-prior Decoder

- **Base model:** Wan2.2-5B (pretrained video diffusion model)
- **Architecture:** Original model retained; text condition replaced with physical-language context P_c
- **First frame conditioning:** I⁰ provided as clean visual condition → source of static appearance
- **Fine-tuning:** LoRA with rank 32
- **Training objective:** Standard flow-matching (Liu et al., 2023b)
  - Conditioned on: P_c, I⁰, clean latent x⁰, diffusion timestep τ ~ U(0,1)

### Pure-noise Warm-up

- **Problem:** Pretrained diffusion decoder may ignore physical-language context by relying on partially corrupted target information + existing denoising prior
- **Solution:** Initialize all future-frame latents from pure noise during warm-up
- **Effect:** Decoder must rely on physical-language context + first frame to reconstruct
- **After warm-up:** Restore standard flow-matching noise schedule

### Data Curation Pipeline

**Stage 1 — Pretraining:**
- Start with ~50K hours raw footage
- Filter: duplicates, compression artifacts, corrupted frames, watermarks, resolution/duration failures, abrupt shot transitions
- Result: ~10K hours unlabeled videos

**Stage 2 — SFT:**
- Second-pass filtering: aesthetic quality, motion magnitude, VLM-judged state-transition observability
- ~1K hours simulated videos (filtered for rendering quality, valid duration, sufficient object motion)
- Combined: ~5M four-second video clips

### Curriculum Training Recipe

| Phase | Resolution | Duration | Notes |
|-------|-----------|----------|-------|
| Pretraining 1 | 256×448 | 1s | Learn local state changes |
| Pretraining 2 | 256×448 | 2s | Longer-range evolution |
| Pretraining 3 | 256×448 | 4s | Full clip duration |
| SFT | 512×896 | 4s | Curated 5M-clip corpus |
| Final | 512×896 | 4s | Freeze tokenizer, refine decoder only |

---

## Physical Language Reasoner (Sec. 3.3)

### Architecture

- **Base model:** Qwen3-VL-4B (pretrained VLM)
- **Vocabulary extension:** One atomic symbol per FSQ index (25K new tokens)
- **Input:** First frame I⁰ + text prompt c
- **Output:** Autoregressive prediction of length-256 physical-language sequence over 25K-symbol vocabulary
- **Conditional distribution:** Factorized in temporal order

### Training Data Generation

1. Encode each training video with frozen Physical Language Tokenizer: z = T_φ(V)
2. VLM generates caption summarizing only high-level initiating action (not fine-grained transitions)
3. First frame + caption = inputs; tokenizer output = target

### Two-stage Training

| Stage | Data | Purpose |
|-------|------|---------|
| Stage 1: Continued Pretraining | 5M general clips | Adapt VLM to autoregressive physical-language prediction; establish correspondence among text intent, visual state, and state transition |
| Stage 2: SFT | ~1M motion-rich, physically informative clips | Improve physical plausibility and precision while preserving broad capability |

**SFT data filtering:** VLM-based rich-motion filter + physical filter → retain clips with salient state changes and physically informative interactions + curated simulator-generated samples.

---

## Implementation Details (Sec. 4.1)

| Component | Configuration |
|-----------|--------------|
| Spatiotemporal encoder | Wan2.2 VAE encoder (pretrained weights) |
| Diffusion decoder | Wan2.2-5B (LoRA rank 32) |
| FSQ levels | (8, 5, 5, 5, 5, 5) → 25K vocabulary |
| Q-Former queries | 32 learnable transition queries |
| Physical-language sequence length | 256 tokens (for 33-frame video) |
| Reasoner base model | Qwen3-VL-4B |
| Video specs | 4-second clips, 8 FPS, 512×896 resolution |
