# Experimental Details: PhiZero

## Video Generation Benchmarks

### Physics-IQ Verified (Rädsch et al., 2026)
- **What it measures:** Physical outcome fidelity by comparing generated videos with real reference videos using rule-based metrics
- **PhiZero result:** Best IQ-Score (41.2)
- **Key qualitative finding:** Wan2.2-5B baseline produces visually plausible interactions but fails to generate expected physical consequences (e.g., tennis ball strikes rubber duck but duck remains unaffected). PhiZero generates coherent downstream effects: collision-induced displacement, deformation under gravity, chain reactions, changing shadows.

### PhyGround (Lin et al., 2026)
- **What it measures:** Physical-law adherence with physics-specialized VLM judges
- **PhiZero result:** Highest Physics Score and Overall Score

### WorldModelBench (Li et al., 2026a)
- **What it measures:** General world-modeling capability (physics adherence + overall video quality)
- **PhiZero result:** Best Physics Adherence and Total scores

---

## Video Understanding Benchmarks

All three use pairwise comparison: distinguish physically/causally valid video from matched invalid counterpart.

### Evaluation Protocol
1. Encode both videos into physical-language sequences
2. Compute log-likelihoods under Physical Language Reasoner
3. Select video with higher likelihood as valid

### IntPhys2 (Bordes et al., 2025)
- **Task:** Intuitive physics understanding
- **Result:** Competitive

### LikePhys (Yuan et al., 2025)
- **Task:** Physical-plausibility discrimination in simulated scenes
- **Result:** Competitive

### YoCausal (Xie et al., 2026)
- **Task:** Temporal and causal understanding in real-world videos
- **Result:** Competitive

---

## Reconstruction Performance (Table 7)

| Method | Token Count | Reconstruction Quality |
|--------|-------------|----------------------|
| Wan2.2 VAE | 44,800 continuous | High |
| PhiZero Tokenizer | 256 discrete | Best among highly compressed |

- Evaluated on 500 four-second real-world videos at 8 FPS, 512×896 resolution
- Token count excludes first-frame condition
- **175× compression** vs. dense VAE while retaining sufficient information for high-quality reconstruction

---

## Ablation Studies

### Physical Language Tokenizer Design (Table 8)

| Ablation | Effect |
|----------|--------|
| Replace diffusion decoder with deterministic decoder | **Largest degradation** — diffusion prior recovers fine-grained appearance, allows compact bottleneck to focus on transitions |
| Replace transition-level Q-Former with global Q-Former | Degrades reconstruction — validates local temporal inductive bias |
| Remove pure-noise warm-up | Consistently reduces performance — warm-up prevents decoder shortcut |

### Physical Language Reasoner Design (Table 9, on Physics-IQ Verified)

| Ablation | Effect |
|----------|--------|
| Prompt enhancement only (no physical language) | Improves Wan2.2-5B baseline but remains substantially below PhiZero — natural-language reasoning alone insufficient |
| Remove simulation data | Reduces performance — sim-generated interactions improve physical plausibility |
| Replace two-stage training with joint training | Degrades performance — dedicated SFT on motion-rich clips improves precision |

---

## Analysis of Physical Language

### Transferability (Fig. 5)
- Encode source video's state transition → edit first frame to different appearance → decode unchanged physical-language sequence
- Transferred videos preserve transition patterns (pouring, viscous spreading, liquid flow) across appearance/background changes
- Demonstrates physical language captures **reusable transition information** transferable across visual appearances

### Semantic Structure (Fig. 6)
- **Autonomous driving (nuScenes):** 4 categories (left turn, right turn, straight, stationary), 300 samples each
  - Stationary clips form distinct cluster
  - Steering patterns arranged along continuous manifold
- **Robotic manipulation (AGI-Bot RealRobot):** 4 gripper patterns (downward+closing, upward+closing, downward+opening, sweeping), 300 samples each
  - Four patterns form compact, largely separated clusters
- **Method:** Aggregate transition features → PCA (20D) → UMAP (3D)
- Physical language organizes videos by **how the world changes**, not by visual content
