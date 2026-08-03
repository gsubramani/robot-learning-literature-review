# PhiZero: A World Model Built Around Physical Language

**Authors:** Shuyao Shang, Yuqi Wang, Ruopeng Gao, Xu Chen, Tieniu Tan, Lue Fan, Zhaoxiang Zhang  
**Affiliation:** NLPR, Institute of Automation, Chinese Academy of Sciences (CASIA)  
**arXiv:** [2607.28624](https://arxiv.org/abs/2607.28624)  
**Project Page:** [https://phi-zero.github.io/](https://phi-zero.github.io/)  
**Code:** [https://github.com/yaoyao-jpg/PhiZero](https://github.com/yaoyao-jpg/PhiZero) (coming soon)  

---

## TL;DR

PhiZero introduces **physical language** — a compact, discrete representation of world-state transitions learned self-supervised from in-the-wild videos. Instead of predicting future videos directly in pixel space (where physical dynamics remain implicit), PhiZero follows a **reason-then-render** paradigm: it first infers future world evolution as a physical-language sequence, then renders the inferred transitions into video. The system comprises a **Physical Language Tokenizer** (encodes video transitions into discrete tokens via a transition-level Q-Former + FSQ + diffusion-prior decoder) and a **Physical Language Reasoner** (a VLM-initialized autoregressive model that predicts physical-language sequences from the current frame and action intent). PhiZero achieves state-of-the-art on physical video generation and understanding benchmarks, and demonstrates zero-shot cross-embodiment motion transfer, sim-to-real transfer, and interactive controllable world modeling — all enabled by the appearance-disentangled nature of physical language.

---

## Key Contributions

1. **Physical language:** A compact discrete representation of state-transition patterns learned at scale from unlabeled in-the-wild videos through self-supervised learning. Uses only 256 discrete symbols to represent a 4-second video's state transitions — **175× fewer tokens** than a dense VAE representation (44,800 tokens).
2. **PhiZero world model:** A reason-then-render paradigm that separates dynamics inference (physical-language reasoning) from pixel-level synthesis (video rendering), making world evolution an explicit reasoning target rather than an implicit pixel-space prediction.
3. **Broad validation and applications:** State-of-the-art on three generation benchmarks (Physics-IQ Verified, PhyGround, WorldModelBench) and competitive on three understanding benchmarks (IntPhys2, LikePhys, YoCausal). Demonstrates physically realistic video generation, fine-grained action-conditioned simulation, interactive rollouts, and zero-shot motion transfer across embodiments and visual domains.

---

## Method Overview

PhiZero comprises two complementary components:

### Physical Language Tokenizer (Sec. 3.2)

Learns to compress video state transitions into a discrete physical-language sequence through self-supervised video reconstruction.

- **Encoder with Transition-level Q-Former:** A spatiotemporal encoder (Wan2.2 VAE) extracts latent features. For each pair of adjacent latent states, a shared Q-Former (32 learnable queries) extracts transition representations. This local temporal inductive bias reduces compression complexity while preserving temporal ordering.
- **Finite Scalar Quantization (FSQ):** Transition features are discretized using FSQ with levels (8,5,5,5,5,5), yielding a **25K-symbol vocabulary**. No separately learned codebook needed.
- **Diffusion-prior Decoder:** A pretrained video diffusion model (Wan2.2-5B) serves as decoder, replacing text conditioning with physical-language context. The first frame is provided as a clean visual condition (anchors static appearance), so the discrete bottleneck focuses on state changes. Fine-tuned with LoRA (rank 32).
- **Pure-noise Warm-up:** Future-frame latents initialized from pure noise during warm-up to prevent the decoder from ignoring the physical-language context. Forces reliance on physical language + first frame.
- **Token count:** A 33-frame video → 9 temporal latent states → (9−1)×32 = **256 physical-language tokens**.

### Physical Language Reasoner (Sec. 3.3)

Predicts future state transitions from the current visual state and textual action intent.

- **Initialized from Qwen3-VL-4B** (pretrained VLM), leveraging visual-semantic knowledge and commonsense priors.
- **Vocabulary extension:** One atomic symbol per FSQ index added to the VLM's vocabulary.
- **Autoregressive prediction:** Given first frame I⁰ and text prompt c, predicts a length-256 physical-language sequence.
- **Training data:** Supervision generated offline by encoding training videos with the frozen tokenizer. VLM-generated captions describe only high-level initiating actions (not fine-grained transitions) to prevent the text from revealing outcomes.
- **Two-stage training:** Stage 1 = continued pretraining on 5M general clips; Stage 2 = SFT on ~1M motion-rich, physically informative clips.

### Data Pipeline

- **Pretraining:** ~50K hours raw video → filtered to ~10K hours unlabeled videos.
- **SFT:** Stricter filtering (aesthetic quality, motion magnitude, VLM-judged state-transition observability) + ~1K hours simulated videos → ~5M four-second clips.
- **Curriculum:** Progressive temporal (1s → 2s → 4s) and spatial (256×448 → 512×896) resolution increase.

See [details/method.md](details/method.md) for full technical details.

---

## Experimental Results

### Video Generation (3 benchmarks)

| Benchmark | Metric | PhiZero Result |
|-----------|--------|----------------|
| **Physics-IQ Verified** | IQ-Score | **Best** (41.2) |
| **PhyGround** | Physics Score, Overall Score | **Highest** |
| **WorldModelBench** | Physics Adherence, Total Score | **Best** |

PhiZero generates state transitions that closely match real-world physical outcomes across diverse open-domain phenomena (collisions, deformation, chain reactions, shadow changes).

### Video Understanding (3 benchmarks)

| Benchmark | Task | Result |
|-----------|------|--------|
| **IntPhys2** | Intuitive physics | Competitive |
| **LikePhys** | Physical-plausibility discrimination | Competitive |
| **YoCausal** | Temporal/causal understanding | Competitive |

Evaluation protocol: encode both valid/invalid videos into physical-language sequences, compute log-likelihoods under the Reasoner, select higher-likelihood video as valid.

### Reconstruction Performance

| Method | Token Count | Quality |
|--------|-------------|---------|
| Wan2.2 VAE | 44,800 continuous | High |
| **PhiZero Tokenizer** | **256 discrete** | **Best among highly compressed** |

### Key Ablation Findings

- **Diffusion decoder is critical:** Replacing with deterministic decoder causes largest degradation — diffusion prior recovers fine-grained appearance, letting bottleneck focus on transitions.
- **Transition-level Q-Former > global Q-Former:** Local temporal inductive bias matters.
- **Pure-noise warm-up helps:** Prevents decoder shortcut of relying on corrupted target info.
- **Simulation data improves physical plausibility:** Removing sim data reduces performance on Physics-IQ Verified.
- **Two-stage training > joint training:** Dedicated SFT on motion-rich clips improves precision.

See [details/experiments.md](details/experiments.md) for full experimental details.

---

## Broader Applications

### Physically Realistic Video World Model
Models diverse real-world dynamics: ocean waves, liquid pouring, objects in hot oil, metal can explosions. Captures complete temporal evolution and fine-grained consequences.

### Controllable and Interactive World Model
- **Autonomous driving (nuScenes):** Captures fine-grained steering magnitude variations.
- **Robotic manipulation (AGI-Bot RealRobot):** Accurately follows action signals for gripper movements.
- **Interactive rollouts:** Updates camera viewpoint/position under sequential control inputs while maintaining temporal consistency.

### Zero-Shot Cross-Embodiment and Sim-to-Real Transfer
- **Human → Unitree G1 humanoid:** Full-body motion transfer via physical language.
- **Human hand → Sharpa dexterous hand:** Hand motion transfer without target-specific training.
- **Sim-to-real (LIBERO):** Transform first frames to realistic visual domain, render original simulated state transitions under new appearance.

See [details/applications.md](details/applications.md) for application details.

---

## Analysis of Physical Language

### Transferability
Physical language disentangles state transitions from visual appearance. Encoding a source video's transition, editing the first frame to a different appearance, and decoding the unchanged physical-language sequence preserves transition patterns (pouring, viscous spreading, liquid flow) across appearance changes.

### Semantic Structure
- **Autonomous driving:** Stationary clips form distinct cluster; steering patterns arranged along continuous manifold.
- **Robotic manipulation:** Four gripper transition patterns form compact, largely separated clusters.
- Physical language organizes videos by **how the world changes**, not by visual content.

---

## Limitations

1. **Empirical, not symbolic:** Physical language is learned as an empirical representation, not grounded in interpretable physical variables or formal laws.
2. **Observational coverage:** Coverage constrained by what can be visually observed — tactile interactions and microscopic dynamics are difficult to model.
3. **Scale:** Current implementation uses relatively small-scale models and limited training corpus relative to physical world diversity.

---

## Relevance to Robot Learning

PhiZero is highly relevant to robot learning literature:

- **World models for robot learning:** Provides a principled world model that separates dynamics reasoning from visual rendering — directly applicable to model-based RL and planning.
- **Cross-embodiment transfer:** Zero-shot motion transfer from human to robot embodiments via appearance-disentangled physical language could address the scarcity of real-robot interaction data.
- **Sim-to-real:** Physical language enables transferring simulated state transitions to realistic visual domains, potentially bridging the sim-to-real gap for demonstration generation.
- **Action-conditioned simulation:** Fine-grained action-conditioned world modeling supports interactive rollouts for autonomous driving and robotic manipulation.
- **Learned representations:** The physical language representation could serve as an intermediate for improving spatiotemporal understanding in VLMs and supporting planning in embodied policies.

---

## Future Directions

- Physical language as explicit intermediate representation for VLMs and embodied policies
- Transferring human video state-transition patterns to robotic embodiments at scale
- Hierarchical/recurrent prediction for long-horizon world modeling
- Scaling with stronger backbones, more compute, and larger/more diverse training data

---

## Sources

- [arXiv Paper](https://arxiv.org/abs/2607.28624)
- [HTML Version](https://arxiv.org/html/2607.28624v1)
- [Project Page](https://phi-zero.github.io/)
- [GitHub Repository](https://github.com/yaoyao-jpg/PhiZero) (code coming soon)
- [alphaXiv Discussion](https://www.alphaxiv.org/overview/2607.28624)

See [sources/arxiv_paper.md](sources/arxiv_paper.md) for detailed source summary.
