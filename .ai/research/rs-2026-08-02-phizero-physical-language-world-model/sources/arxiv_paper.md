# Source: arXiv Paper — PhiZero

## Source Information

- **Title:** PhiZero: A World Model Built Around Physical Language
- **Authors:** Shuyao Shang, Yuqi Wang, Ruopeng Gao, Xu Chen, Tieniu Tan, Lue Fan, Zhaoxiang Zhang
- **Affiliation:** NLPR, Institute of Automation, Chinese Academy of Sciences (CASIA)
- **arXiv ID:** 2607.28624 (v1)
- **URL:** https://arxiv.org/abs/2607.28624
- **HTML:** https://arxiv.org/html/2607.28624v1
- **PDF:** https://arxiv.org/pdf/2607.28624
- **Project Page:** https://phi-zero.github.io/
- **GitHub:** https://github.com/yaoyao-jpg/PhiZero (code coming soon)
- **DOI:** https://doi.org/10.48550/arXiv.2607.28624

## Summary

PhiZero introduces a physical world model built around "physical language" — a compact discrete representation of world-state transitions learned from in-the-wild videos through self-supervision. The key innovation is a reason-then-render paradigm that separates dynamics inference (in physical-language space) from pixel-level synthesis (video rendering).

The system has two components:
1. **Physical Language Tokenizer:** Encodes video state transitions into 256 discrete tokens using a transition-level Q-Former + FSQ (25K vocabulary) + diffusion-prior decoder (Wan2.2-5B with LoRA). Uses 175× fewer tokens than dense VAE.
2. **Physical Language Reasoner:** Qwen3-VL-4B-initialized autoregressive model that predicts physical-language sequences from first frame + text prompt. Two-stage training (continued pretraining on 5M clips, SFT on 1M motion-rich clips).

Key results: SOTA on Physics-IQ Verified (41.2 IQ-Score), PhyGround, and WorldModelBench. Competitive on IntPhys2, LikePhys, YoCausal. Demonstrates zero-shot cross-embodiment transfer (human → Unitree G1, human hand → dexterous hand), sim-to-real transfer (LIBERO), and interactive controllable world modeling (nuScenes, AGI-Bot RealRobot).

## Key Technical Details

- **FSQ levels:** (8, 5, 5, 5, 5, 5) → 25K vocabulary
- **Physical-language sequence length:** 256 tokens (for 33-frame video at 8 FPS, 4 seconds)
- **Training data:** ~50K hours raw → ~10K hours pretraining → ~5M clips SFT → ~1M clips reasoner SFT
- **Resolution:** 512×896 (SFT stage)
- **Curriculum:** Progressive temporal (1s → 2s → 4s) and spatial (256×448 → 512×896) scaling

## BibTeX

```bibtex
@article{shang2026phizero,
  title   = {PhiZero: A World Model Built Around Physical Language},
  author  = {Shang, Shuyao and Wang, Yuqi and Gao, Ruopeng and Chen, Xu and Tan, Tieniu and Fan, Lue and Zhang, Zhaoxiang},
  year    = {2026},
  note    = {Preprint}
}
```

## Limitations (from Appendix E)

1. Physical language is empirical, not grounded in interpretable physical variables or formal laws
2. Coverage constrained by visual observability — tactile/microscopic dynamics difficult
3. Current scale limited (small models, limited training corpus)

## Future Work (from Appendix E)

1. Physical language as intermediate for VLMs and embodied policies
2. Large-scale human-to-robot state-transition transfer
3. Hierarchical/recurrent prediction for long-horizon world modeling
4. Scaling with stronger backbones, more compute, larger datasets

## References and Links

- [arXiv Abstract](https://arxiv.org/abs/2607.28624)
- [arXiv HTML](https://arxiv.org/html/2607.28624v1)
- [arXiv PDF](https://arxiv.org/pdf/2607.28624)
- [Project Page](https://phi-zero.github.io/)
- [GitHub Repository](https://github.com/yaoyao-jpg/PhiZero)
- [alphaXiv Discussion](https://www.alphaxiv.org/overview/2607.28624)
- [Cool Papers Discovery](https://papers.cool/arxiv/2607.28624)

### Key Related Works Cited

- **Wan2.2** (Wan et al., 2025) — Base video diffusion model for decoder and encoder
- **Qwen3-VL-4B** (Bai et al., 2025a) — Base VLM for Physical Language Reasoner
- **FSQ** (Mentzer et al., 2024) — Finite Scalar Quantization for discretization
- **Q-Former** (Li et al., 2023) — Transition-level query architecture
- **Flow matching** (Liu et al., 2023b) — Diffusion decoder training objective
- **LoRA** (Hu et al., 2022) — Efficient fine-tuning of diffusion decoder
- **nuScenes** (Caesar et al., 2020) — Autonomous driving benchmark
- **AGI-Bot RealRobot** (Bu et al., 2025a) — Robotic manipulation dataset
- **LIBERO** (Liu et al., 2023a) — Sim-to-real transfer demonstration
