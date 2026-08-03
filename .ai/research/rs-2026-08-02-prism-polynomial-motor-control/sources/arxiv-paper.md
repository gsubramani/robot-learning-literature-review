# Source: arXiv Paper — PRISM

## Bibliographic Information

- **Title:** PRISM: Polynomial Representations for Interaction-Structured Motor Control
- **Authors:** Seung Hyun Lee, Stella X. Yu
- **Affiliation:** University of Michigan, Ann Arbor
- **arXiv ID:** 2607.23473 (v1)
- **Submission Category:** cs.RO
- **Submission Date:** July 2026
- **License:** CC-BY 4.0
- **DOI:** 10.48550/arXiv.2607.23473

## Links

- [arXiv abstract](https://arxiv.org/abs/2607.23473)
- [PDF](https://arxiv.org/pdf/2607.23473)
- [HTML (full text)](https://arxiv.org/html/2607.23473v1)
- [TeX Source](https://arxiv.org/src/2607.23473)
- [Project Page](https://lsh3163.github.io/prism/)
- [GitHub Code](https://github.com/lsh3163/prism)
- [NASA ADS](https://ui.adsabs.harvard.edu/abs/arXiv:2607.23473)
- [Google Scholar](https://scholar.google.com/scholar_lookup?arxiv_id=2607.23473)
- [Semantic Scholar](https://api.semanticscholar.org/arXiv:2607.23473)

## Abstract

Robot policies are typically MLPs mapping observations to actions. Yet robot observations are physical variables, and many action-relevant cues arise not from individual variables but from their interactions; power, inertial effects, contact, slip, and compliance depend on products among observable signals. We introduce PRISM, a policy representation that makes polynomial interactions among observable physical variables explicit, learnable, and compact. Rather than listing all polynomial terms, PRISM uses a factorized polynomial module to expose higher-order interaction features efficiently. In reinforcement learning, it keeps the standard MLP backbone but applies a gradually activated element-wise polynomial function after it. In imitation learning, it replaces linear proprioceptive conditioning in Diffusion Policy with a polynomial layer trained end-to-end. Across humanoid locomotion and contact-rich manipulation, PRISM improves performance over standard MLP policies and larger MLPs with matched capacity, showing that interaction structure cannot be replaced by capacity alone. It also yields sensorless compliant behavior without force, wrench, tactile input, contact labels, or admittance control. These results suggest that polynomial representations should become a standard architectural choice for embodied motor control.

**Keywords:** Polynomial interaction, Motor control, Sensorless compliance

## BibTeX

```bibtex
@article{lee2026prism,
  title = {PRISM: Polynomial Representations for Interaction-Structured Motor Control},
  author = {Lee, Seung Hyun and Yu, Stella X.},
  journal = {arXiv preprint arXiv:2607.23473},
  year = {2026},
  doi = {10.48550/arXiv.2607.23473}
}
```

## Paper Structure

1. **Introduction** — Motivation: first-order variables vs. multiplicative interactions; three contributions
2. **Related Work** — Polynomial/multiplicative neural networks; proprioceptive robot control
3. **Method** — Deployable policy setting; interaction-structured representation; RL integration; imitation learning integration
4. **Experimental Results** — Setups; baseline comparisons; latent physical interaction analysis
5. **Limitations** — Low-degree assumption; missing sensory coverage; fixed controllers
6. **Conclusion** — Polynomial representations as standard architectural choice
7. **Appendix A** — Experimental details (Humanoid-Gym, LIBERO, stronger backbones, compliance baselines)
8. **Appendix B** — Additional results (LIBERO breakdown, dynamics-conditioned representation, perturbations, physical-response probes, SO-101 real-robot pilot)
9. **Appendix C** — Ablations (polynomial degree)

## Additional Sources

- [The Neural Feed summary](https://theneuralfeed.com/article/prism-polynomial-representations-for-interaction-structured-motor-control/c6y8Pudl) — Third-party summary confirming key findings
- [GitHub README](https://github.com/lsh3163/prism) — Implementation details, integration patches for BFM-Zero and SmolVLA, training commands
