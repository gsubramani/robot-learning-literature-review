# PRISM: Polynomial Representations for Interaction-Structured Motor Control

**Authors:** Seung Hyun Lee, Stella X. Yu  
**Affiliation:** University of Michigan, Ann Arbor  
**arXiv:** [2607.23473](https://arxiv.org/abs/2607.23473)  
**Project Page:** [https://lsh3163.github.io/prism/](https://lsh3163.github.io/prism/)  
**Code:** [https://github.com/lsh3163/prism](https://github.com/lsh3163/prism)  
**License:** CC-BY 4.0  

---

## TL;DR

Standard robot policy MLPs receive only first-order physical variables (positions, velocities, commands), but many action-relevant cues — power, inertial effects, contact, slip, compliance — arise from **multiplicative interactions** among those variables. PRISM introduces a **factorized polynomial module** that makes these interactions explicit, learnable, and compact, without enumerating all monomials. It drops into existing RL and imitation-learning pipelines with minimal change, improves performance over standard and larger-capacity MLP baselines, and yields **sensorless compliant behavior** without force/tactile sensors or admittance control.

---

## Key Contributions

1. **Identifies a limitation of default MLP policies:** They receive first-order physical variables, while many action-relevant cues arise from multiplicative interactions (e.g., torque × velocity = power, velocity × velocity = Coriolis/centrifugal effects).
2. **Introduces PRISM:** A compact polynomial policy representation using factorized element-wise polynomial interactions. Can be gradually activated in RL (after MLP backbone) or replace linear proprioceptive conditioning in Diffusion Policy / SmolVLA.
3. **Demonstrates broad gains:** Improves humanoid locomotion, contact-rich manipulation, and physical-quantity probing — without deployment-time sensing overhead. Interaction structure cannot be replaced by capacity alone.

---

## Method Overview

PRISM modifies only the **proprioceptive conditioning pathway** while preserving the rest of the policy architecture (visual encoder, action interface, training objective, low-level controller).

### Core Representation

Given deployment-available proprioceptive input `x_t`, PRISM computes two learned affine factors, then forms a second-order representation:

```
ψ(x_t) = W₁x_t + α₂ ⊙ (W₁x_t ⊙ W₂x_t)
```

- First term: standard first-order (linear) path
- Second term: factorized quadratic interactions via element-wise product
- `α₂` is learned per latent feature, initialized near zero (starts as standard linear projection, learns interactions end-to-end)
- Extends recursively to degree K without enumerating all monomials

### Two Integration Modes

| Mode | How PRISM is Applied | What Stays Unchanged |
|------|---------------------|---------------------|
| **RL (PPO)** | Gradually activated element-wise polynomial function after MLP backbone | Actor core, action space, reward, low-level controller |
| **Imitation Learning (Diffusion Policy / SmolVLA)** | Replaces linear proprioceptive conditioning layer with polynomial layer | Visual path, diffusion process, VLM backbone, action expert |

See [details/method.md](details/method.md) for full mathematical formulation.

---

## Experimental Results

### Humanoid Locomotion (Humanoid-Gym)

- PRISM achieves best episode length, linear-velocity tracking error, and survival rate
- **Larger MLP fails to bridge the gap** — gains come from interaction structure, not parameter count
- Degree 3 performs best; Degree 2 used as default (captures most gain with simpler architecture)
- PRISM nearly doubles survival rate; wider MLP stays at baseline

### Contact-Rich Manipulation (LIBERO)

- PRISM achieves highest task success rate across Spatial, Long, Object, Goal suites
- Reaches **91% success without force as a policy input**
- Yields **sensorless compliant behavior** — maintains low, stable contact-force profile without force/wrench/tactile input
- Outperforms Minimalist Compliance Control (MCC) in the deployable sensorless setting
- Largest gains on long-horizon tasks

### Stronger Backbone Validation

- **BFM-Zero** (locomotion): PRISM reduces tracking EMD under nominal, low-friction, and payload-mass perturbations
- **SmolVLA** (VLA manipulation): PRISM improves average LIBERO success across all suites
- In both cases, PRISM outperforms larger-capacity conditioner controls while using fewer parameters

### Latent Physical Interaction Analysis

- Linear probes on frozen representations show PRISM makes slip, joint power, contact impulse, and contact work **significantly more linearly recoverable** than baselines
- Ablation of individual degree-2 latent factors reveals highest-impact factors involve velocity memory, cross-joint velocity, and state–velocity interactions
- Factor names are post-hoc interpretations, not predefined variables

See [details/experiments.md](details/experiments.md) for detailed results.

---

## Key Insights

- **Interaction structure ≠ capacity:** Simply scaling MLP width does not substitute for explicit multiplicative interactions. The input basis matters.
- **Sensorless compliance:** Polynomial proprioceptive conditioning functions as an implicit force-free compliance mechanism — the policy contextualizes environment boundaries without force sensors.
- **Physical probes emerge naturally:** Learned polynomial features encode mechanics-inspired quantities (slip, power, contact impulse, work) in a linearly accessible way, without being trained to do so.
- **Backbone-agnostic:** PRISM works across PPO-based RL, Diffusion Policy, and VLA (SmolVLA) architectures.

---

## Simulation vs. Real Robot

**Main results are entirely in simulation:**
- Humanoid-Gym (locomotion) and LIBERO (manipulation) are both simulation benchmarks
- All quantitative comparisons, ablations, and probing experiments use simulated environments

**One real-robot pilot (Appendix B.5):**
- SO-101 arm performing a mouse-click task
- Trained from 5 leader-arm demonstrations; 10 trials per method
- Diffusion Policy + PRISM completes the click more often than baseline
- Same sensing/control interface (no force/tactile/current input)
- Authors explicitly frame this as "a hardware pilot rather than a definitive real-world benchmark"

---

## Limitations

- Assumes key cues are captured by **low-degree** polynomial interactions among deployment-available observations
- Less effective when failures depend on long-horizon history, unobserved contact geometry, material properties, or higher-order dynamics
- Cannot compensate for missing sensory coverage
- Experiments limited to humanoid locomotion and contact-rich manipulation with fixed controllers
- **Real-world validation is limited to a small SO-101 pilot** — no comprehensive real-robot benchmarks
- Future work: adaptive polynomial degree, temporal interactions, broader morphologies, diverse real-world contacts

---

## Relevance to Robot Learning Literature

PRISM addresses a fundamental architectural question in robot learning: **what inductive bias should policy networks encode?** While most work focuses on scaling data, models, or sensing, PRISM shows that a simple change to the **input representation basis** — exposing multiplicative interactions — yields consistent gains across domains. This connects to:

- **Physics-informed robot learning:** Unlike approaches that add auxiliary physical estimators or loss terms, PRISM embeds physical interaction structure directly in the representation without explicit physical supervision
- **Multiplicative neural networks:** Builds on polynomial neural networks (Pi-Net, etc.) and multiplicative interactions literature, applying these ideas to motor control
- **Sensorless control:** Challenges the assumption that compliance requires force/tactile sensing, showing that proprioception alone suffices when interaction structure is explicit

---

## Sources

- [arXiv abstract page](https://arxiv.org/abs/2607.23473)
- [arXiv HTML (full text)](https://arxiv.org/html/2607.23473v1)
- [Project page](https://lsh3163.github.io/prism/)
- [GitHub code](https://github.com/lsh3163/prism)
- [The Neural Feed summary](https://theneuralfeed.com/article/prism-polynomial-representations-for-interaction-structured-motor-control/c6y8Pudl)

See [sources/arxiv-paper.md](sources/arxiv-paper.md) for full source summary.
