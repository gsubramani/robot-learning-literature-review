# PRISM Experimental Results Details

## Experimental Setup

### Two Evaluation Domains

1. **Humanoid-Gym** — sensor-minimal humanoid locomotion; actor maintains balance and tracks commanded velocities using only proprioceptive and command observations
2. **LIBERO** — contact-rich imitation-learning manipulation tasks with RGB-based visuomotor policies

### Baselines

**Locomotion:**
- Standard MLP actor
- Larger-capacity MLP actor (tests whether gains come from parameter count)
- Polynomial actor

**Manipulation:**
- Diffusion Policy (base visuomotor policy)
- Minimalist Compliance Control (MCC) paired with Diffusion Policy
  - MCC-Sensorless: delayed/noisy contact-force proxy, deployable
  - MCC-Oracle: simulator contact force, non-deployable upper bound

All methods use same demonstrations, RGB observations, proprioceptive state, relative EEF action space, and low-level controller.

### Metrics

| Domain | Metrics |
|--------|---------|
| Locomotion | Episode length, linear-velocity tracking error, yaw-velocity tracking error, survival rate |
| LIBERO | Success rate, smoothness, position error, orientation error |

Simulation results averaged over 5 random seeds.

---

## Results: Humanoid Locomotion (Humanoid-Gym)

- PRISM achieves **best episode length, linear-velocity tracking error, and survival rate**
- **Larger MLP fails to bridge the gap** with baseline — gains come from interaction structure, not capacity
- Degree 3 performs best; Degree 2 used as default (captures most gain, simpler architecture)
- PRISM nearly **doubles survival rate**; wider MLP stays at baseline
- Visual validation: PRISM tracks target velocities with minimal oscillation; MLP alternatives suffer severe compounding velocity drift and early falls
- PRISM features exhibit tighter convergence bounds across training seeds

---

## Results: Force-Free Polynomial Conditioning in Manipulation (LIBERO)

- PRISM achieves **highest task success rate** across Spatial, Long, Object, Goal suites
- Reaches **91% success without force as a policy input**
- **Sensorless compliant behavior:**
  - MCC tracking displays destabilizing force spikes upon initial contact
  - PRISM maintains low, stable contact-force profile throughout rollout
  - PRISM approaches quickly before contact, then reduces EEF speed after contact
  - Functions as implicit force-free compliance mechanism
- Outperforms MCC-Sensorless in deployable setting
- Improves successful-rollout position and orientation errors
- Largest gains on **long-horizon tasks**

---

## Results: Stronger Policy Backbones

### BFM-Zero (Humanoid Locomotion)

- PRISM reduces tracking EMD under:
  - Nominal dynamics
  - Low friction perturbations
  - Payload-mass perturbations
- Outperforms larger-capacity conditioner control with fewer parameters

### SmolVLA (LIBERO Multi-Task Manipulation)

- One multi-task policy trained jointly on Spatial, Object, Goal, and Long
- PRISM improves average LIBERO success across all suites
- Outperforms larger-capacity conditioner control with fewer parameters

**Training command (SmolVLA):**
```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.state_conditioner_type=prism \
  --policy.state_conditioner_num_layers=2 \
  --policy.state_conditioner_product_mode=gated_quadratic \
  --policy.state_conditioner_gate_scale_init=1e-2 \
  --policy.state_conditioner_use_rmsnorm=true \
  --policy.scheduler_warmup_steps=100 \
  --policy.scheduler_decay_steps=100000 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --batch_size=64 \
  --num_workers=8 \
  --seed=1000 \
  --steps=100000
```

---

## Results: Latent Physical Interaction Analysis

### Linear Probing

- Trained linear probes on **frozen** policy representations to predict future mechanics-inspired responses
- PRISM makes physical proxies **significantly more linearly recoverable** than baselines:
  - Locomotion: slip velocity (PCC), joint power (MSE)
  - Manipulation: contact impulse (MSE), contact work (MSE)
- Despite using strictly sensor-minimal inputs

### Factor Ablation

- Ablate one degree-2 latent factor at a time (set to zero, keep everything else)
- Measure mean absolute change in predicted action
- Name factors by inspecting dominant input-space product from largest weights in affine branches
- **Highest-impact locomotion factors:**
  - Velocity memory
  - Cross-joint velocity
  - State–velocity interactions
- Ablating these factors directly shifts joint-position commands
- Factor names are **post-hoc interpretations**, not predefined variables

### Dynamics Shifts

- Dynamics shifts (e.g., friction changes) separate in PRISM's learned state space
- Qualitative feature visualization confirms structured representation

---

## Appendix Details

### A.1 Humanoid-Gym Locomotion
- Standard PPO training setup
- Same observation interface, action space, reward function across all methods

### A.2 LIBERO Diffusion Policy
- End-to-end training of base policy + PRISM conditioning
- Same RGB, proprioceptive, action, and low-level-control interface

### A.3 Validation on Stronger Backbones
- BFM-Zero: actor and simulator interface unchanged
- SmolVLA: only proprioceptive projection replaced

### A.4 Compliance Baselines and Diagnostics
- MCC-Sensorless applies noisy sensorless wrench correction at execution time
- MCC-Oracle uses simulator contact force (non-deployable upper bound)

### B.1-B.5 Additional Results
- LIBERO suite breakdown
- Dynamics-conditioned representation
- Evaluation-time perturbations
- Physical-response probes
- SO-101 real-robot pilot

### C.1 Ablations
- Polynomial degree ablation (Degree 2 vs 3 vs higher)
