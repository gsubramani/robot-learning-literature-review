# PRISM Method Details

## Problem Formulation

Robot policies π_θ(a_t | o_t) map observations to actions. The observation is partitioned:

- **x_t ∈ ℝ^d**: deployment-available proprioceptive state/history (polynomial conditioning applied here)
- **c_t**: remaining inputs (commands, images, language, unmodified state)

The policy does **not** receive latent physical quantities (contact forces, terrain friction, object mass, external perturbations) at deployment. PRISM does not attempt to identify these directly — it provides interaction-sensitive features that encode their **observable effects**.

---

## Interaction-Structured Proprioceptive Representation

### Second-Order (Default)

Given input x_t, compute two learned affine factors:

```
f₁ = W₁x_t + b₁
f₂ = W₂x_t + b₂
```

The representation is:

```
ψ(x_t) = f₁ + α₂ ⊙ (f₁ ⊙ f₂)
```

Where:
- `⊙` = element-wise multiplication
- `α₂` = learned per-latent-feature scale, initialized near zero
- First term preserves direct first-order (linear) path
- Second term introduces factorized quadratic interactions
- Policy can retain linear features where sufficient, strengthen quadratic interactions where they improve control

### General Degree-K Extension

PRISM extends recursively to polynomial degree K:

```
ψ_K(x_t) = f₁ + α₂⊙(f₁⊙f₂) + α₃⊙(f₁⊙f₂⊙f₃) + ... + α_K⊙(f₁⊙...⊙f_K)
```

Each additional factor increases max polynomial degree by one while preserving all lower-order pathways. No explicit monomial basis expansion needed.

### Output Projection

The representation is mapped to conditioning dimension:

```
z_t = g_η(ψ_K(x_t))
```

Where g_η is a learned projection (MLP).

---

## Integration: Reinforcement Learning (PPO)

1. PRISM encodes proprioceptive state/history into z_t
2. Actor combines z_t with remaining policy inputs c_t
3. Actor predicts continuous action distribution (Gaussian mean)
4. Trained with standard PPO objective
5. **Unchanged:** action space, reward function, policy objective, low-level controller
6. Privileged simulator info (if available) restricted to critic only — unavailable to deployed actor

Key design: **gradual activation** — α₂ initialized near zero, so representation starts as standard linear projection and learns interaction contribution end-to-end.

---

## Integration: Imitation Learning

### Diffusion Policy

- PRISM replaces standard linear proprioceptive conditioning with polynomial layer
- z_t^proprio combined with visual representation, fed to diffusion action model
- Diffusion noise-prediction objective and action-generation process unchanged

### SmolVLA (VLA)

- PRISM replaces standard proprioceptive `state_proj` branch
- Visual-language backbone, visual encoder, and action-expert interface remain intact
- Only the proprioceptive projection is swapped

### Training

- PRISM parameters optimized using backbone's native imitation-learning objective
- No auxiliary physical supervision, interaction labels, additional deployment sensing, or action controller modification

---

## Implementation (from GitHub)

```python
conditioner = PRISMConditioner(
    input_dim=32,
    output_dim=1152,
    hidden_dim=...,
    degree=2,
    interaction_mode=...,
    gate_init=1e-2,
    post_layers=2,
    use_rmsnorm=True,
)
```

Use `conditioner.polynomial_features(proprioception)` to inspect learned polynomial features.

### Supported Backbones

| Backbone | PRISM Changes | Kept Unchanged |
|----------|--------------|----------------|
| BFM-Zero | Deployable `history_actor` representation | Actor core, actions, simulator, objective |
| LeRobot / SmolVLA | Proprioceptive `state_proj` branch | VLM, visual path, action-expert interface |

---

## Why Not Just Feature Lifting?

A naive approach would append all coordinate products to the observation. Problems:
- Scales as O(d^K) for degree K in d dimensions — quickly infeasible
- No learning — all monomials included regardless of relevance

PRISM's factorized approach:
- Learned projections form compact latent factors first
- Element-wise products in latent space — O(d × hidden_dim × K) cost
- Each interaction direction is learned from data, not pre-specified
- Standard MLP actor is the first-order special case (K=1)
