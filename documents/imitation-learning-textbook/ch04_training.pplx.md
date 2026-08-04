# Chapter 4: How Models Are Trained

Imitation learning is ultimately an optimization problem: given a dataset of expert demonstrations, minimize the discrepancy between what the policy does and what the expert would have done. This chapter covers the full training lifecycle — from the choice of training paradigm and loss function, through algorithm-specific loops for ACT and Diffusion Policy, to fine-tuning large vision-language-action models and the infrastructure needed to run it all reliably.

---

## 4.1 Training Paradigms Overview

Before writing a single line of training code, you need to decide *how* your policy will acquire knowledge. Four broad paradigms are in active use.

### 4.1.1 From-Scratch Training

The simplest approach: initialize a randomly weighted model and train it entirely on your robot demonstration dataset. ACT ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)) and early behavioral-cloning systems fall here. The policy architecture is typically compact (tens of millions of parameters), so it fits on a single consumer GPU and converges in hours.

**When it works:** Task-specific, narrow domains with a few hundred to a few thousand demonstrations and no need to generalize to novel language instructions or object categories.

**When it breaks:** The model has seen no world knowledge; generalization is limited to the distribution of your own demonstrations.

### 4.1.2 Pre-Train Then Fine-Tune

Train a large backbone on broad data (internet-scale vision-language corpora, or large multi-task robot datasets), then specialize it to a target task or robot morphology. OpenVLA ([Kim et al., 2024](https://arxiv.org/abs/2406.09246)) and Octo follow this recipe, starting from pretrained VLMs or vision transformers and fine-tuning on task demonstrations. π₀ ([Black et al., 2024](https://arxiv.org/abs/2410.24164)) extends this further by first pretraining a flow-matching action head on a large robot corpus, then fine-tuning on target tasks.

**Key benefit:** The pretrained backbone encodes rich visual and semantic representations that require far fewer robot demonstrations to specialize.

**Key cost:** Fine-tuning large models requires careful regularization (LoRA, frozen layers) to avoid destroying the pretrained representations.

### 4.1.3 Co-Training

Instead of a strict pretrain → fine-tune sequence, mix robot demonstration data *with* broad non-robot data throughout training. RT-2 ([Brohan et al., 2023](https://arxiv.org/abs/2307.15818)) co-trains a VLM on both web-scale vision-language data and robot action data simultaneously, so the model never forgets its language grounding. π₀.5 adopts a similar approach with heterogeneous robot data across multiple morphologies.

**Key benefit:** Prevents catastrophic forgetting of web knowledge; the model retains strong semantic generalization even as it learns motor skills.

**Key cost:** Requires carefully tuned mixing ratios; naïve mixing can cause one data source to dominate.

### 4.1.4 Continual / Lifelong Learning

The model is updated incrementally as new demonstrations arrive, without full retraining. This is the least mature paradigm in robot imitation learning and is an active research area. Challenges include catastrophic forgetting of previously learned tasks and maintaining consistency across changing robot hardware.

### 4.1.5 Paradigm Comparison

| Paradigm | Data Efficiency | Generalization | Compute Cost | Adaptation Speed |
|---|---|---|---|---|
| From-scratch (ACT) | Low — needs 50–500+ demos per task | Narrow — in-distribution only | Low — 1 GPU, hours | Fast — retrain in hours |
| Pre-train + fine-tune (OpenVLA) | High — 10–50 demos can suffice | Broad — inherits VLM knowledge | Medium — fine-tune on 1–8 GPUs | Moderate — hours to days |
| Co-training (RT-2, π₀.5) | Medium — needs large total dataset | Broad — maintained web grounding | High — hundreds of TPUs/GPUs | Slow — full training run needed |
| Continual learning | High in theory | Moderate — catastrophic forgetting risk | Low per update | Very fast | 

---

## 4.2 Behavioral Cloning Training Loop

Behavioral cloning (BC) reduces imitation learning to supervised regression: given observations, predict actions. The training loop is intentionally simple.

```python
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

for batch in dataloader:
    images, joint_pos, actions = batch
    # images:    (B, T_obs, C, H, W)
    # joint_pos: (B, D_q)
    # actions:   (B, D_a)

    pred_actions = policy(images, joint_pos)

    # L1 or MSE depending on task
    loss = F.mse_loss(pred_actions, actions)  # or F.l1_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 4.2.1 L1 vs MSE Loss

Both losses minimize prediction error, but their geometry differs in important ways.

**MSE** penalizes large errors quadratically. This makes the optimum the conditional *mean* of the action distribution. When the expert action distribution is multimodal (e.g., the expert sometimes goes left and sometimes goes right), MSE pushes the prediction toward the average — a position neither mode actually occupies, leading to blurry or physically infeasible actions.

**L1** penalizes errors linearly. The optimum is the conditional *median*, which is more robust to outliers and tends to produce sharper predictions. ACT uses L1 for its reconstruction loss precisely for this reason ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)).

In practice:
- Use L1 when action distributions may be multimodal or when outlier demonstrations are present.
- Use MSE when the action distribution is unimodal and well-behaved (e.g., simple pick-and-place with consistent demonstrations).

### 4.2.2 Absolute vs Delta Actions

**Absolute actions** represent the target state directly (e.g., target joint angles). They are easier to supervise but sensitive to initialization: if the robot starts in a slightly different configuration, absolute actions may not transfer.

**Delta actions** represent changes relative to the current state (e.g., $\Delta q_t = q_{t+1} - q_t$). They generalize better to starting-pose variation and are more natural for velocity-controlled robots, but compound integration errors over long horizons.

A common compromise: use delta actions for Cartesian end-effector control (small, bounded deltas) and absolute actions for joint-space control where the target configuration is well-defined.

### 4.2.3 Normalization Strategies

Normalization is critical. Without it, high-variance action dimensions (e.g., a base joint spanning ±180°) will dominate the loss and slow convergence for low-variance dimensions (e.g., a finger joint spanning ±5°).

**Per-dimension normalization** to zero mean and unit variance (computed on the training set) is the standard approach:

```math
\hat{a}_i = \frac{a_i - \mu_i}{\sigma_i + \epsilon}
```

For joint positions, normalizing to $[-1, 1]$ using dataset min/max is common in ACT:

```math
\hat{q}_i = 2 \cdot \frac{q_i - q_i^{\min}}{q_i^{\max} - q_i^{\min}} - 1
```

Store the normalization statistics alongside model weights so they can be applied consistently at inference time.

---

## 4.3 ACT Training in Detail

Action Chunking with Transformers (ACT) ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)) extends plain BC with two key ideas: (1) predicting *chunks* of future actions rather than a single step, and (2) using a Conditional Variational Autoencoder (CVAE) to model the latent style of the expert's behavior, avoiding mode-averaging.

### 4.3.1 Full Training Loop with CVAE

During training, the CVAE encoder sees the ground-truth actions and produces a distribution over a latent style vector $z$. The decoder (the policy network) receives the observation *and* a sample from this distribution, and must reconstruct the action chunk.

```python
optimizer = torch.optim.AdamW(
    list(cvae_encoder.parameters()) + list(policy.parameters()),
    lr=1e-5
)

for batch in dataloader:
    images, joint_pos, actions = batch
    # images:    (B, num_cams, C, H, W)
    # joint_pos: (B, D_q)           — current joint positions
    # actions:   (B, k, D_a)        — k=100 future steps, D_a=14 for bimanual

    # --- CVAE encoder: approximate posterior q(z | a, q) ---
    mu, log_var = cvae_encoder(actions, joint_pos)   # (B, 32) each

    # Reparameterization trick
    std = (0.5 * log_var).exp()
    eps = torch.randn_like(std)
    z = mu + eps * std                               # (B, 32)

    # --- Policy decoder: p(a | images, q, z) ---
    pred_actions = policy(images, joint_pos, z)      # (B, k, D_a)

    # --- Losses ---
    recon_loss = F.l1_loss(pred_actions, actions)

    # KL divergence: D_KL[q(z|a,q) || p(z)] with p(z) = N(0, I)
    kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1).mean()

    beta = 10.0
    loss = recon_loss + beta * kl_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Why the KL term?** The KL loss regularizes $z$ toward a standard Gaussian prior $\mathcal{N}(0, I)$. At inference time, the encoder is not available (no ground-truth actions to condition on), so $z$ is set to the prior mean $\mathbf{0}$. The KL term ensures the encoder's posterior does not deviate too far from this prior, keeping the distribution well-matched.

**Why $\beta = 10$?** The $\beta$-VAE formulation ([Higgins et al., 2017](https://openreview.net/forum?id=Sy2fzU9gl)) up-weights the KL term to encourage a more disentangled and regular latent space. In ACT, $\beta = 10$ was found empirically to balance reconstruction quality against posterior collapse.

The KL loss for a diagonal Gaussian posterior has the closed-form:

```math
\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{d_z} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
```

The total ACT loss is:

```math
\mathcal{L}_{\text{ACT}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}
```

### 4.3.2 Inference with Temporal Ensembling

At test time, the policy predicts a chunk of $k$ future actions at every timestep, but only needs to execute one action. Naïvely executing only the first predicted action discards useful signal from the rest of the chunk. ACT uses *temporal ensembling*: overlapping predictions from multiple recent inference calls are combined with exponential weighting, giving more influence to fresher predictions.

```python
from collections import defaultdict
import math

action_buffer = defaultdict(list)  # maps future timestep -> list of (action, weight)
m = 0.01  # exponential decay rate

for t in range(episode_len):
    obs = get_observation()

    # At test time, use the prior mean z = 0
    z = torch.zeros(1, 32, device=device)
    actions_chunk = policy(obs.images, obs.joint_pos, z)  # (1, k, D_a)
    actions_chunk = actions_chunk[0]                       # (k, D_a)

    # Register this chunk's predictions into the buffer
    for i, a in enumerate(actions_chunk):
        weight = math.exp(-m * i)
        action_buffer[t + i].append((a.detach(), weight))

    # Compute weighted average of all predictions for current timestep
    acts, weights = zip(*action_buffer[t])
    w_tensor = torch.tensor(weights, device=device)
    a_stack = torch.stack(acts, dim=0)           # (num_predictions, D_a)
    current_action = (w_tensor[:, None] * a_stack).sum(0) / w_tensor.sum()

    execute(current_action)
    del action_buffer[t]  # free memory
```

The weight assigned to prediction $i$ steps in the future is:

```math
w_i = e^{-m \cdot i}
```

This gives full weight to a prediction made for the *current* step and exponentially less weight to predictions that were made far in advance. Temporal ensembling smooths out jitter in individual predictions and has been shown to improve task success rates by several percentage points ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)).

### 4.3.3 ACT Hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Learning rate | $1 \times 10^{-5}$ | AdamW; lower than typical due to pretrained backbone |
| Batch size | 8 | Limited by GPU memory for multi-camera setups |
| Chunk size $k$ | 100 | Covers ~2 seconds at 50 Hz |
| Latent dim $d_z$ | 32 | Style vector size |
| $\beta$ (KL weight) | 10 | Increase if posterior collapse observed |
| Training steps | ~20,000 | Converges in ~5 hours on a single RTX 2080 Ti |
| Image augmentation | None (default) | Small datasets risk overfitting with augmentation |
| Joint normalization | $[-1, 1]$ | Per-joint min/max from training set |

---

## 4.4 Diffusion Policy Training

Diffusion Policy ([Chi et al., 2023](https://arxiv.org/abs/2303.04137)) frames action generation as a denoising process. Rather than predicting actions directly, the network learns to denoise a Gaussian noise vector into a clean action sequence over $T$ diffusion steps. This naturally handles multimodal action distributions without requiring a VAE.

### 4.4.1 DDPM Training

The Denoising Diffusion Probabilistic Model (DDPM) forward process gradually adds noise to the clean action $a_0$:

```math
q(a_t | a_0) = \mathcal{N}\!\left(a_t;\, \sqrt{\bar{\alpha}_t}\, a_0,\, (1 - \bar{\alpha}_t) I\right)
```

where $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$ and $\{\beta_s\}$ is a noise schedule. The network is trained to predict the noise $\epsilon$ that was added:

```python
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(
    num_train_timesteps=100,
    beta_schedule="squaredcos_cap_v2",   # cosine schedule
    clip_sample=True,
    prediction_type="epsilon"
)

obs_encoder = ObsEncoder()   # encodes images + joint_pos -> feature vector
noise_net = ConditionalUNet1D()  # or Transformer-based noise predictor

optimizer = torch.optim.AdamW(
    list(obs_encoder.parameters()) + list(noise_net.parameters()),
    lr=1e-4
)

for batch in dataloader:
    obs, actions = batch
    # obs.images:    (B, T_obs, C, H, W)
    # obs.joint_pos: (B, T_obs, D_q)
    # actions:       (B, T_pred, D_a)   — prediction horizon

    B = actions.shape[0]

    # Sample random noise and diffusion timesteps
    noise = torch.randn_like(actions)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (B,), device=actions.device).long()

    # Forward diffusion: add noise to actions
    noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

    # Encode observations into conditioning features
    obs_features = obs_encoder(obs.images, obs.joint_pos)  # (B, D_obs)

    # Predict the noise that was added
    pred_noise = noise_net(noisy_actions, timesteps, obs_features)

    loss = F.mse_loss(pred_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

MSE on the predicted noise is appropriate here because the network is predicting a single noise target — not a multimodal action distribution. The multimodality in the *action* space is captured by the stochastic denoising process itself.

### 4.4.2 DDIM Inference

Full DDPM inference requires 100 sequential denoising steps, which is too slow for real-time control. Denoising Diffusion Implicit Models (DDIM) dramatically accelerate this: 10 DDIM steps produce action quality comparable to 100 DDPM steps by taking larger, deterministic integration steps.

```python
from diffusers import DDIMScheduler

ddim_scheduler = DDIMScheduler(num_train_timesteps=100)
ddim_scheduler.set_timesteps(10)  # use 10 inference steps

def diffusion_inference(obs, obs_encoder, noise_net, device):
    obs_features = obs_encoder(obs.images, obs.joint_pos)

    # Start from pure noise
    noisy_actions = torch.randn(
        1, T_pred, D_a, device=device
    )

    for t in ddim_scheduler.timesteps:      # [900, 800, ..., 0] with 10 steps
        t_batch = t.unsqueeze(0).to(device)
        pred_noise = noise_net(noisy_actions, t_batch, obs_features)
        noisy_actions = ddim_scheduler.step(
            pred_noise, t, noisy_actions
        ).prev_sample

    return noisy_actions  # clean action sequence
```

### 4.4.3 Receding Horizon Execution

Like ACT, Diffusion Policy uses action chunking with a receding horizon to balance responsiveness and smoothness:

| Parameter | Typical Value | Meaning |
|---|---|---|
| $T_p$ (prediction horizon) | 16 steps | Length of generated action chunk |
| $T_a$ (action horizon) | 8 steps | Number of actions actually executed |
| $T_o$ (observation horizon) | 2 steps | Observation context length |

Predicting $T_p = 16$ but only executing $T_a = 8$ ensures the policy can anticipate further into the future while remaining reactive to new observations every 8 steps. This is a key design choice: larger $T_a/T_p$ ratios increase reactivity but reduce the benefit of planning ahead.

---

## 4.5 VLA Fine-Tuning

Vision-Language-Action models (VLAs) inherit large pretrained vision-language backbones and extend them with action prediction heads. Fine-tuning these models efficiently requires careful choices about which parameters to update.

### 4.5.1 OpenVLA Fine-Tuning with LoRA

OpenVLA ([Kim et al., 2024](https://arxiv.org/abs/2406.09246)) is a 7B-parameter VLA built on Prismatic VLM. Full fine-tuning at this scale requires significant compute and risks destroying pretrained visual-language grounding. Low-Rank Adaptation (LoRA) adds small trainable rank-decomposition matrices to selected layers, updating only ~1.5% of total parameters while preserving the frozen backbone.

```python
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=32,                                   # rank of the update matrices
    lora_alpha=16,                          # scaling factor (effective lr = alpha/r)
    target_modules=["q_proj", "v_proj"],    # apply LoRA to attention Q and V
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 107,347,968 || all params: 7,339,425,792 || trainable: 1.46%

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

for batch in dataloader:
    input_ids, attention_mask, action_token_ids = batch
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=action_token_ids
    )
    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

LoRA works by replacing the weight update $\Delta W \in \mathbb{R}^{d \times d}$ with a low-rank factorization:

```math
\Delta W = \frac{\alpha}{r} \cdot BA, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times d}, \quad r \ll d
```

At initialization, $A$ is drawn from a Gaussian and $B = 0$, so $\Delta W = 0$ and the model starts from the pretrained weights.

### 4.5.2 Full Fine-Tuning vs LoRA vs Frozen Backbone

| Strategy | Trainable Params | When to Use |
|---|---|---|
| Frozen backbone | Action head only (~1M) | Very small datasets (<50 demos); risk of overfitting is high |
| LoRA | ~1–5% of total | Most fine-tuning scenarios; 50–500 demos; single GPU feasible |
| Full fine-tuning | 100% | Large task-specific datasets (1000+ demos); multi-GPU required; best final performance |

**Practical guidance:**
- Start with LoRA. If performance plateaus and you have enough data, consider full fine-tuning.
- Frozen backbone fine-tuning is rarely optimal; even the lightest LoRA typically outperforms it.
- When using full fine-tuning, apply a *differential learning rate*: smaller LR for the backbone (e.g., $2 \times 10^{-5}$) and larger LR for the action head (e.g., $1 \times 10^{-4}$).

### 4.5.3 Action Tokenization for VLAs

VLAs produce actions by generating discrete tokens using the language model's existing vocabulary, extended with special action tokens. Continuous action values are discretized into bins:

```python
def tokenize_action(
    action_vector: torch.Tensor,
    num_bins: int = 256,
    action_range: tuple = (-1.0, 1.0)
) -> torch.Tensor:
    """Discretize a continuous action vector into integer bin indices."""
    lo, hi = action_range
    normalized = (action_vector - lo) / (hi - lo)          # [0, 1]
    normalized = normalized.clamp(0.0, 1.0)
    token_ids = (normalized * (num_bins - 1)).round().long()
    return token_ids                                        # values in [0, num_bins - 1]


def detokenize_action(
    token_ids: torch.Tensor,
    num_bins: int = 256,
    action_range: tuple = (-1.0, 1.0)
) -> torch.Tensor:
    """Reconstruct a continuous action vector from bin indices."""
    lo, hi = action_range
    normalized = token_ids.float() / (num_bins - 1)        # [0, 1]
    return normalized * (hi - lo) + lo                     # [lo, hi]
```

With 256 bins over a $[-1, 1]$ range, the quantization error per dimension is at most:

```math
\epsilon_{\text{quant}} = \frac{2}{2 \times 256} = \frac{1}{256} \approx 0.004
```

This is typically within acceptable tolerance for robot control. OpenVLA uses 256 bins per action dimension, appending action tokens to the end of the language output sequence so that the standard cross-entropy training objective applies uniformly.

**Trade-off note:** Tokenization makes training architecturally simple (reuse the LM head), but loses the continuous action space. For fine-grained manipulation (e.g., sub-millimeter assembly), consider a hybrid approach: a discrete token selects a mode, and a continuous regression head refines within that mode.

---

## 4.6 Co-Training Strategies

### 4.6.1 Why Co-Training

The fundamental challenge in fine-tuning large models on robot data is *catastrophic forgetting*: gradient updates that optimize for robot tasks overwrite the weights encoding web knowledge. This hurts zero-shot generalization and instruction following.

Co-training addresses this by continuously mixing in non-robot data throughout training, so the model is always being updated on both task types. The result is a model that can follow novel language instructions (because it retains VLM knowledge) and execute physical skills (because it has robot data).

### 4.6.2 Mixing Ratio

The ratio of robot data to non-robot (VQA, captioning, web) data is a critical hyperparameter. RT-2 ([Brohan et al., 2023](https://arxiv.org/abs/2307.15818)) found that a roughly 50/50 mix (by number of tokens seen) strikes a good balance: too much robot data degrades language benchmarks, too little slows motor skill acquisition.

A practical approach is *task-proportional sampling*: sample each data source with probability proportional to its dataset size, capped to prevent any single source from dominating:

```python
dataset_sizes = {
    "robot_demos": 130_000,
    "vqa_data":    1_000_000,
    "web_captions": 5_000_000,
}

# Cap the influence of large datasets with temperature scaling
temperature = 0.7
raw_weights = {k: v ** temperature for k, v in dataset_sizes.items()}
total = sum(raw_weights.values())
sampling_probs = {k: v / total for k, v in raw_weights.items()}
```

Temperature $\tau < 1$ compresses the distribution, giving more weight to smaller datasets (including robot data) than their raw size would suggest.

### 4.6.3 π₀.5 Co-Training

π₀.5 extends co-training to *heterogeneous robot data*: demonstrations from different robot morphologies (single-arm, bimanual, mobile manipulators) and different tasks are mixed in a single training run. Key design choices:

- **Semantic prediction heads:** In addition to action tokens, the model predicts high-level semantic outputs (object categories, task completion signals). This encourages the model to build shared semantic representations across tasks.
- **Per-morphology action heads:** While the language backbone is shared, separate lightweight action decoders are used per robot morphology to handle different action spaces without cross-morphology confusion.
- **Curriculum within co-training:** Easy tasks and low-noise demonstrations are up-weighted early in training to provide stable gradients; harder tasks are phased in as the model stabilizes.

---

## 4.7 Training Infrastructure

### 4.7.1 Distributed Training

For models that fit on a single GPU (ACT, small Diffusion Policy), single-machine training is straightforward. For larger models, two parallelism strategies dominate:

**Data Parallelism (DP):** Each GPU holds a full copy of the model and processes a different mini-batch. Gradients are averaged across GPUs before the optimizer step. Scales well to many GPUs for models that fit in a single GPU's memory.

```python
# PyTorch DDP setup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
model = DDP(model, device_ids=[local_rank])
```

**Model Parallelism (MP) / Tensor Parallelism:** For models too large to fit on one GPU (7B+ VLAs), model layers are split across GPUs. Libraries like DeepSpeed ZeRO, Megatron-LM, or FSDP (Fully Sharded Data Parallel) handle this automatically.

```python
# FSDP for large VLA fine-tuning
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32
    )
)
```

### 4.7.2 Mixed Precision

**BF16 (brain float 16):** Preferred for large VLMs. BF16 has the same exponent range as FP32 (8 exponent bits) but fewer mantissa bits (7 vs 23). This prevents the overflow/underflow issues that plague FP16 for large-scale training, while still halving memory usage and doubling arithmetic throughput on modern accelerators (A100, H100, TPU-v4+).

**FP32:** Preferred for small models (ACT, vanilla BC) where the compute overhead is negligible and numerical precision matters for gradient stability.

**FP16:** Acceptable for inference; avoid for training large models due to overflow risk.

As a rule of thumb: use BF16 for any model with more than ~100M parameters, FP32 for everything smaller.

### 4.7.3 Gradient Checkpointing

Gradient checkpointing (also called activation checkpointing) trades compute for memory by recomputing activations during the backward pass rather than storing them all. For Transformer-based models, this typically reduces activation memory by 60–70% at the cost of ~30% more compute per backward pass.

```python
# Enable gradient checkpointing in HuggingFace models
model.gradient_checkpointing_enable()

# Or manually via PyTorch
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(layer, x):
    return checkpoint(layer, x, use_reentrant=False)
```

Use gradient checkpointing when training VLAs at full fine-tune scale, especially when batch size is memory-constrained.

### 4.7.4 Typical Training Setups

| Model | Hardware | Training Time | Notes |
|---|---|---|---|
| ACT | 1× RTX 2080 Ti (11GB) | ~5 hours | ~20k steps, batch size 8 |
| Diffusion Policy (CNN) | 1× RTX 3090 (24GB) | ~12 hours | 3000 epochs, batch size 64 |
| OpenVLA fine-tune (LoRA) | 4× A100 (80GB) | 4–8 hours | Single task, 200 demos |
| OpenVLA full fine-tune | 64× A100 | ~2 days | Multi-task |
| OpenVLA pretraining | 2048× TPU-v5e | ~2 weeks | From scratch on OXE dataset |
| π₀ | 256× H100 (estimated) | Days–weeks | Full pretraining + post-training |

---

## 4.8 Common Training Pitfalls and Solutions

### 4.8.1 Posterior Collapse in CVAE

**Symptom:** The KL loss quickly drops to near zero early in training. The encoder stops using the latent variable $z$, and the model effectively ignores the style information. The decoder learns to predict actions without using $z$.

**Diagnosis:** Monitor $\mathcal{L}_{\text{KL}}$ separately from $\mathcal{L}_{\text{recon}}$. If KL < 0.1 nats within the first 1000 steps, collapse is likely occurring.

**Solutions:**
1. **Increase $\beta$:** A higher KL weight forces the encoder to maintain a non-trivial posterior. Try $\beta \in \{10, 20, 50\}$.
2. **KL annealing:** Start with $\beta = 0$ (pure reconstruction) and linearly increase to the target $\beta$ over the first 5000 steps. This gives the encoder time to learn useful representations before KL pressure is applied.
3. **Reduce decoder capacity:** A decoder that is too powerful can ignore $z$ entirely. Consider adding dropout or reducing the number of decoder layers.

### 4.8.2 Mode Averaging in Plain BC

**Symptom:** The trained policy produces hesitant, "average" actions when the demonstration data contains multiple valid strategies for a task. For example, when an object can be grasped from either side, the policy reaches toward the midpoint between the two grasp positions.

**Root cause:** MSE minimization converges to the conditional mean of the action distribution. When this distribution is multimodal, the mean is not a valid action.

**Solutions:**
- Switch from plain BC to **ACT** (CVAE captures multi-modality via the latent variable) or **Diffusion Policy** (denoising process naturally represents multimodal distributions).
- If staying with plain BC, collect more demonstrations that are consistent in strategy (reduce the modality of the dataset).

### 4.8.3 Overfitting with Small Datasets

**Symptom:** Training loss continues to decrease but validation loss or real-world success rate plateaus or worsens. The model memorizes specific demonstration trajectories rather than learning generalizable skills.

**Solutions:**
- **Image augmentation:** Color jitter, random crops, Gaussian blur. Even modest augmentation significantly helps. However, be conservative: aggressive spatial augmentation (large crops, rotations) can destroy depth cues critical for manipulation.
- **L2 weight decay:** `AdamW` with `weight_decay=1e-4` is standard.
- **Reduce model capacity:** If you have 50 demos, you do not need a 100M parameter model.
- **Early stopping:** Monitor validation loss and stop when it begins to increase.

### 4.8.4 Instability with High-Dimensional Action Spaces

**Symptom:** Loss oscillates or diverges when predicting 7+ DOF joint actions, especially for bimanual robots (14 DOF).

**Solutions:**
- **Normalize all action dimensions** to $[-1, 1]$ or zero mean / unit variance. Unnormalized actions with large dynamic ranges destabilize gradients.
- **Use L1 loss instead of MSE:** L1 provides more uniform gradient magnitude across scales, while MSE's quadratic scaling amplifies large errors disproportionately.
- **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` prevents explosive gradients.
- **Reduce learning rate** and use a warm-up schedule (linear or cosine).

### 4.8.5 Camera Calibration Drift

**Symptom:** Policy performance degrades over days or weeks of deployment, even without changing the task, due to subtle shifts in camera position or lighting.

**Solutions:**
- **Per-episode image normalization:** Normalize image statistics (mean, std) within each episode rather than using fixed statistics from training time.
- **Augment training data** with color jitter and brightness variation so the model is robust to lighting changes.
- **Regular recalibration:** Establish a weekly recalibration ritual for camera extrinsics and check that workspace images match the training distribution.

---

## 4.9 Evaluation During Training

Monitoring training loss alone is insufficient. A policy with low training loss may still fail in deployment due to compounding errors, distribution shift, or the covariate shift problem inherent to BC. Regular rollout evaluation is essential.

### 4.9.1 Evaluation Protocol

The gold-standard metric for a manipulation policy is the **task success rate**: the fraction of rollout episodes in which the robot completes the task within an episode length limit.

```python
def evaluate_policy(policy, env, num_episodes=50, device="cuda"):
    """Run N rollouts and return success rate and mean episode length."""
    successes = 0
    episode_lengths = []

    policy.eval()
    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            step = 0

            while not done and step < env.max_episode_steps:
                action = policy.get_action(obs)     # handles chunking/ensembling internally
                obs, reward, done, info = env.step(action)
                step += 1

            successes += int(info.get("success", False))
            episode_lengths.append(step)

    return {
        "success_rate": successes / num_episodes,
        "mean_episode_length": sum(episode_lengths) / len(episode_lengths),
    }
```

### 4.9.2 Evaluation Schedule

Running 50 rollouts is expensive (often 10–30 minutes of real robot time). A practical schedule:

- **Early training:** Evaluate every 2000 steps to catch catastrophic failures quickly.
- **Mid training:** Evaluate every 5000 steps.
- **Near convergence:** Evaluate every 1000 steps to detect when performance plateaus.

For simulation-based evaluation (where rollouts are cheap), evaluate every 500–1000 steps.

### 4.9.3 Key Metrics

| Metric | Description | Notes |
|---|---|---|
| `success@50` | Success rate over 50 rollouts | Primary metric; report with 95% CI |
| Mean episode length | Average steps to task completion | Lower is better for fast tasks |
| Contact quality | Force/torque during contact events | Proxy for skill smoothness |
| Policy FPS | Inference frequency | Must exceed control frequency |

**Confidence intervals:** With 50 rollouts, the 95% confidence interval for a success rate $\hat{p}$ is approximately:

```math
\hat{p} \pm 1.96 \sqrt{\frac{\hat{p}(1-\hat{p})}{N}}
```

For $\hat{p} = 0.7, N = 50$, this gives $\pm 0.127$ — a wide interval. Report confidence intervals alongside success rates to avoid over-interpreting small differences.

### 4.9.4 Detecting Overfitting vs Underfitting

| Observation | Likely Cause | Action |
|---|---|---|
| Training loss ↓, success rate flat | Compounding errors / covariate shift | Add DAgger, or add more diverse demonstrations |
| Training loss ↓, success rate ↓ | Overfitting | Add augmentation, reduce model capacity |
| Training loss high, success rate low | Underfitting | Increase model capacity, train longer, check normalization |
| Training loss oscillating | LR too high or unnormalized actions | Reduce LR, clip gradients, check normalization |

---

## Chapter Summary

This chapter covered the full training lifecycle for imitation learning policies:

- **Paradigm selection** (from-scratch, pre-train+fine-tune, co-training) determines data efficiency, compute requirements, and generalization. For most practitioners, pre-train+fine-tune with LoRA is the best starting point.
- **BC training loops** are simple but sensitive to loss function choice (L1 vs MSE), action representation (absolute vs delta), and normalization.
- **ACT** adds a CVAE to handle multi-modal demonstrations, with temporal ensembling at inference to smooth execution. Key settings: $\beta = 10$, chunk size 100, $z = \mathbf{0}$ at test time.
- **Diffusion Policy** handles multi-modality through the denoising process itself. Use DDIM (10 steps) for real-time inference and receding-horizon execution ($T_p = 16, T_a = 8$).
- **VLA fine-tuning** with LoRA updates ~1.5% of parameters while preserving pretrained representations; action tokenization maps continuous values to discrete bins.
- **Co-training** maintains web knowledge by mixing robot and non-robot data throughout training; task-proportional sampling with temperature controls the effective mixing ratio.
- **Infrastructure choices** — BF16 mixed precision, FSDP, gradient checkpointing — are critical for training large VLAs efficiently.
- **Common pitfalls** (posterior collapse, mode averaging, overfitting) each have well-understood diagnostics and mitigations.
- **Evaluation** must include real rollouts (success@50), not just training loss; report confidence intervals.

---

## References

- Zhao, T., Kumar, V., Levine, S., & Finn, C. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. *arXiv:2304.13705*. <https://arxiv.org/abs/2304.13705>
- Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. *arXiv:2303.04137*. <https://arxiv.org/abs/2303.04137>
- Kim, M. J., Pertsch, K., Karamcheti, S., Mees, T., Balakrishna, A., Burchfiel, B., ... & Finn, C. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. *arXiv:2406.09246*. <https://arxiv.org/abs/2406.09246>
- Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., ... & Zeng, A. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. *arXiv:2307.15818*. <https://arxiv.org/abs/2307.15818>
- Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., ... & Loquercio, A. (2024). π₀: A Vision-Language-Action Flow Model for General Robot Control. *arXiv:2410.24164*. <https://arxiv.org/abs/2410.24164>
- Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR 2017*. <https://openreview.net/forum?id=Sy2fzU9gl>
