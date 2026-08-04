# Chapter 6: Compute Requirements

This chapter gives practitioners a concrete, hardware-grounded view of what it costs to train, fine-tune, and deploy imitation learning policies — from a single consumer GPU on a lab bench to a multi-thousand-accelerator cluster. The goal is not to dwell on specifications for their own sake, but to equip you to make sound hardware decisions for your own project.

---

## 1. Compute Overview

Imitation learning policies span roughly six orders of magnitude in parameter count — from the ~10k-parameter ALVINN network of the 1980s to the 55B-parameter RT-2 — and the associated compute requirements span a similarly vast range. Three dimensions matter most in practice:

| Dimension | Why it matters |
|---|---|
| **Training compute** | Total GPU/TPU-hours; determines cloud cost and time-to-result |
| **Inference latency** | Dictates which control frequencies are achievable on-robot |
| **Memory footprint** | Sets the minimum VRAM for a given hardware platform |

These three dimensions interact. Reducing memory through quantization typically reduces inference latency; reducing parameters through distillation affects both training and inference. The spectrum of hardware spans:

- **Consumer GPU (RTX 2080 Ti / RTX 3090):** 11–24 GB VRAM; suitable for small policies and fine-tuning
- **Workstation GPU (A100 40/80 GB, H100 80 GB):** Standard research/industry training unit
- **Multi-GPU nodes (8× A100):** Medium-scale pre-training of 100M–1B parameter models
- **TPU pods / GPU clusters (256+ chips):** Foundation model pre-training; VLA training at scale

---

## 2. Small-Scale: ACT and Similar

**Action Chunking with Transformers (ACT)** represents the entry point for modern IL practitioners: a transformer-encoder/decoder architecture that predicts a chunk of $k$ future actions, trained with a CVAE objective.

**Parameters:** ~80 million (visual encoder + transformer)

**Training hardware:** 1× NVIDIA RTX 2080 Ti (11 GB VRAM)
- Batch size: 8
- GPU VRAM during training: ~4 GB (well within 11 GB budget)
- Wall-clock training time: approximately 5 hours for a single-task policy on a few hundred demonstrations

**Inference:**
- Single forward pass: ~10 ms on the same GPU
- This satisfies a 100 Hz budget; ACT is typically deployed at **50 Hz** in practice
- The action-chunking mechanism means the policy executes a pre-planned chunk at each inference call, amortizing the 10 ms cost across multiple timesteps

**Suitability:** Individual researchers, university labs, robot arm benchtop setups. No cloud compute required. The small footprint also means fast iteration: a new experiment can be trained overnight.

```python
# ACT inference loop (pseudocode)
model = ACTPolicy(num_queries=100, hidden_dim=512, ...)
model.load_state_dict(torch.load("act_checkpoint.pt"))
model.eval()

chunk_size = 20       # execute 20 actions per inference call
control_freq_hz = 50  # 20ms per step

obs_buffer = deque(maxlen=1)
action_queue = deque()

with torch.no_grad():
    while robot.is_running():
        if len(action_queue) == 0:
            obs = obs_buffer[-1]         # dict: images + qpos
            actions = model(obs)         # shape: (chunk_size, action_dim)
            action_queue.extend(actions)
        
        action = action_queue.popleft()
        robot.apply_action(action)
        time.sleep(1.0 / control_freq_hz)
```

---

## 3. Medium-Scale: Octo and Diffusion Policy

### Octo

Octo is a transformer-based generalist robot policy pre-trained on the Open X-Embodiment dataset. It uses a modular architecture: a shared token-based observation trunk followed by task-specific diffusion or regression heads.

- **Parameters:** ~90 million
- **Pre-training hardware:** 8× NVIDIA A100 (40 GB)
- **Pre-training time:** ~24 hours
- **Inference:** 30–50 ms (regression head); longer with diffusion head

### Diffusion Policy

Diffusion Policy frames action prediction as a conditional denoising diffusion process:

```math
\mathbf{a}_0 = \text{DenoisingNet}\left(\mathbf{a}_T, \mathbf{o}, T\right), \quad \mathbf{a}_T \sim \mathcal{N}(0, I)
```

Two architectural variants:
- **CNN variant:** ~256M parameters; 1–2× A100 for 4–8 hours training
- **Transformer variant:** similar scale

**Inference latency — the critical tradeoff:**

| Scheduler | Denoising Steps | Latency | Control Frequency |
|---|---|---|---|
| DDPM | 100 | 100–200 ms | 5–10 Hz |
| DDIM | 10 | 10–20 ms | 50–100 Hz |

For real-time robot control, DDIM with 10 steps is the standard choice. The quality penalty relative to DDPM is minimal for unimodal action distributions but can matter for highly multimodal tasks.

**Memory optimizations for medium-scale training:**

```python
# Mixed precision + gradient checkpointing for Diffusion Policy on a single A100
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint_sequential

scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)

for batch in dataloader:
    obs, actions = batch
    
    with autocast(dtype=torch.bfloat16):
        # Gradient checkpointing on the visual encoder backbone
        # trades compute for memory: recomputes activations during backward
        noise = torch.randn_like(actions)
        t = torch.randint(0, num_diffusion_steps, (actions.shape[0],))
        noisy_actions = noise_scheduler.add_noise(actions, noise, t)
        
        pred_noise = model(noisy_actions, obs, t)
        loss = F.mse_loss(pred_noise, noise)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

BF16 mixed precision cuts VRAM by roughly 40% with negligible numerical impact for IL training, making the CNN Diffusion Policy trainable on a single A100 40 GB card.

---

## 4. Large-Scale: OpenVLA and π₀

At this scale, policies are built on top of pre-trained vision-language model (VLM) backbones, and the resource asymmetry between pre-training and fine-tuning becomes dramatic.

### OpenVLA

OpenVLA is a 7B-parameter open-source vision-language-action model built on Llama 2, with a visual encoder that fuses features from DINOv2 and SigLIP ([Kim et al., 2024](https://arxiv.org/abs/2406.09246)).

**Pre-training:**
- Hardware: 2048× TPU-v5e chips (~5,000 TPU-v5e-hours estimated)
- Data: 970k real-world robot demonstrations from Open X-Embodiment
- Trained at scale with BF16 precision

**Fine-tuning with LoRA (the practitioner path):**

```python
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Load base model in 4-bit for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
base_model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

# LoRA: only ~100M parameters become trainable out of 7B
lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 98,304,000 || all params: 7,241,728,000
#         trainable%: 1.36%
```

- Hardware: 1× A100 (80 GB), approximately 6 hours on task-specific data
- Reduces trainable parameters from 7B to ~100M

**Inference latency:**

| Mode | Latency | Effective Control Rate |
|---|---|---|
| FP16 (full precision) | ~200 ms | ~5 Hz |
| INT4 quantization | ~40 ms | ~25 Hz |

**Memory footprint:**

| Precision | VRAM (inference) |
|---|---|
| FP16 | ~14 GB |
| INT4 (BitsAndBytes) | ~4 GB |

OpenVLA demonstrates a key empirical result: it outperforms RT-2-X (55B parameters) by 16.5% absolute task success across 29 tasks, with 7× fewer parameters ([Kim et al., 2024](https://arxiv.org/abs/2406.09246)). Parameter count is not the primary driver of downstream manipulation performance.

### π₀ (Pi-Zero)

π₀ (Physical Intelligence, 2024) combines a pre-trained VLM backbone (estimated 3–7B parameters) with a flow matching action head, enabling smooth, continuous action generation.

**Architecture:** VLM backbone for language and vision understanding; flow matching head for action generation:

```math
\frac{d\mathbf{a}}{dt} = v_\theta(\mathbf{a}_t, t, \mathbf{o}), \quad t \in [0, 1]
```

**Compute estimates:**
- Pre-training: ~256× H100 GPUs, several weeks (estimated; not officially disclosed)
- Inference: flow matching solved in ~10 NFE (neural function evaluations), ~50 ms per action
- Fine-tuning: VLM backbone frozen or partially frozen; flow head trained on task data

---

## 5. Very Large Scale: RT-2 (55B) and GATO

### RT-2

RT-2 (Robotic Transformer 2) is built on the PaLI-X and PaLM-E VLM backbones at 55B parameters — an order of magnitude larger than OpenVLA.

**Training:**
- Hardware: Estimated thousands of TPU-v4 chip-days (Google internal; not officially disclosed)
- The model was not trained with roboticists' typical hardware in mind; it represents what is feasible only at hyperscaler scale

**Inference:**
- ~1–2 seconds per action at full precision
- Not suitable for real-time robot control (≥10 Hz) without distillation or specialized hardware acceleration
- RT-2-X, the cross-embodiment variant, retains the 55B scale

**Key empirical lesson:** OpenVLA (7B) outperforms RT-2-X (55B) by 16.5% on generalist manipulation tasks ([Kim et al., 2024](https://arxiv.org/abs/2406.09246)). This strongly suggests that **data diversity and fine-tuning strategy matter more than raw parameter count** for robot manipulation. Scaling laws from NLP do not cleanly transfer to robotics.

### GATO

GATO (Reed et al., 2022) is a 1.2B–1.8B parameter multi-modal, multi-task agent that tokenizes everything — images, text, actions, observations — into a flat sequence and processes it with a causal transformer.

- Trained on 604 tasks including robot manipulation, video games, and language tasks
- Inference for robotics: ~100 ms, limiting control frequency
- Primarily a research vehicle demonstrating generalist sequential decision-making; not optimized for real-time deployment

---

## 6. Compute Efficiency Strategies

When real-time deployment or limited hardware are constraints, the following techniques are standard practice.

### 6.1 Quantization

Post-training quantization (PTQ) reduces model precision from FP32/BF16 to INT8 or INT4. For VLA-class models based on LLM backbones, the BitsAndBytes library provides seamless integration:

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # double-quantize the quantization constants
    bnb_4bit_quant_type="nf4"         # NF4 data type: normal float 4-bit
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

**Empirical results for VLAs:** INT4 quantization incurs less than 5% drop in task success rate while reducing VRAM from ~14 GB (FP16) to ~4 GB (INT4), and latency from ~200 ms to ~40 ms ([Kim et al., 2024](https://arxiv.org/abs/2406.09246)).

For smaller policies (ACT, Diffusion Policy), quantization is rarely necessary; the models are already compact enough for INT16 or FP16 operation.

### 6.2 LoRA / QLoRA Fine-tuning

Low-Rank Adaptation (LoRA) inserts trainable rank-$r$ matrices into the attention projection layers while freezing the original weights:

```math
W' = W + \Delta W = W + \frac{\alpha}{r} BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k}
```

For a 7B model with $r = 32$, LoRA reduces trainable parameters from 7B to ~100M — a 70× reduction in gradient memory.

**QLoRA** combines INT4 quantization of the frozen base model with LoRA adapters stored in BF16, making full-model fine-tuning feasible on a single A100 80 GB card for models up to ~13B parameters.

```python
# QLoRA fine-tuning loop (pseudocode)
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(base_model)
model = get_peft_model(model, lora_config)

for batch in robot_demos_dataloader:
    images, instructions, actions = batch
    
    with autocast(dtype=torch.bfloat16):
        outputs = model(
            pixel_values=images,
            input_ids=instructions,
            labels=actions
        )
        loss = outputs.loss
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Save only the LoRA adapters (~200MB), not the full 14GB model
model.save_pretrained("openvla_lora_task_specific")
```

### 6.3 Architecture Sparsification

Rather than post-hoc quantization, sparsification techniques reduce computation during the forward pass.

**MoLe-VLA (2025):** Mixture-of-Layers VLA with a Spatial-Temporal Aware Router (STAR) that dynamically skips LLM layers based on the robot's current state ([Zhang et al., 2025](https://arxiv.org/abs/2503.20384)). Key results:
- Up to 5.6× reduction in computational cost vs. a standard LLM-based VLA
- 8% *improvement* in mean success rate across ten RLBench tasks (the STAR router also acts as a regularizer)
- Cognition Self-Knowledge Distillation (CogKD) compensates for lost representational capacity from skipped layers

The core insight: not every input token requires all 32–40 transformer layers. Simple grasps can be resolved with shallow computation; complex spatial reasoning benefits from deep processing.

**MoE-ACT (2026 estimate):** Sparse mixture-of-experts extension of ACT, routing different manipulation primitives to specialized expert heads. Still emerging in the literature; reported improvements of 15–25% inference speedup on multi-task ACT benchmarks.

### 6.4 Distillation

Knowledge distillation trains a fast student policy to match the output distribution of a slow teacher policy:

```math
\mathcal{L}_\text{distill} = D_\text{KL}\left(p_\text{student}(\mathbf{a} | \mathbf{o}) \| p_\text{teacher}(\mathbf{a} | \mathbf{o})\right)
```

**One-Step Diffusion Policy (OneDP):** Distills a pre-trained multi-step diffusion policy into a single-step action generator by minimizing KL divergence along the diffusion chain ([Wang et al., 2024](http://arxiv.org/abs/2410.21257)). Results:
- Action prediction frequency improves from **1.5 Hz → 62 Hz** (41× speedup)
- Requires only 2–10% additional pre-training compute for convergence
- Evaluated on 6 simulation tasks and 4 real Franka robot tasks; achieves state-of-the-art success rates

```python
# OneDP distillation loss (pseudocode)
# teacher: pre-trained DDPM diffusion policy
# student: single-step generator

def onedp_loss(teacher, student, obs, real_actions):
    # Sample along the diffusion chain
    t_vals = torch.linspace(0, 1, num_steps)
    total_loss = 0
    
    for t in t_vals:
        noisy_actions = noise_scheduler.add_noise(real_actions, torch.randn_like(real_actions), t)
        
        with torch.no_grad():
            teacher_score = teacher.predict_score(noisy_actions, obs, t)
        
        student_score = student.predict_score(noisy_actions, obs, t)
        
        # KL along diffusion chain minimized by score matching
        total_loss += F.mse_loss(student_score, teacher_score)
    
    return total_loss / num_steps
```

---

## 7. Compute Reference Table

The table below consolidates published and estimated figures. Training hardware and time should be treated as approximate where marked — many papers do not report full infrastructure details.

| Model | Params | Training Hardware | Training Time | Inference Latency | VRAM (Inference) | Fine-tune Feasibility |
|---|---|---|---|---|---|---|
| ALVINN | ~10K | Workstation (1989) | Hours | <1 ms | Negligible | Full retrain trivial |
| BC baseline (MLP) | 1–10M | 1× RTX 2080 Ti | <1 h | <1 ms | <1 GB | Full retrain, minutes |
| ACT | ~80M | 1× RTX 2080 Ti | ~5 h | ~10 ms | ~4 GB | Full retrain on consumer GPU |
| Diffusion Policy (CNN) | ~256M | 1–2× A100 | 4–8 h | 5 ms (DDIM) / 100 ms (DDPM) | ~4 GB | Full retrain on single A100 |
| RT-1 | 35M | 8× TPU-v3 | ~2 days | ~20 ms | ~2 GB | Full retrain feasible |
| Octo | ~90M | 8× A100 | ~24 h | 30–50 ms | ~4 GB | Full retrain on 8× A100 |
| RDT-1B | 1.2B | Multi-GPU (unspecified) | Unspecified | ~50 ms | ~10 GB (BF16) | LoRA on 1× A100 80GB |
| RT-2 | 55B | 1000s of TPU-v4 chip-days | Weeks | ~1–2 s | ~110 GB (FP16) | LoRA only; impractical without TPU |
| OpenVLA | 7B | 2048× TPU-v5e | ~2 weeks est. | 40 ms (INT4) / 200 ms (FP16) | 4 GB (INT4) / 14 GB (FP16) | LoRA on 1× A100 80GB, ~6 h |
| π₀ | ~3–7B est. | ~256× H100 est. | Several weeks est. | ~50 ms (10-step flow) | ~8–14 GB | Flow head fine-tune on 1–4× A100 |
| π₀.5 | >7B est. | >256× H100 est. | Several weeks est. | ~50–100 ms | ~16 GB est. | Partial fine-tune; hardware requirements TBD |

*Estimates marked where official figures are not published. TPU and H100 figures are not directly comparable; H100 is roughly 2–3× faster than TPU-v5e for transformer inference.*

---

## 8. Real-Time Deployment Constraints

Real-time robot control imposes hard latency budgets that vary by the level of the control hierarchy:

| Control Level | Typical Frequency | Latency Budget | Policy Type |
|---|---|---|---|
| Low-level joint position | 500–1000 Hz | <1 ms | PD controller (not IL) |
| High-frequency IL (joint velocity/torque) | 50–100 Hz | 10–20 ms | ACT, Diffusion Policy (DDIM) |
| Mid-frequency arm Cartesian | 10–25 Hz | 40–100 ms | Diffusion Policy, Octo |
| High-level task/language grounding | 1–5 Hz | 200 ms–1 s | VLAs (OpenVLA, π₀) |

### Meeting Real-Time Requirements

**ACT at 50 Hz:** The 10 ms inference budget is comfortably met. The action-chunking mechanism (predicting 20–100 steps ahead) further smooths execution: even if the policy is queried every 20 timesteps, actions are interpolated at the full 50 Hz.

**Diffusion Policy at 10 Hz:** DDPM (100 steps, ~100 ms) falls short of a 100 ms budget. DDIM with 10 steps (~10 ms per step, ~10 ms total) is required. The step count is a hyperparameter to tune based on your latency budget.

**VLAs at 3–10 Hz:** Large VLAs (OpenVLA, π₀) run at 3–10 Hz even with INT4 quantization. Two deployment strategies address this:

1. **Hierarchical decomposition:** VLA runs at low frequency (2–5 Hz) and outputs a high-level goal or waypoint; a lightweight low-level policy (ACT or PD controller) executes at 50–100 Hz between VLA calls.

2. **Action chunking at VLA outputs:** The VLA predicts a sequence of future actions (e.g., 10 steps), which are executed open-loop at higher frequency.

### On-Robot vs. Cloud Inference

```
On-robot (edge):
  + No network latency
  + Works in network-constrained environments
  + Safer (no dependency on cloud connectivity)
  - Limited VRAM: NVIDIA Orin (16–32 GB unified memory) is the current ceiling for edge VLAs
  - Power and thermal constraints

Cloud inference:
  + Unlimited GPU VRAM; can run FP16 VLAs without quantization
  - Round-trip latency: 10–50 ms for LAN, 50–200 ms for WAN
  - Safety concern: network failure mid-task requires fallback policy
  - Privacy: robot observations leave the robot
```

For ACT and Diffusion Policy, on-robot inference on an RTX 3090 or Jetson AGX Orin is standard practice. For 7B VLAs, a workstation GPU (A100/H100 in a rack next to the robot) is the typical deployment target; full cloud inference is used primarily for evaluation and data collection rather than real-time control.

---

## Summary

The compute landscape for IL policies divides into three practical tiers:

1. **Consumer GPU (~80M params):** ACT-class models train in hours on a single RTX 2080 Ti and infer at 50 Hz. Entry point for any lab.
2. **Research cluster (90M–1.2B params):** Octo, Diffusion Policy, RDT-1B. Pre-training requires 8× A100 days; fine-tuning is accessible on 1–4 A100s.
3. **Foundation VLA (7B+):** OpenVLA and π₀. Pre-training requires hyperscaler infrastructure (TPU pods or H100 clusters), but LoRA fine-tuning on a single A100 80 GB democratizes task adaptation. Quantization to INT4 makes inference feasible on consumer hardware.

The critical practical takeaway: **parameter count does not predict manipulation performance.** OpenVLA (7B) outperforms RT-2-X (55B) while being 100× cheaper to fine-tune. Invest in data diversity and fine-tuning quality before scaling parameters.
