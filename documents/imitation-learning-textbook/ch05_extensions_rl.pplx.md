# Chapter 5 — Extensions: Reinforcement Learning, Hierarchical Policies, and Beyond

> **Who this chapter is for.** You have a working IL pipeline—behavioral cloning or DAgger—and it is not good enough. This chapter maps out the principled extensions: when to bolt RL on top, how to structure policies hierarchically, how to generalize across tasks and embodiments, and how to keep the system safe when it strays from the demonstration distribution.

---

## 5.1 Why IL Alone Is Not Enough

Behavioral cloning learns a mapping \(\pi_\theta : \mathcal{S} \to \mathcal{A}\) by minimizing supervised loss on demonstration data. The fundamental limitation is that the policy can never exceed the ceiling set by the demonstrations themselves.

### 5.1.1 The Coverage Problem

The demonstration dataset \(\mathcal{D} = \{(s_i, a_i)\}\) covers only a small slice of state space. Any state outside that slice is out-of-distribution for \(\pi_\theta\). Since the policy has never seen recovery from those states, it has no principled way to return to the nominal trajectory. Performance is bounded above by the expert's performance on the *exact* states in \(\mathcal{D}\).

### 5.1.2 Compounding Errors on Long Horizons

On a horizon of \(T\) steps with per-step error \(\epsilon\), the expected total cost under behavioral cloning grows as \(O(\epsilon T^2)\) rather than \(O(\epsilon T)\). The quadratic dependence on \(T\) makes long-horizon tasks fragile. DAgger reduces this to linear growth by querying the expert online, but it still cannot *discover* behaviors not present in any demonstration.

### 5.1.3 No Mechanism for Self-Improvement

IL policies are passive learners. They cannot use trial-and-error to discover that a particular grasp orientation leads to fewer drops, or that a slightly different approach angle avoids collisions. Every bit of improvement requires additional human effort in the form of new demonstrations.

### 5.1.4 The Need for Online Improvement

The natural remedy is to augment IL with a reinforcement signal. RL allows the policy to explore, fail, and update based on outcomes—without requiring new human demonstrations. The challenge is that RL on physical robots is expensive and risky, while RL in simulation suffers from the sim-to-real gap. The methods in this chapter address this tension with varying tradeoffs.

---

## 5.2 IRL and GAIL — The Classic Extensions

Before the era of large pre-trained models, two frameworks dominated the "IL + reward signal" literature: Inverse Reinforcement Learning (IRL) and Generative Adversarial Imitation Learning (GAIL). Understanding them is essential because their core ideas resurface in every modern variant.

### 5.2.1 Inverse Reinforcement Learning (IRL)

IRL flips the RL problem: instead of learning a policy given a reward, it learns a reward function \(R\) given expert demonstrations, then derives a policy from that reward.

**MaxEnt IRL (Ziebart et al., 2008).** The core insight is that among all reward functions consistent with observed behavior, choose the one that makes the demonstrated trajectories *most likely under a maximum-entropy distribution*. Formally, trajectories are distributed as:

\[
P(\tau) \propto \exp\!\left(R_\psi(\tau)\right)
\]

where \(R_\psi(\tau) = \sum_t r_\psi(s_t, a_t)\) is a learned reward parameterized by \(\psi\). The maximum-entropy principle avoids overfitting to idiosyncrasies of the demonstrations; it produces the least-committal reward consistent with the feature expectations observed in the data.

The bi-level optimization is:

\[
\max_\psi \; \min_\pi \; \mathbb{E}_\pi[R_\psi(\tau)] - \mathbb{E}_{\pi^*}[R_\psi(\tau)]
\]

The inner minimization finds the policy that minimizes the learned reward (i.e., that most looks like the expert). The outer maximization finds the reward that best separates the expert from the current policy. In practice, this requires alternating between running RL with the current reward and updating the reward to increase the margin.

**Limitations of IRL.** The alternating optimization is expensive—each reward update triggers a full RL solve. In high-dimensional visual domains, the reward parameterization must be learned jointly with the policy. GAIL addresses both issues.

### 5.2.2 GAIL — Generative Adversarial Imitation Learning

[Ho and Ermon (2016)](https://arxiv.org/abs/1606.03476) showed that IRL with a particular class of cost regularizers is equivalent to matching the *occupancy measure* (the joint distribution over state-action pairs) between the expert and the learned policy. This reframing yields a GAN-style training procedure that sidesteps the need for an explicit reward function.

**Discriminator objective.** A binary classifier \(D_\phi : \mathcal{S} \times \mathcal{A} \to [0,1]\) is trained to distinguish expert state-action pairs from policy-generated ones:

\[
\max_\phi \; \mathbb{E}_{\pi^*}[\log D_\phi(s,a)] + \mathbb{E}_{\pi_\theta}[\log(1 - D_\phi(s,a))]
\]

**Policy objective.** The policy is trained with RL using \(-\log D_\phi(s,a)\) as a proxy reward (the log-likelihood that the discriminator thinks the action came from the expert), regularized by a causal entropy term:

\[
\min_\theta \; -\mathbb{E}_{\pi_\theta}[\log D_\phi(s,a)] + \lambda H(\pi_\theta)
\]

The entropy bonus \(\lambda H(\pi_\theta)\) prevents the policy from collapsing to a single mode, mirroring the MaxEnt IRL spirit.

**Algorithm sketch.**

```python
# GAIL training loop (pseudocode)
for iteration in range(num_iterations):
    # 1. Collect on-policy rollouts
    policy_trajs = collect_rollouts(policy, env, n_steps)

    # 2. Update discriminator
    expert_batch = sample_expert(expert_buffer, batch_size)
    policy_batch = sample_policy(policy_trajs, batch_size)

    d_loss = -(expert_batch["log_D"] + (1 - policy_batch["D"]).log().mean())
    discriminator_optimizer.zero_grad()
    d_loss.backward()
    discriminator_optimizer.step()

    # 3. Compute proxy rewards from discriminator
    with torch.no_grad():
        rewards = -torch.log(discriminator(policy_trajs["obs"],
                                           policy_trajs["act"]))

    # 4. Update policy with any RL algorithm (e.g., PPO)
    policy_loss = ppo_update(policy, policy_trajs, rewards)
```

**Practical considerations.**

| Concern | Detail |
|---|---|
| Requires online interaction | Yes — GAIL must collect real rollouts every iteration |
| Expert data efficiency | High — a few dozen trajectories can suffice |
| Reward hacking | Common — the policy finds states where \(D\) is fooled, even if behavior is wrong |
| Stability | Sensitive to discriminator learning rate; gradient penalty helps |
| Scalability | Computationally heavy; not practical for real-robot loops without simulation |

GAIL works well in simulation but rarely used directly on physical hardware today. Its conceptual descendants—reward shaping from VLMs, preference-based learning—avoid the online interaction requirement.

---

## 5.3 Residual RL on Top of IL Policies

A practically popular pattern is to use IL to get most of the way there, then train a *residual* RL policy to handle the remaining precision gap. This avoids the cold-start problem of pure RL while still allowing self-correction.

### 5.3.1 Core Idea

The combined policy is:

\[
a = \pi_\mathrm{IL}(s) + \pi_\mathrm{RL}(s)
\]

The IL policy \(\pi_\mathrm{IL}\) (frozen after pre-training) provides a strong baseline action. The residual policy \(\pi_\mathrm{RL}\) starts near zero and learns small corrections. This decomposition has several advantages:

- **Warm start:** RL exploration begins close to the task-relevant manifold rather than in random space.
- **Safety:** Because \(\pi_\mathrm{RL}\) is initialized near zero, early in training the combined action is almost identical to the IL policy.
- **Precision:** The residual can compensate for the contact forces and small positional offsets that demonstrations cannot fully capture.

The residual RL approach has been applied in high-precision assembly (see ResiP, [Ankile et al., 2024](https://arxiv.org/abs/2407.16677)) with sparse rewards, where pure BC achieves ~30% success and residual RL brings it above 80%.

### 5.3.2 Implementation

```python
import torch
import torch.nn as nn

class ResidualPolicy(nn.Module):
    """
    Combines a frozen IL policy with a trainable residual policy.
    The IL policy provides a warm-start action; residual learns corrections.
    """
    def __init__(self, il_policy: nn.Module, residual_net: nn.Module,
                 residual_scale: float = 0.1):
        super().__init__()
        self.il_policy = il_policy        # frozen — no grad
        self.residual = residual_net      # trained with RL
        self.residual_scale = residual_scale  # clip exploration magnitude

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_action = self.il_policy(obs)   # shape: [B, action_dim]

        delta = self.residual(obs)              # small learned correction
        delta = torch.tanh(delta) * self.residual_scale

        return base_action + delta

    def parameters_to_train(self):
        """Only the residual network is trained by RL."""
        return self.residual.parameters()


# Minimal training loop (PPO or SAC on the residual)
def train_residual(policy: ResidualPolicy, env, rl_algo, n_epochs: int):
    for epoch in range(n_epochs):
        obs, done = env.reset(), False
        while not done:
            action = policy(torch.tensor(obs))
            obs, reward, done, _ = env.step(action.numpy())
            rl_algo.observe(obs, action, reward, done)
        rl_algo.update(policy.parameters_to_train())
```

**Design choices.** The residual scale `residual_scale` should be tuned: too large and the residual overshoots; too small and it cannot correct meaningful errors. A common initialization is \(\approx 10\%\) of the nominal action range.

---

## 5.4 RL Fine-Tuning of IL Policies (Post-2023)

The modern paradigm for large VLA models differs from classical residual RL. Instead of a separate residual head, the entire pre-trained policy is fine-tuned with RL signals—analogous to RLHF in language models.

### 5.4.1 RLHF-Style Fine-Tuning

Human preference feedback can be used to align a robot policy with user intent when that intent is difficult to specify as a reward function.

**RAPL (Tian et al., 2024)** — Representation-Aligned Preference-based Learning — addresses the prohibitive cost of learning visual reward functions from scratch ([Tian et al., 2024](https://arxiv.org/abs/2412.04835)). The key insight is to decouple reward learning into two stages:

1. **Encoder alignment.** Human preference labels (A preferred over B) are used to fine-tune a pre-trained vision encoder so its representation space aligns with the human's notion of task success.
2. **Dense reward via feature matching.** The aligned encoder defines a dense visual reward through feature-space distance: states that look more like the goal in the aligned representation receive higher reward.

Applied to pre-trained Diffusion Policies, RAPL achieves alignment with **5× less human preference data** than traditional RLHF. The visual reward then drives standard RL fine-tuning.

```python
# RAPL reward computation (simplified)
class RAPLReward(nn.Module):
    def __init__(self, aligned_encoder: nn.Module, goal_embedding: torch.Tensor):
        super().__init__()
        self.encoder = aligned_encoder   # fine-tuned on preference data
        self.goal_emb = goal_embedding   # encoded goal image

    def forward(self, obs_image: torch.Tensor) -> torch.Tensor:
        obs_emb = self.encoder(obs_image)
        # Cosine similarity as dense reward
        reward = torch.nn.functional.cosine_similarity(
            obs_emb, self.goal_emb.expand_as(obs_emb), dim=-1
        )
        return reward
```

### 5.4.2 GRPO for VLAs: WMPO

Group Relative Policy Optimization (GRPO) eliminates the need for a separate value network by comparing a *group* of sampled trajectories against each other. For each prompt (observation + instruction), the policy generates \(G\) rollouts; the advantage of rollout \(i\) is:

\[
\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_j\}_{j=1}^G)}{\mathrm{std}(\{r_j\}_{j=1}^G)}
\]

The policy is updated with a clipped surrogate objective:

\[
\mathcal{L}_\mathrm{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^G \min\!\left(\frac{\pi_\theta(a_i|o)}{\pi_{\theta_\mathrm{old}}(a_i|o)} \hat{A}_i,\; \mathrm{clip}\!\left(\frac{\pi_\theta}{\pi_{\theta_\mathrm{old}}}, 1-\varepsilon, 1+\varepsilon\right)\hat{A}_i\right) - \beta \, D_\mathrm{KL}(\pi_\theta \| \pi_\mathrm{ref})
\]

**WMPO (Zhu et al., 2025)** — World Model-based Policy Optimization — applies GRPO to VLAs without requiring real robot interaction ([Zhu et al., 2025](https://arxiv.org/abs/2511.09515)). The key components are:

1. **Pixel-based world model.** A learned model predicts future video frames from current observations and actions. Unlike latent-space world models, pixel-level prediction keeps the "imagined" trajectory aligned with the VLA's image-based feature representations pre-trained on web-scale data.
2. **On-policy GRPO.** The VLA generates groups of action sequences; the world model rolls them forward to produce imagined trajectories; a task reward evaluates the outcomes; GRPO updates the VLA policy.

Results: WMPO substantially improves sample efficiency over off-policy methods, exhibits emergent self-correction behaviors, and demonstrates robust generalization without needing to reset a physical robot thousands of times.

```python
# WMPO training loop (conceptual pseudocode)
def wmpo_update(vla, world_model, reward_fn, obs, instruction, G=8):
    """
    vla:          Vision-Language-Action model (policy to be fine-tuned)
    world_model:  Pixel-based video prediction model
    reward_fn:    Task success evaluator
    G:            Group size (number of rollouts per prompt)
    """
    action_groups = []
    rewards = []

    for _ in range(G):
        # Sample an action sequence from the VLA
        actions = vla.sample(obs, instruction)      # [T, action_dim]

        # Roll out in imagination via pixel-based world model
        imagined_frames = world_model.rollout(obs, actions)  # [T, H, W, C]

        # Evaluate task reward from imagined final frame
        r = reward_fn(imagined_frames[-1], instruction)

        action_groups.append(actions)
        rewards.append(r)

    rewards = torch.tensor(rewards)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # GRPO policy update
    loss = grpo_loss(vla, obs, instruction, action_groups, advantages)
    loss.backward()
```

### 5.4.3 VLA + RL: Emerging Approaches

**MoRE (Zhao et al., 2025)** applies RL fine-tuning to VLAs for quadruped locomotion ([Zhao et al., 2025](https://arxiv.org/abs/2503.08007)). The architecture introduces a *mixture-of-robotic-experts* design: multiple LoRA modules act as distinct experts within a shared MLLM backbone, forming a sparse-activated MoE. The policy is trained as a Q-function rather than a standard behavior clone, enabling effective learning from automatically collected mixed-quality data. MoRE outperforms baselines across six locomotion skills and generalizes to out-of-distribution scenarios.

**Key challenge: sparse rewards in manipulation.** Most manipulation tasks offer a single binary reward at task completion. Strategies to overcome this:

- **Dense reward shaping from VLMs.** Query a VLM at each timestep with an observation image and ask it to rate progress toward the goal. This converts sparse supervision into dense guidance without human reward engineering.
- **Subgoal reward decomposition.** Define intermediate checkpoints (object grasped, object lifted, object placed) and reward each separately.
- **Hindsight relabeling.** Treat reached states as goals, providing reward even for failed trajectories.

---

## 5.5 Hierarchical Imitation Learning

Long-horizon tasks—"clean the kitchen," "assemble the shelf"—consist of sequences of subtasks with natural temporal structure. Hierarchical policies exploit this structure by decomposing decision-making across multiple timescales.

### 5.5.1 Goal-Conditioned IL

The canonical hierarchical decomposition separates planning from execution:

- **High-level policy** \(\pi_\mathrm{hi}(g \mid s)\): predicts a subgoal \(g \in \mathcal{G}\) (e.g., a desired state, object pose, or language description) given the current state.
- **Low-level policy** \(\pi_\mathrm{lo}(a \mid s, g)\): selects primitive actions to achieve \(g\).

The high-level policy operates at a longer timescale (every \(k\) steps), while the low-level policy runs at full control frequency. This **temporal abstraction** reduces the effective horizon for each level:

\[
T_\mathrm{eff} = \frac{T}{k}
\]

Training from demonstrations is done by relabeling. Each trajectory \(\{(s_t, a_t)\}\) is annotated with subgoals by segmenting it into skill segments. The high-level policy is trained on \((s_t, g_{t+k})\) pairs; the low-level policy on \((s_t, g, a_t)\) tuples where \(g\) is the segment endpoint.

```python
class HierarchicalPolicy(nn.Module):
    def __init__(self, hi_policy: nn.Module, lo_policy: nn.Module,
                 subgoal_horizon: int = 10):
        super().__init__()
        self.hi = hi_policy         # predicts subgoals every k steps
        self.lo = lo_policy         # predicts actions given (obs, subgoal)
        self.k = subgoal_horizon
        self._step = 0
        self._current_subgoal = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self._step % self.k == 0:
            # Re-plan at high level
            self._current_subgoal = self.hi(obs)   # e.g., target image or embedding
        self._step += 1

        # Low-level policy conditioned on current subgoal
        return self.lo(obs, self._current_subgoal)

    def reset(self):
        self._step = 0
        self._current_subgoal = None
```

### 5.5.2 The Options Framework

The **options framework** (Sutton, Precup & Singh, 1999) formalizes temporally extended actions. An option \(\omega\) is a triple:

\[
\omega = \left(\mathcal{I}_\omega \subseteq \mathcal{S},\; \pi_\omega : \mathcal{S} \times \mathcal{A} \to [0,1],\; \beta_\omega : \mathcal{S} \to [0,1]\right)
\]

- \(\mathcal{I}_\omega\): **initiation set** — states from which the option can be invoked.
- \(\pi_\omega\): **intra-option policy** — low-level behavior while the option runs.
- \(\beta_\omega\): **termination condition** — probability of terminating at each state.

A high-level policy \(\pi_\Omega : \mathcal{S} \times \Omega \to [0,1]\) selects options; each option runs to termination before the high-level policy re-selects. This yields a Semi-Markov Decision Process at the top level.

**Learning options from demonstrations.** Options can be discovered from demonstrations by segmenting trajectories into reusable skill primitives. Common approaches include:

- **Bottleneck state detection:** states frequently visited across many trajectories are candidate subgoal states that define option boundaries.
- **Option-critic** (Bacon et al., 2017): end-to-end gradient-based learning of intra-option policies and termination conditions simultaneously.

**HiRT (Zhang et al., 2024)** is a practical hierarchical VLA that pairs a VLM at low frequency with a high-frequency transformer policy ([Zhang et al., 2024](https://arxiv.org/abs/2410.05273)). VLMs with billions of parameters introduce latency (often 100–500 ms per inference call) that is incompatible with real-time robot control. HiRT resolves this by running the VLM every \(k\) steps to produce slowly-updated semantic features, while a lightweight high-frequency vision-based policy uses those features to generate real-time actions. Results: HiRT doubles control frequency on static tasks with comparable success, and improves dynamic task success from 48% to 75%.

### 5.5.3 Language as Hierarchy

Natural language provides a semantically rich intermediate representation for hierarchical decomposition. Two prominent patterns:

**Language subgoal prediction (\(\pi_{0.5}\)).** Physical Intelligence's \(\pi_{0.5}\) model uses a two-stage inference procedure: at each control cycle, the model first generates a high-level semantic subtask label (e.g., "pick up the cutting board"), then conditions action generation on that label. Co-training on heterogeneous data—other robots, web video, verbal instructions—enables broad generalization to unseen environments and objects.

**Visual chain-of-thought (CoT-VLA).** Rather than language tokens, CoT-VLA (2025) uses *predicted future image frames* as the intermediate reasoning step ([CoT-VLA, CVPR 2025](https://arxiv.org/html/2503.22020v1)):

\[
\underbrace{\hat{s}_{t+n}}_{\text{subgoal image}} \leftarrow f_\theta(s_t, \ell) \qquad \text{then} \qquad a_{t:t+m} \leftarrow g_\theta(s_t, \hat{s}_{t+n}, \ell)
\]

The model first autoregressively predicts a future frame representing the desired intermediate state, then generates an action chunk to reach it. Pretraining on action-less video data improves downstream performance even on different robot platforms. CoT-VLA achieves a 17% improvement over the prior state of the art on real-world manipulation and 6% in simulation. Crucially, improvements in subgoal image generation quality translate directly to higher task success—a virtuous cycle as generative models improve.

```python
class CoTVLA(nn.Module):
    """
    CoT-VLA: predict a visual subgoal, then generate actions.
    Uses a hybrid attention mechanism: causal for generation,
    full attention for action chunking.
    """
    def __init__(self, backbone, action_head, chunk_size: int = 10):
        super().__init__()
        self.backbone = backbone          # VILA-U or similar unified VLM
        self.action_head = action_head    # generates chunk of actions
        self.chunk_size = chunk_size

    def forward(self, obs: torch.Tensor, instruction: str):
        # Stage 1: Generate subgoal image tokens (visual CoT)
        subgoal_tokens = self.backbone.generate_image(
            obs, instruction, n_frames_ahead=10
        )   # shape: [B, n_image_tokens]

        # Stage 2: Generate action chunk conditioned on obs + subgoal
        action_chunk = self.action_head(
            obs, subgoal_tokens, instruction
        )   # shape: [B, chunk_size, action_dim]

        return action_chunk   # execute all chunk_size actions before re-planning

    @torch.inference_mode()
    def select_action(self, obs: torch.Tensor, instruction: str):
        chunk = self.forward(obs, instruction)
        return chunk[:, 0, :]   # return first action; buffer remainder
```

---

## 5.6 Multi-Task and Task Generalization

A policy that handles only one task is rarely deployable. Multi-task IL allows a single model to handle diverse task distributions, and—when generalization works—unseen tasks.

### 5.6.1 Task Conditioning

The policy is conditioned on a task descriptor \(z\), which can take several forms:

| Task Representation | Method | Pros | Cons |
|---|---|---|---|
| One-hot task ID | Early multi-task BC | Simple, no ambiguity | Cannot generalize to unseen tasks |
| Language embedding (CLIP/USE) | BC-Z, RT-2, OpenVLA | Zero-shot to new language instructions | Encoder quality matters |
| Goal image embedding | Octo, relay policies | Grounded in visual space | Requires a goal image at test time |
| Language + goal image | Octo (both modes) | Flexible specification | Training complexity |

Language conditioning is now the default. Frozen pre-trained CLIP or T5 text encoders provide rich semantic embeddings; importantly, they can generalize to novel language descriptions at test time.

```python
class LanguageConditionedPolicy(nn.Module):
    def __init__(self, obs_encoder, lang_encoder, action_head):
        super().__init__()
        self.obs_enc = obs_encoder
        self.lang_enc = lang_encoder   # e.g., frozen CLIP text encoder
        self.action_head = action_head

    def forward(self, obs: torch.Tensor, task_lang: list[str]) -> torch.Tensor:
        obs_feat = self.obs_enc(obs)                          # [B, d_obs]

        with torch.no_grad():
            lang_feat = self.lang_enc.encode_text(task_lang)  # [B, d_lang]

        fused = torch.cat([obs_feat, lang_feat], dim=-1)
        return self.action_head(fused)
```

### 5.6.2 Transfer Learning in IL

Modern IL follows a two-stage protocol mirroring NLP fine-tuning:

1. **Pre-train** on a large, diverse dataset (e.g., OXE, DROID) to acquire general visual-motor representations.
2. **Fine-tune** on a small, task-specific dataset (10–200 demonstrations) to specialize.

**BC-Z (Jang et al., 2022)** demonstrated zero-shot language task generalization at scale ([Jang et al., 2022](https://proceedings.mlr.press/v164/jang22a/jang22a.pdf)). Trained on 25,877 episodes across 100 diverse manipulation tasks with language and video task conditioning, BC-Z achieves non-zero success on 24 of 29 held-out tasks, averaging 44% success with language conditioning—without any robot demonstrations for those tasks.

**Few-shot IL.** After pre-training on diverse tasks, a model like ACT or Diffusion Policy can be fine-tuned to a new task with 10–50 demonstrations, compared to the 200+ demonstrations required when training from scratch. The pre-trained backbone provides generic visuomotor representations; fine-tuning only adapts the task-specific head or the top layers.

### 5.6.3 Cross-Embodiment Transfer

Robots differ in degrees of freedom, sensor suites, end-effector morphology, and action space conventions. Cross-embodiment transfer attempts to share knowledge across these differences.

**The core challenge.** A joint position of \([0.2, -0.5, \ldots]\) on a Franka Panda carries entirely different physical meaning than the same numerical values on a UR5. Naive concatenation of heterogeneous data leads to conflicting gradients and poor convergence.

**Approaches:**

- **Shared latent space.** Map proprioception from different embodiments to a common embedding space. The backbone learns embodiment-agnostic representations; embodiment-specific heads decode to robot-specific actions.
- **Action tokenization.** Discretize continuous actions into tokens shared across embodiments, treating different robots as speaking the "same language." This is the approach taken by FAST and similar methods.
- **Soft prompting (X-VLA).** X-VLA ([X-VLA, 2025](https://arxiv.org/abs/2510.10274)) assigns a small set of learnable embedding vectors to each data source (robot/camera configuration). These soft prompts absorb embodiment-specific variations—hardware layout, camera angles, action space conventions—while the backbone learns an embodiment-agnostic generalist policy. With only 9M trainable parameters (1% of total), X-VLA-0.9B achieves competitive performance on LIBERO and Simpler-WidowX.

**OXE positive transfer result.** The Open X-Embodiment project (OXE) demonstrated that training RT-1-X on data from 9 different robots yields a 50% higher success rate than each robot-specific model—a clear win from co-training ([Open X-Embodiment, 2023](https://arxiv.org/html/2310.08864v4)). RT-2-X showed ~3× generalization improvement over single-embodiment training. The key lesson: diversity of embodiments functions like data augmentation for the representations, provided the architecture can handle the heterogeneity.

---

## 5.7 Data Augmentation as an Extension

Data augmentation techniques extend the effective coverage of a fixed demonstration dataset without requiring new human effort.

### 5.7.1 DART — Disturbances for Augmenting Robot Trajectories

**The core idea.** Behavioral cloning suffers from covariate shift because the trained policy makes small errors that accumulate, visiting states the demonstrations never cover. DART (Laskey et al., 2017) addresses this by injecting noise into the *supervisor's* actions during data collection, forcing the supervisor to demonstrate recovery from small perturbations ([Laskey et al., 2017](https://arxiv.org/abs/1703.09327)).

Formally, demonstrated actions are perturbed as:

\[
\tilde{a} = a^* + \epsilon, \qquad \epsilon \sim \mathcal{N}(0, \Sigma)
\]

The noise covariance \(\Sigma\) is optimized to approximate the error distribution of the trained policy, making the injected noise *just large enough* to simulate realistic errors without overwhelming the supervisor.

**Results.** On the MuJoCo Humanoid task, DART is up to 3× faster in computation than DAgger and reduces the supervisor's cumulative reward by only 5% during training, while DAgger-induced rollouts achieve 80% less cumulative reward. On a physical grasping-in-clutter task with human supervisors, DART achieves a 62% performance improvement over behavioral cloning.

```python
def collect_dart_demonstrations(supervisor, env, noise_cov, n_demos):
    """
    Collect demonstrations with injected noise (DART protocol).
    noise_cov: covariance matrix for action perturbations (optimized offline)
    """
    dataset = []
    noise_dist = torch.distributions.MultivariateNormal(
        loc=torch.zeros(env.action_dim),
        covariance_matrix=noise_cov
    )

    for _ in range(n_demos):
        obs = env.reset()
        traj = []
        done = False
        while not done:
            expert_action = supervisor.act(obs)
            # Inject calibrated noise
            noise = noise_dist.sample()
            noisy_action = expert_action + noise.numpy()
            # Supervisor corrects from the perturbed state
            obs_next, _, done, _ = env.step(noisy_action)
            traj.append({"obs": obs, "action": noisy_action})
            obs = obs_next
        dataset.extend(traj)
    return dataset
```

**Practical note.** DART is off-policy (noise is injected during data collection, not at policy deployment) and works with human supervisors—unlike DAgger, which requires the human to watch the robot's own policy execute. This makes DART far more practical for teleoperation-based data collection.

### 5.7.2 RoVi-Aug — Robot and Viewpoint Augmentation

Datasets like OXE have imbalanced distributions: some robot embodiments and camera viewpoints are heavily over-represented. Policies trained on such data overfit to the dominant configuration and fail when deployed with a different camera angle or on a different robot arm.

**RoVi-Aug** (Chen et al., 2024) uses image-to-image generative models to synthesize demonstrations with different robot embodiments and camera viewpoints ([Chen et al., 2024](https://arxiv.org/abs/2409.03403)). Given an original demonstration image showing Robot A from View V, the generative model produces a version showing Robot B from View V', while preserving the scene content and object positions.

Key results:
- **Zero-shot cross-embodiment deployment:** A policy trained on RoVi-Aug data deploys on an unseen robot with significantly different camera angles—without test-time adaptation.
- **No test-time overhead:** Unlike Mirage (a test-time adaptation baseline), RoVi-Aug requires no extra processing at inference.
- **Multi-robot co-training:** Training on original + augmented data enables multi-robot, multi-task policies. Success rates improve by up to 30%.

```python
# RoVi-Aug augmentation pipeline (conceptual)
class RoViAugmentation:
    def __init__(self, image_gen_model, target_embodiment_config, target_viewpoint):
        self.gen_model = image_gen_model  # e.g., ControlNet or InstructPix2Pix
        self.target_embodiment = target_embodiment_config
        self.target_viewpoint = target_viewpoint

    def augment(self, original_image: torch.Tensor, scene_mask: torch.Tensor):
        """
        Synthesize image with target robot embodiment and camera viewpoint.
        scene_mask: segmentation of robot vs. background + objects
        """
        prompt = f"Replace robot with {self.target_embodiment}, "
                   f"view from {self.target_viewpoint}, preserve objects"
        augmented = self.gen_model.translate(
            original_image, prompt, mask=scene_mask
        )
        return augmented   # same (s, a) label, new visual appearance
```

---

## 5.8 Safety and Robustness Extensions

Deploying IL policies in the real world requires mechanisms for detecting and avoiding failures, especially as policies are extended via RL or hierarchical structures.

### 5.8.1 Safe DAgger

Standard DAgger accepts the policy's actions whenever possible and queries the expert only at a fixed rate. **SafeDAgger** and **EnsembleDAgger** instead query the expert only when the policy's confidence is sufficiently high, and defer to the expert otherwise.

**EnsembleDAgger** (Menda et al., 2018) approximates policy uncertainty using an ensemble of \(N\) neural networks. Each network \(f_i\) predicts an action; the ensemble disagreement serves as an uncertainty estimate:

\[
\sigma^2_\mathrm{ensemble}(s) = \frac{1}{N} \sum_{i=1}^N \|f_i(s) - \bar{f}(s)\|^2
\]

where \(\bar{f}(s) = \frac{1}{N}\sum_i f_i(s)\). The policy acts autonomously if \(\sigma^2_\mathrm{ensemble}(s) < \chi\) (doubt threshold); otherwise, the expert is queried. This dual rule—low discrepancy *and* low doubt—constrains the probability of failure more tightly than single-threshold approaches.

```python
class EnsembleDAggerPolicy(nn.Module):
    def __init__(self, ensemble: list[nn.Module], doubt_threshold: float = 0.05):
        super().__init__()
        self.ensemble = nn.ModuleList(ensemble)
        self.chi = doubt_threshold

    def forward(self, obs: torch.Tensor):
        actions = torch.stack([net(obs) for net in self.ensemble])  # [N, B, A]
        mean_action = actions.mean(dim=0)                            # [B, A]
        variance = actions.var(dim=0).mean(dim=-1)                  # [B]
        return mean_action, variance

    def should_query_expert(self, obs: torch.Tensor) -> torch.Tensor:
        _, variance = self.forward(obs)
        return variance > self.chi   # True → query expert, False → act autonomously
```

### 5.8.2 Action Filtering with Control Barrier Functions

**Control Barrier Functions (CBFs)** provide formal safety guarantees by defining a safe set \(\mathcal{C} = \{x : h(x) \geq 0\}\) and enforcing a constraint that the system stays within it. A CBF-based safety filter takes the policy's nominal action \(u_\mathrm{nom}\) and solves a Quadratic Program (QP) to find the nearest safe action:

\[
u^* = \underset{u}{\arg\min}\; \|u - u_\mathrm{nom}\|^2 \quad \text{subject to} \quad \dot{h}(x, u) \geq -\alpha(h(x))
\]

where \(\alpha\) is a class-\(\mathcal{K}\) function. This filter can be applied as a post-processing layer on top of any IL or RL policy without retraining.

**PACS** (Path-Consistent Safety for ACT policies) applies this idea to diffusion-based and ACT policies for robot manipulation, filtering generated action chunks through a CBF layer before execution.

```python
import cvxpy as cp
import numpy as np

def cbf_safety_filter(nominal_action: np.ndarray,
                       state: np.ndarray,
                       h_fn, dh_dx_fn,
                       f_fn, g_fn,
                       alpha: float = 1.0) -> np.ndarray:
    """
    QP-based CBF safety filter.
    h_fn:    safety function h(x) ≥ 0 defines safe set
    dh_dx_fn: gradient ∂h/∂x
    f_fn, g_fn: control-affine dynamics ẋ = f(x) + g(x)u
    """
    n_action = len(nominal_action)
    u = cp.Variable(n_action)

    # Objective: stay close to nominal action
    objective = cp.Minimize(cp.sum_squares(u - nominal_action))

    # CBF constraint: dh/dx (f(x) + g(x)u) ≥ -alpha * h(x)
    h_val = h_fn(state)
    dh = dh_dx_fn(state)
    f_val = f_fn(state)
    g_val = g_fn(state)

    constraints = [dh @ (f_val + g_val @ u) >= -alpha * h_val]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)

    return u.value if prob.status == "optimal" else nominal_action
```

**Practical limitations.** CBFs require knowledge of the system dynamics \((f, g)\) and an explicit safety function \(h\). For high-dimensional visual policies, both are difficult to obtain. Current work addresses this with learned CBFs or abstract dynamics models.

### 5.8.3 Failure Detection

Even with safety layers, detecting when a policy is *about to fail* allows the system to stop, backtrack, or request help before irreversible actions are taken.

**SAFE** (2025) introduces a multitask failure detector for VLAs ([SAFE, 2025](https://arxiv.org/abs/2506.09937)). The key observation is that a VLA's internal hidden-state representations are well-separated for successful and failing rollouts—and this separation is *generic across tasks*. SAFE trains a lightweight probe (MLP or LSTM) on top of frozen VLA features:

\[
s_t = \sigma\!\left(f_\mathrm{SAFE}(\mathbf{e}_{0:t})\right) \in [0, 1]
\]

where \(\mathbf{e}_{0:t}\) is the sequence of VLA hidden states and \(s_t\) is the predicted failure probability. The threshold for triggering an alert is calibrated using **conformal prediction**, which provides distribution-free coverage guarantees: the system will correctly flag a failure with at least \(1 - \delta\) probability, regardless of the task distribution.

SAFE is compatible with OpenVLA, \(\pi_0\), and \(\pi_0\)-FAST and generalizes to unseen tasks without per-task recalibration.

```python
class SAFEFailureDetector(nn.Module):
    """
    Lightweight failure detector trained on VLA internal features.
    Uses conformal prediction for threshold calibration.
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, vla_features: torch.Tensor) -> torch.Tensor:
        """
        vla_features: [B, T, feature_dim] — hidden states from VLA final layer
        returns: [B] failure probability scores
        """
        lstm_out, _ = self.lstm(vla_features)
        score = self.head(lstm_out[:, -1, :]).squeeze(-1)
        return score


def calibrate_threshold(detector, calibration_data, target_fpr: float = 0.1):
    """
    Conformal prediction threshold calibration.
    Guarantees failure detection recall ≥ 1 - target_fpr on unseen tasks.
    """
    scores = []
    labels = []  # 1 = failure, 0 = success
    for features, label in calibration_data:
        score = detector(features).item()
        scores.append(score)
        labels.append(label)

    # Find threshold s.t. FPR ≤ target_fpr on calibration set
    scores = torch.tensor(scores)
    labels = torch.tensor(labels)
    failure_scores = scores[labels == 1]
    threshold = torch.quantile(failure_scores, target_fpr)
    return threshold.item()
```

---

## 5.9 Summary: The IL + RL Spectrum

The methods in this chapter span a spectrum from fully offline IL to fully online RL. The table below summarizes the key tradeoffs to guide method selection.

| Method | Core Idea | Requires Online Interaction | Sample Efficiency | Key Limitation |
|---|---|---|---|---|
| Behavioral Cloning (baseline) | Supervised learning on demonstrations | No | High | Covariate shift; bounded by demo quality |
| DART | Noise-injected demonstrations | No (offline) | High | Cannot discover truly new behaviors |
| DAgger / EnsembleDAgger | Iterative expert queries | Yes (expert queries) | Moderate | Expert must be available at training time |
| MaxEnt IRL | Learn reward from demos, then RL | Yes (for RL inner loop) | Low | Expensive bi-level optimization |
| GAIL | Adversarial reward from discriminator | Yes (real rollouts) | Moderate | Reward hacking; sensitive to discriminator |
| Residual RL | RL corrections on top of frozen IL | Yes (real/sim rollouts) | Moderate-High | Requires good IL warm start |
| RAPL / RLHF | Visual rewards from human preferences | Partial (human labels) | High | Human labeling cost |
| WMPO / GRPO | On-policy RL via world model | No (imagined rollouts) | High | World model fidelity limits performance |
| Goal-conditioned IL | Hierarchical subgoal decomposition | No | High | Requires segmentation of demonstrations |
| HiRT | VLM planning + fast transformer execution | No (training); Yes (inference) | High | VLM latency at planning level |
| CoT-VLA | Visual chain-of-thought (subgoal images) | No | High | Subgoal generation quality limits policy |
| RoVi-Aug | Generative data augmentation | No | High | Generative model quality and cost |
| Safe DAgger | Uncertainty-gated expert queries | Yes (expert queries) | Moderate | Ensemble disagreement may not track true risk |
| CBF Filtering | Post-hoc action projection to safe set | No | N/A | Requires dynamics model and safety function |
| SAFE | VLA feature probing for failure detection | No | N/A | Trained on seen tasks; some distribution shift |
| MoRE | RL fine-tuning of MoE-VLA as Q-function | Yes (mixed-quality data) | High | Q-function training stability |
| Cross-embodiment (X-VLA) | Soft prompts per embodiment | No | High | Prompt design; prompt warm-up required |

### Choosing a Method in Practice

Follow this decision tree:

1. **Do you have access to a simulator that matches your real environment?** → Yes: WMPO or residual RL in sim, then transfer. No: go to 2.
2. **Can you afford real-robot RL interactions?** → Yes: GAIL or residual RL. No: go to 3.
3. **Can you collect additional human demonstrations?** → Yes: DART or DAgger. No: go to 4.
4. **Do you have a large pre-trained VLA?** → Yes: RLHF/RAPL for preference alignment, or CoT-VLA/HiRT for long-horizon tasks. No: BC with data augmentation.

For safety requirements, layer EnsembleDAgger (during training) and CBF filtering + SAFE (during deployment) on top of any of the above.

---

## Chapter Notes

**MaxEnt IRL.** Ziebart, B. D., Maas, A., Bagnell, J. A., & Dey, A. K. (2008). Maximum entropy inverse reinforcement learning. *AAAI*.

**GAIL.** Ho, J., & Ermon, S. (2016). Generative adversarial imitation learning. *NeurIPS*. [arXiv:1606.03476](https://arxiv.org/abs/1606.03476)

**DART.** Laskey, M., Lee, J., Fox, R., Dragan, A., & Goldberg, K. (2017). DART: Noise injection for robust imitation learning. *CoRL*. [arXiv:1703.09327](https://arxiv.org/abs/1703.09327)

**EnsembleDAgger.** Menda, K., Driggs-Campbell, K., & Kochenderfer, M. J. (2018). EnsembleDAgger: A Bayesian approach to safe imitation learning. *IROS*. [arXiv:1807.08364](https://arxiv.org/abs/1807.08364)

**BC-Z.** Jang, E., et al. (2022). BC-Z: Zero-shot task generalization with robotic imitation learning. *CoRL*. [proceedings.mlr.press](https://proceedings.mlr.press/v164/jang22a/jang22a.pdf)

**OXE / RT-X.** Open X-Embodiment Collaboration (2023). Open X-Embodiment: Robotic learning datasets and RT-X models. [arXiv:2310.08864](https://arxiv.org/html/2310.08864v4)

**Options Framework.** Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. *Artificial Intelligence*, 112(1-2), 181-211.

**RAPL.** Tian, R., Wu, Y., Xu, C., Tomizuka, M., Malik, J., & Bajcsy, A. (2024). Maximizing alignment with minimal feedback. [arXiv:2412.04835](https://arxiv.org/abs/2412.04835)

**HiRT.** Zhang, J., Guo, Y., Chen, X., Wang, Y., Hu, Y., & Shi, J. (2024). HiRT: Enhancing robotic control with hierarchical robot transformers. *CoRL*. [arXiv:2410.05273](https://arxiv.org/abs/2410.05273)

**RoVi-Aug.** Chen, L. Y., et al. (2024). RoVi-Aug: Robot and viewpoint augmentation for cross-embodiment robot learning. *CoRL Oral*. [arXiv:2409.03403](https://arxiv.org/abs/2409.03403)

**MoRE.** Zhao, H., et al. (2025). MoRE: Unlocking scalability in reinforcement learning for quadruped vision-language-action models. *ICRA*. [arXiv:2503.08007](https://arxiv.org/abs/2503.08007)

**WMPO.** Zhu, F., et al. (2025). WMPO: World model-based policy optimization for vision-language-action models. [arXiv:2511.09515](https://arxiv.org/abs/2511.09515)

**CoT-VLA.** (2025). CoT-VLA: Visual chain-of-thought reasoning for vision-language-action models. *CVPR*. [arXiv:2503.22020](https://arxiv.org/html/2503.22020v1)

**X-VLA.** (2025). X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-language-action model. *ICLR*. [arXiv:2510.10274](https://arxiv.org/abs/2510.10274)

**SAFE.** (2025). SAFE: Multitask failure detection for vision-language-action models. [arXiv:2506.09937](https://arxiv.org/abs/2506.09937)

**ResiP.** Ankile, L., et al. (2024). From imitation to refinement: Residual RL for precise assembly. [arXiv:2407.16677](https://arxiv.org/abs/2407.16677)
