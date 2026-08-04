# Chapter 2: Models — ACT, VLAs, and Related Architectures

---

## 1. Behavioral Cloning Baseline

### 1.1 Standard Formulation

The simplest approach to imitation learning is behavioral cloning (BC): treat the expert demonstration dataset as supervised data and train a policy $\pi_\theta$ to maximize the likelihood of observed actions given observations. The training objective is:

$$
\mathcal{L}_{BC} = \mathbb{E}_{(s,a) \sim \mathcal{D}} \left[-\log \pi_\theta(a \mid s)\right]
$$

where $\mathcal{D} = \{(s_i, a_i)\}_{i=1}^{N}$ is the demonstration dataset collected from an expert policy $\pi^*$. For continuous action spaces, $\pi_\theta$ is typically a Gaussian policy; for discrete spaces, a softmax classifier. The loss reduces to mean squared error (MSE) when the policy outputs a deterministic point estimate.

A standard CNN or ResNet encodes image observations, a shallow MLP or lightweight transformer serves as the policy head, and training proceeds by standard SGD/Adam. This setup is operationally straightforward and surprisingly competitive on short-horizon tasks with near-i.i.d. test conditions.

### 1.2 Why Naive BC Fails at Precision Tasks

Despite its simplicity, vanilla BC has well-documented failure modes that become severe for precision manipulation:

**Compounding errors (distribution shift).** BC trains on the marginal state distribution $d^{\pi^*}(s)$ induced by the expert. At test time, the learner's own policy $\pi_\theta$ is executed, inducing a different distribution $d^{\pi_\theta}(s)$. Even tiny per-step errors push the agent into states not covered by $\mathcal{D}$, where the policy has no training signal. Ross et al. (2011, DAgger) showed that the cumulative error grows as $O(T^2 \epsilon)$ for a horizon-$T$ task with per-step error $\epsilon$, compared to $O(T \epsilon)$ for an interactive learner. For long-horizon dexterous tasks, this quadratic blowup is catastrophic.

**Non-stationarity of demonstrations.** Human demonstrations exhibit temporal correlations: a human expert implicitly conditions each action on a mental model of what they just did and plan to do next. Treating $(s_t, a_t)$ pairs as i.i.d. ignores this — the learned policy sees individual frames rather than the coherent intention behind a motion.

**Multimodality.** When a task admits multiple valid strategies (e.g., grasping an object from the left or the right), a unimodal Gaussian policy outputs the average of both modes, which corresponds to neither. This causes the policy to "average" over demonstrations in action space, producing physically invalid actions.

**High-frequency oscillation.** At 50 Hz control, predicting a single action at each step means any small noise in the policy's output directly drives the robot. Without temporal smoothing, even well-trained policies produce jerky, unstable trajectories.

The architectures in the remainder of this chapter each address one or more of these failure modes.

---

## 2. ACT: Action Chunking with Transformers

> **Reference:** Zhao et al., 2023 — [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)

### 2.1 Motivation: Action Chunking

The central idea in ACT is **action chunking**: instead of predicting a single action $a_t \in \mathbb{R}^{14}$ at each step, predict a contiguous sequence of $k$ actions $a_{t:t+k} \in \mathbb{R}^{k \times 14}$ and execute them open-loop before querying the policy again.

This has two complementary benefits:

**Reduced effective horizon.** If the true task horizon is $T$ and the policy predicts chunks of length $k$, the number of policy queries is $T/k$. Because compounding errors accumulate with each policy query, chunking reduces the effective error accumulation to $O((T/k)^2 \epsilon)$ under the DAgger error model. For $k=100$ at 50 Hz (2 seconds of open-loop execution), this is a 100× reduction in compounding error opportunities.

**Temporal coherence.** Predicting an entire motion segment forces the model to commit to a coherent motion plan. Individual steps within the chunk are consistent with one another by construction, eliminating the high-frequency oscillation problem.

The practice of predicting multi-step action sequences has a psychological basis in work on human motor chunking (Lai et al., 2022), which shows that humans decompose complex movements into pre-planned chunks rather than selecting individual muscle activations at each millisecond.

### 2.2 Architecture: CVAE + Transformer

A standard BC model predicts a single action given an observation. To predict an action chunk $a_{t:t+k}$, two design challenges arise: (1) the output is high-dimensional and temporally structured, and (2) demonstrations of the same task may follow qualitatively different motion strategies (multimodality). ACT addresses both with a **Conditional Variational Autoencoder (CVAE)** wrapped around a transformer backbone.

The CVAE introduces a latent style variable $z \in \mathbb{R}^{32}$ that captures the "style" or "intention" of the demonstrated motion segment (e.g., approaching from the left versus the right). At training time, $z$ is inferred from the action chunk and observation via an encoder network. At test time, $z$ is fixed to zero (the prior mean), acting as a regularizer that encourages the policy to produce the most common demonstration style.

#### 2.2.1 CVAE Encoder (Inference Network)

The encoder approximates the posterior:

$$
q_\phi(z \mid a_{t:t+k}, \bar{o}_t)
$$

**Inputs:**
- Joint positions $\bar{o}_t \in \mathbb{R}^{14}$ (7 DOF per arm, both arms).
- Action sequence $a_{t:t+k} \in \mathbb{R}^{k \times 14}$.

**Architecture:** A BERT-like transformer encoder with 4 self-attention layers. The input tokens are constructed as follows:

1. A learnable `[CLS]` token is prepended.
2. The joint position observation $\bar{o}_t \in \mathbb{R}^{14}$ is projected to $\mathbb{R}^{512}$ and appended as a single token.
3. Each of the $k$ action vectors $a_{t+i} \in \mathbb{R}^{14}$, $i = 0, \ldots, k-1$, is projected to $\mathbb{R}^{512}$ and appended, yielding $k$ tokens.
4. 1D sinusoidal positional embeddings are added to the sequence.

Total encoder input: $(1 + 1 + k) \times 512$ tokens.

**Output:** The hidden state at the `[CLS]` position is passed through two separate linear heads to produce:
- Mean: $\mu \in \mathbb{R}^{32}$
- Log-variance: $\log \sigma^2 \in \mathbb{R}^{32}$

The posterior is a diagonal Gaussian: $q_\phi(z \mid a_{t:t+k}, \bar{o}_t) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$.

#### 2.2.2 CVAE Decoder (Policy Network)

The decoder is the actual deployable policy. It takes as input the visual observations, joint positions, and the style variable $z$, and outputs the predicted action chunk $\hat{a}_{t:t+k}$.

**Vision backbone.** Four RGB cameras capture the scene at resolution $480 \times 640$. Each image is processed independently by a **ResNet-18** backbone (pretrained on ImageNet), which outputs a spatial feature map of shape $H' \times W' \times 512$ (with stride-32 downsampling, this is approximately $15 \times 20 = 300$ spatial locations). Each spatial location becomes a token in $\mathbb{R}^{512}$, yielding $300$ tokens per camera. **2D sinusoidal position embeddings** are added to encode spatial layout within each camera.

Total visual tokens: $4 \times 300 = 1200$ tokens in $\mathbb{R}^{512}$.

**Joint position encoding.** The 14D joint position vector is projected to $\mathbb{R}^{512}$ via a linear layer, contributing 1 token.

**Style variable encoding.** The 32D style variable $z$ (either sampled from $q_\phi$ during training, or set to $\mathbf{0}$ at test time) is projected to $\mathbb{R}^{512}$ via a linear layer, contributing 1 token.

**Total decoder input to transformer encoder:**

$$
(1200 + 1 + 1) = 1202 \text{ tokens}, \quad \text{each in } \mathbb{R}^{512}
$$

**Transformer encoder.** A standard transformer encoder with:
- 4 self-attention layers
- 8 attention heads
- Hidden dimension 512
- FFN intermediate dimension 3200

All 1202 tokens attend to one another, building a rich joint representation of visual context, proprioception, and motion style.

**Transformer decoder.** A standard transformer decoder with:
- 7 cross-attention layers
- 8 attention heads
- $k$ learnable positional queries $Q \in \mathbb{R}^{k \times 512}$, one per output action step

Each query cross-attends to the encoder's 1202-token output, and the decoder applies self-attention across queries within each layer. The output is $k$ hidden states in $\mathbb{R}^{512}$.

**Output head.** A two-layer MLP projects each of the $k$ hidden states to $\mathbb{R}^{14}$, producing the predicted chunk:

$$
\hat{a}_{t:t+k} \in \mathbb{R}^{k \times 14}
$$

These are **absolute joint positions** (not deltas) for both arms.

**Reconstruction loss.** The paper evaluates both MSE and L1 reconstruction losses and finds **L1 to perform better**, possibly because L1 is less sensitive to large outliers in demonstration data and encourages sparse, decisive motion predictions.

The full decoder architecture is summarized below:

```
RGB Cameras (×4)
    └─► ResNet-18 ─► 300 × 512 spatial tokens (+ 2D sinpos embeds) × 4
                                                              │
Joint positions (14D) ─► Linear ─► 1 × 512 token            │
                                                              │ cat
Style variable z (32D) ─► Linear ─► 1 × 512 token            │
                                                              ▼
                                              1202 × 512 token sequence
                                                              │
                                              Transformer Encoder (4 layers, 8 heads)
                                                              │
                                                       1202 × 512
                                                              │
                                    k learned positional queries ─► Transformer Decoder (7 layers)
                                                              │
                                                        k × 512
                                                              │
                                                    MLP output head
                                                              │
                                                        k × 14
```

### 2.3 Training Objective

The CVAE objective is the standard evidence lower bound (ELBO), combining a reconstruction term and a KL regularization term:

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \, \mathcal{L}_{\text{reg}}
$$

$$
\mathcal{L}_{\text{recon}} = \mathbb{E}_{z \sim q_\phi} \left[ \| \hat{a}_{t:t+k} - a_{t:t+k} \|_1 \right]
$$

$$
\mathcal{L}_{\text{reg}} = D_{\text{KL}}\!\left( q_\phi(z \mid a_{t:t+k}, \bar{o}_t) \,\Big\|\, \mathcal{N}(0, I) \right)
$$

The KL term has a closed form for diagonal Gaussians:

$$
D_{\text{KL}} = \frac{1}{2} \sum_{j=1}^{32} \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)
$$

The weight $\beta = 10$ was found through ablation; it is large enough to encourage the posterior to stay close to the prior (ensuring $z = \mathbf{0}$ at test time is a reasonable action mode) without collapsing the latent space.

> **Note on the training/inference gap.** At training time, $z \sim q_\phi(\cdot \mid a_{t:t+k}, \bar{o}_t)$ — the encoder has access to the future action chunk. At inference time, the encoder is discarded entirely and $z = \mathbf{0}$ (the prior mean). This is intentional: the CVAE framework ensures that $z = \mathbf{0}$ is a valid, high-density point of the prior, so the policy conditioned on $z = \mathbf{0}$ produces a reasonable "average" behavior. The KL term enforces this by penalizing posteriors that deviate far from $\mathcal{N}(0, I)$.

### 2.4 Temporal Ensembling

Action chunking introduces a tension: if the policy is queried only every $k$ steps, it cannot react to unexpected perturbations within a chunk. Executing a full 100-step open-loop chunk at 50 Hz means 2 seconds of blind execution — far too long for contact-rich manipulation.

ACT resolves this with **temporal ensembling**: the policy is queried at every timestep $t$, but each query produces a chunk $\hat{a}_{t:t+k}$. At time $t$, multiple overlapping predictions are available:
- Query at step $t$: predictions for $t, t+1, \ldots, t+k-1$
- Query at step $t-1$: predictions for $t, t+1, \ldots, t+k-2$
- ...
- Query at step $t-j$: prediction for $t$ (the $(j+1)$-th element of that chunk)

Denoting $A_t[i]$ as the prediction for time $t$ made by the query at $t - i$, the executed action is the weighted average:

$$
a_t = \frac{\sum_{i=0}^{\min(t,k-1)} w_i \, A_t[i]}{\sum_{i=0}^{\min(t,k-1)} w_i}, \qquad w_i = \exp(-m \cdot i)
$$

where $m$ controls recency weighting. With $m = 0$, all predictions receive equal weight (pure averaging). With $m \to \infty$, only the most recent chunk is used (pure chunking with no ensembling). The default $m = 0.01$ creates a gentle exponential decay that weights recent predictions more but still smooths over the older ones, providing responsiveness while suppressing jerky transitions.

In practice, a FIFO ring buffer of size $k$ is maintained for each joint dimension, with incoming predictions inserted at the front and the weighted mean computed at each step.

### 2.5 PyTorch Implementation

Below is a realistic pseudocode implementation of ACT, capturing the key data flow:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class CVAEEncoder(nn.Module):
    """BERT-like encoder that infers z from (action chunk, joint positions)."""

    def __init__(self, action_dim=14, chunk_size=100, hidden_dim=512,
                 latent_dim=32, num_heads=8, num_layers=4):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.obs_proj = nn.Linear(action_dim, hidden_dim)  # joint pos → token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Embedding(chunk_size + 2, hidden_dim)  # CLS + obs + k actions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=3200, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, actions, joint_pos):
        # actions: (B, k, 14), joint_pos: (B, 14)
        B, k, _ = actions.shape

        # Project inputs to hidden_dim
        act_tokens = self.action_proj(actions)           # (B, k, 512)
        obs_token = self.obs_proj(joint_pos).unsqueeze(1)  # (B, 1, 512)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, 512)

        # Concatenate: [CLS, obs, a_t, ..., a_{t+k-1}]
        tokens = torch.cat([cls, obs_token, act_tokens], dim=1)  # (B, k+2, 512)
        pos_ids = torch.arange(k + 2, device=actions.device)
        tokens = tokens + self.pos_embed(pos_ids).unsqueeze(0)

        out = self.transformer(tokens)   # (B, k+2, 512)
        cls_out = out[:, 0, :]           # (B, 512)
        return self.mu_head(cls_out), self.logvar_head(cls_out)


class ACTDecoder(nn.Module):
    """Transformer encoder-decoder that maps (visual tokens, z) → action chunk."""

    def __init__(self, chunk_size=100, hidden_dim=512, action_dim=14,
                 num_heads=8, enc_layers=4, dec_layers=7):
        super().__init__()
        # Vision backbone (shared weights across cameras)
        backbone = resnet18(pretrained=True)
        self.vision_backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.vision_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

        # Proprioception and style projections
        self.joint_proj = nn.Linear(action_dim, hidden_dim)
        self.z_proj = nn.Linear(32, hidden_dim)

        # 2D sinusoidal positional embedding (applied per-camera)
        self.register_buffer('cam_pos_embed', self._build_2d_sinpos(15, 20, hidden_dim))

        # Transformer backbone
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=3200, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=3200, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        # Learnable action queries (one per step in chunk)
        self.action_queries = nn.Embedding(chunk_size, hidden_dim)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def _build_2d_sinpos(self, H, W, d_model):
        """Build 2D sinusoidal position embeddings of shape (H*W, d_model)."""
        y_pos = torch.arange(H).float().unsqueeze(1).expand(H, W)
        x_pos = torch.arange(W).float().unsqueeze(0).expand(H, W)
        pe = torch.zeros(H * W, d_model)
        for i in range(d_model // 4):
            freq = 1.0 / (10000 ** (4 * i / d_model))
            pe[:, 4*i]     = torch.sin(x_pos.reshape(-1) * freq)
            pe[:, 4*i + 1] = torch.cos(x_pos.reshape(-1) * freq)
            pe[:, 4*i + 2] = torch.sin(y_pos.reshape(-1) * freq)
            pe[:, 4*i + 3] = torch.cos(y_pos.reshape(-1) * freq)
        return pe  # (300, 512)

    def forward(self, obs_images, joint_pos, z):
        # obs_images: (B, 4, 3, 480, 640)
        # joint_pos:  (B, 14)
        # z:          (B, 32)
        B = obs_images.shape[0]

        # Extract visual tokens from all cameras
        cam_tokens_list = []
        for cam_idx in range(4):
            feat = self.vision_backbone(obs_images[:, cam_idx])  # (B, 512, 15, 20)
            feat = self.vision_proj(feat)                         # (B, 512, 15, 20)
            feat = feat.flatten(2).permute(0, 2, 1)              # (B, 300, 512)
            feat = feat + self.cam_pos_embed.unsqueeze(0)        # add 2D sinpos
            cam_tokens_list.append(feat)
        visual_tokens = torch.cat(cam_tokens_list, dim=1)  # (B, 1200, 512)

        # Proprioception and style tokens
        joint_token = self.joint_proj(joint_pos).unsqueeze(1)   # (B, 1, 512)
        z_token = self.z_proj(z).unsqueeze(1)                   # (B, 1, 512)

        # Full encoder input: [visual (1200), joint (1), z (1)]
        enc_input = torch.cat([visual_tokens, joint_token, z_token], dim=1)  # (B, 1202, 512)
        memory = self.encoder(enc_input)  # (B, 1202, 512)

        # Action queries → decoder
        query_ids = torch.arange(self.action_queries.num_embeddings, device=z.device)
        queries = self.action_queries(query_ids).unsqueeze(0).expand(B, -1, -1)  # (B, k, 512)
        dec_out = self.decoder(queries, memory)  # (B, k, 512)

        return self.output_head(dec_out)  # (B, k, 14)


class ACTPolicy(nn.Module):
    """
    Full ACT policy combining CVAE encoder (training only) and transformer decoder.
    At inference, z is fixed to zeros (prior mean).
    """

    def __init__(self, chunk_size=100, hidden_dim=512, action_dim=14,
                 latent_dim=32, beta=10.0):
        super().__init__()
        self.chunk_size = chunk_size
        self.beta = beta
        self.encoder = CVAEEncoder(
            action_dim=action_dim, chunk_size=chunk_size,
            hidden_dim=hidden_dim, latent_dim=latent_dim
        )
        self.decoder = ACTDecoder(
            chunk_size=chunk_size, hidden_dim=hidden_dim, action_dim=action_dim
        )
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs_images, joint_pos, actions=None):
        """
        Training: actions != None → encode z from (actions, joint_pos),
                  compute ELBO loss.
        Inference: actions is None → z = 0, return predicted chunk.
        """
        B = obs_images.shape[0]

        if actions is not None:
            # ---- Training mode ----
            mu, logvar = self.encoder(actions, joint_pos)
            z = self.reparameterize(mu, logvar)

            pred_actions = self.decoder(obs_images, joint_pos, z)  # (B, k, 14)

            # Reconstruction loss (L1)
            l_recon = F.l1_loss(pred_actions, actions)

            # KL divergence loss
            l_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            loss = l_recon + self.beta * l_kl
            return loss, {"l_recon": l_recon.item(), "l_kl": l_kl.item()}

        else:
            # ---- Inference mode ----
            z = torch.zeros(B, self.latent_dim, device=obs_images.device)
            with torch.no_grad():
                pred_actions = self.decoder(obs_images, joint_pos, z)
            return pred_actions  # (B, k, 14)


class TemporalEnsembler:
    """
    Maintains overlapping chunk predictions and computes exponentially
    weighted average for smooth, reactive execution.
    """

    def __init__(self, chunk_size=100, action_dim=14, m=0.01):
        self.k = chunk_size
        self.m = m
        self.action_dim = action_dim
        # weights[i] = exp(-m * i), i=0 most recent
        self.weights = torch.tensor(
            [torch.exp(torch.tensor(-m * i)) for i in range(chunk_size)]
        )
        # Buffer: list of (chunk, start_time) tuples
        self.buffer = []

    def add_chunk(self, chunk, t):
        """Register a new predicted chunk produced at timestep t."""
        self.buffer.append((chunk, t))
        # Discard chunks that are fully expired
        self.buffer = [(c, s) for c, s in self.buffer if t - s < self.k]

    def get_action(self, t):
        """Compute weighted average of all predictions for timestep t."""
        weighted_sum = torch.zeros(self.action_dim)
        weight_total = 0.0
        for chunk, start in self.buffer:
            offset = t - start
            if 0 <= offset < self.k:
                i = offset  # age of this prediction for time t
                w = self.weights[i].item()
                weighted_sum += w * chunk[offset]
                weight_total += w
        return weighted_sum / weight_total if weight_total > 0 else torch.zeros(self.action_dim)
```

### 2.6 Hyperparameters and Results

| Hyperparameter | Value |
|---|---|
| Learning rate | $1 \times 10^{-5}$ |
| Optimizer | AdamW |
| Batch size | 8 |
| Chunk size $k$ | 100 |
| KL weight $\beta$ | 10 |
| Hidden dimension | 512 |
| FFN dimension | 3200 |
| Encoder layers | 4 |
| Decoder layers | 7 |
| Attention heads | 8 |
| Latent dimension | 32 |
| Total parameters | ~80M |
| Training hardware | 1× NVIDIA RTX 2080 Ti |
| Training time | ~5 hours |
| Inference rate | 50 Hz |
| Temporal ensemble weight $m$ | 0.01 |

**Results.** ACT was evaluated on six real-world bimanual manipulation tasks using a dual-arm ALOHA robot: opening a slot car charging station lid, picking and inserting a battery, assembling a phone stand, inserting a 3.5mm audio jack, threading velcro straps, and slotting cups into a mug rack. These tasks require sub-centimeter precision, two-handed coordination, and contact sensing. ACT achieved **80–90% success rates** on all six tasks, substantially outperforming BC and LSTM-based baselines. The ablation study confirmed that both action chunking and temporal ensembling contribute significantly — removing either degrades performance by 20–40 percentage points on contact-critical phases.

---

## 3. Diffusion Policy

> **Reference:** Chi et al., 2023 — [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)

### 3.1 DDPM Formulation

Diffusion Policy frames action generation as a denoising diffusion probabilistic model (DDPM). Rather than regressing to a deterministic action, the model learns to reverse a Markov noise process that gradually corrupts clean action sequences into Gaussian noise.

**Forward process.** A clean action $x_0 \in \mathbb{R}^{T_p \times D}$ is corrupted over $K$ diffusion steps by adding Gaussian noise:

$$
q(x_k \mid x_0) = \mathcal{N}\!\left(x_k;\; \sqrt{\bar\alpha_k}\, x_0,\; (1 - \bar\alpha_k) I\right)
$$

where $\bar\alpha_k = \prod_{i=1}^k \alpha_i$ and $\{\alpha_i\}$ is a cosine noise schedule. As $k \to K$, $x_k \to \mathcal{N}(0, I)$.

**Reverse process (inference).** Starting from $x_K \sim \mathcal{N}(0, I)$, the model iteratively denoises:

$$
x_{k-1} = \frac{1}{\sqrt{\alpha_k}} \left( x_k - \frac{1 - \alpha_k}{\sqrt{1 - \bar\alpha_k}} \, \epsilon_\theta(x_k, k, o_t) \right) + \sigma_k z, \quad z \sim \mathcal{N}(0, I)
$$

where $\epsilon_\theta(x_k, k, o_t)$ is the learned noise-prediction network conditioned on the current observation $o_t$.

**Training objective.** The model is trained to predict the noise:

$$
\mathcal{L} = \mathbb{E}_{x_0, k, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta\!\left( \sqrt{\bar\alpha_k}\, x_0 + \sqrt{1 - \bar\alpha_k}\, \epsilon,\; k,\; o_t \right) \right\|^2 \right]
$$

This is equivalent to maximizing the ELBO of the data likelihood and admits a simple, stable MSE regression objective.

### 3.2 CNN-Based vs. Transformer-Based Variants

The noise-prediction network $\epsilon_\theta$ can be implemented in two ways:

**CNN (U-Net) variant.** A temporal 1D U-Net processes the noisy action sequence along the time axis, with observation features $o_t$ injected via FiLM conditioning at each resolution level. This is efficient and works well when the observation is relatively low-dimensional or encoded separately.

**Transformer variant.** Observation tokens and noisy action tokens are concatenated and processed by a standard transformer encoder (similar to ACT's decoder backbone). The transformer variant generalizes better to complex, high-dimensional observations (multiple cameras) at the cost of higher compute.

**Receding horizon control.** Rather than executing the full predicted sequence, Diffusion Policy uses a receding horizon strategy: predict a sequence of $T_p$ actions but execute only the first $T_a < T_p$ before re-querying. This balances the smoothness of long predictions with reactivity. Typical values: $T_p = 16$, $T_a = 8$.

### 3.3 Key Advantages over BC and ACT

**Multimodal distributions without CVAE.** Because the generative process starts from random noise, Diffusion Policy can naturally represent multimodal action distributions — different denoising trajectories from different random seeds converge to different action modes. No explicit latent variable inference is needed at test time.

**Arbitrary action-space dimensionality.** The DDPM objective is dimension-agnostic; adding more action dimensions simply increases the size of $x_0$. No architectural changes are needed.

**Training stability.** The regression-to-noise objective is a simple MSE loss with a well-understood gradient signal. Unlike the CVAE, there is no KL term to balance, no posterior collapse risk, and no training/inference gap.

**Smooth trajectories.** The multi-step denoising process acts as an implicit smoother: high-frequency noise is averaged out across denoising steps, producing actions that are inherently more temporally consistent than single-step regression.

The trade-off is inference latency: a full DDPM reverse pass requires $K$ (typically 100) network evaluations. DDIM and consistency distillation reduce this to 10–20 steps with minimal quality loss, making real-time deployment feasible.

---

## 4. Vision-Language-Action (VLA) Models

The models in Section 2 and 3 learn task-specific policies from demonstration data. VLA models scale this paradigm in two dimensions: (1) using large pre-trained vision-language models (VLMs) as backbones to leverage internet-scale visual and linguistic knowledge, and (2) training on massive multi-task, multi-robot datasets to enable generalization and language-conditioned behavior.

### 4.1 RT-1 (Brohan et al., 2022)

> **Reference:** [arXiv:2212.06817](https://arxiv.org/abs/2212.06817)

RT-1 is the first large-scale robot transformer to demonstrate that scale and data diversity lead to better generalization, not just better average performance.

**Architecture.** The pipeline is:

1. **Language encoding:** Natural language task instructions (e.g., "pick up the apple") are encoded using the Universal Sentence Encoder (USE), producing a fixed 512-dimensional embedding.

2. **Vision backbone:** Six RGB image frames (history of observations) are processed by an **EfficientNet-B3** feature extractor. Language conditioning is applied via **FiLM (Feature-wise Linear Modulation)**: at each EfficientNet block, the USE embedding is used to produce per-channel scale and shift parameters that modulate intermediate feature maps. This allows language to gate visual attention (e.g., suppressing irrelevant objects).

3. **Token compression:** The spatial feature maps output by EfficientNet-B3 contain many tokens. A **TokenLearner** module compresses them from 81 spatial tokens to 8 learned tokens per frame, dramatically reducing sequence length while preserving task-relevant information.

4. **Causal transformer:** The 8 tokens from each of 6 frames (48 total) are processed by an **11-layer causal transformer decoder** that attends causally over the token sequence.

5. **Discrete action head:** The 11 action dimensions (7 DOF arm + 3 base + 1 gripper) are each discretized into 256 bins. The transformer produces logits over 256 classes per dimension, and actions are sampled or taken as argmax. An additional "mode" dimension distinguishes robot-control tokens from "terminate" tokens.

**Training data:** 130,000 demonstration episodes across 700+ tasks using 13 robots of the same embodiment (RT-1 robot). Data was collected over 17 months in a real-world office-kitchen environment.

**Scale and results.** RT-1 has **~35M parameters**. On over 700 tasks, it achieves 97% success rate in the training distribution (vs. 76% for BC baselines) and generalizes significantly better to unseen tasks, environments, and distractor objects. The discrete action space with 256 bins per dimension provides ~8mm spatial resolution, sufficient for most tabletop manipulation.

### 4.2 RT-2 (Brohan et al., 2023)

> **Reference:** [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)

RT-2 scales VLA models to the billion-parameter regime by co-fine-tuning pre-trained vision-language models on robot data.

**Key insight.** VLMs pre-trained on web-scale image-text data encode rich semantic knowledge about the physical world. By representing robot actions as text strings and including them in the VLM's standard language modeling objective, this knowledge becomes directly accessible to robot control.

**Action tokenization.** Each action dimension value is discretized to 256 bins and represented as a text string integer between 1 and 256 (e.g., "1 128 91 241 5 101 127" represents one 7-DOF action). These tokens are appended to the VLM's vocabulary, and the model generates action strings token-by-token in its autoregressive output.

**Backbone options.** Two variants were studied:
- **PaLI-X** (55B parameters): A vision-language model with a ViT image encoder and a PaLM language model.
- **PaLM-E** (562B parameters): A multimodal embodied language model.

**Co-training.** The model is fine-tuned jointly on: (a) robot demonstration data (RT-1 dataset), and (b) web-scale VQA and image-caption data. This co-training prevents catastrophic forgetting of web-scale knowledge while adding robotic control capabilities.

**Emergent capabilities.** Because the action model shares weights with the VLM, it inherits capabilities not present in RT-1:
- **Novel object generalization:** Correctly manipulates objects never seen in robot training data but present in web training (e.g., "move the soda can with the recycling logo to the left").
- **Chain-of-thought reasoning:** Can perform intermediate reasoning steps (e.g., "which fruit has the most vitamin C? → kiwi → pick up the kiwi") before outputting actions.
- **Semantic understanding:** Responds correctly to abstract instructions that require world knowledge.

**Evaluation.** Across over 6,000 evaluation trials on a mobile manipulation robot, RT-2-PaLI-X achieves 62% success on novel objects and instructions (vs. 32% for RT-1), demonstrating strong emergent generalization from the co-training recipe.

### 4.3 OpenVLA (Kim et al., 2024)

> **Reference:** [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)

OpenVLA brings VLA capabilities to a 7B-parameter open-source model, enabling training and deployment on consumer hardware.

**Architecture.** Three components are fused:
1. **DINOv2** (ViT-L/14): Self-supervised visual encoder, strong spatial features.
2. **SigLIP** (ViT-SO400M/14): Contrastive vision-language encoder, strong semantic alignment with text.
3. **Llama 2** (7B): Open-source language model backbone.

Image features from DINOv2 and SigLIP are concatenated along the channel dimension and projected into Llama 2's embedding space via a lightweight MLP connector (a design closely following LLaVA 1.5). Language instructions and image tokens are interleaved in the standard Llama 2 autoregressive format.

**Action tokenization.** Each of the 7 action dimensions is discretized into 256 uniform bins over its empirical range in the training data. The 256 bin tokens are appended to Llama 2's vocabulary (expanding the vocabulary from 32,000 to 32,256 tokens). During training, the model outputs the 7 action tokens autoregressively after the instruction and image context.

**Training data.** OpenVLA is trained on the **Open X-Embodiment (OXE)** dataset: ~970,000 demonstration episodes spanning 29 robot manipulation tasks, multiple embodiments (RT-1 robot, Franka, xArm, UR5), and multiple labs. This is the most diverse public robotics dataset assembled as of 2024.

**Efficient fine-tuning.** For new tasks or robot embodiments, OpenVLA supports **LoRA** (Low-Rank Adaptation) fine-tuning: only rank-16 adapter matrices in the attention layers are updated, reducing trainable parameters by ~99%. A single task can be fine-tuned on one consumer GPU (RTX 4090) in a few hours.

**Results.** OpenVLA (7B) outperforms RT-2-X (55B, PaLI-X backbone) by **16.5 percentage points** in success rate on the WidowX robot evaluation suite, despite having 7× fewer parameters. This efficiency gain comes from: the stronger DINOv2+SigLIP visual encoder, the higher-quality OXE training data, and Llama 2's strong language priors.

**Training infrastructure.** The full model was trained on 2,048 TPU-v5e chips for two weeks, but the released checkpoint enables researchers to skip this cost.

### 4.4 π₀ (Physical Intelligence, 2024)

> **Reference:** [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)

π₀ ("pi-zero") combines a pre-trained VLM backbone with a **flow matching** action head, separating the semantic understanding (VLM) from the continuous action generation (flow matching). This hybrid design avoids the quantization artifacts of discrete action tokenization while retaining language generalization.

**Flow matching action head.** Flow matching is a simulation-free generative modeling framework that learns to transport samples from a simple prior $x_0 \sim \mathcal{N}(0, I)$ to the data distribution $x_1 = \text{action}$ along straight-line paths:

$$
\frac{dx}{dt} = v_\theta(x_t, t, o), \quad x_t = t \cdot x_1 + (1-t) \cdot x_0, \quad t \in [0, 1]
$$

The velocity field $v_\theta$ is trained to predict the direction from noise toward the action:

$$
\mathcal{L} = \mathbb{E}_{t, x_0, x_1} \left[ \left\| v_\theta\!\left( x_1 t + x_0(1-t),\; t,\; o \right) - (x_1 - x_0) \right\|^2 \right]
$$

At inference, integration of this ODE from $t=0$ to $t=1$ traces a near-straight path from noise to action, requiring far fewer function evaluations (typically 10–20 Euler steps) than DDPM while achieving comparable or better sample quality.

**Architecture.** The VLM backbone (based on PaliGemma) processes language instructions and multi-camera image observations, producing a rich context embedding. The flow matching head conditions its velocity field $v_\theta$ on this context via cross-attention, and predicts continuous robot actions (joint positions or end-effector poses) without discretization.

**Why this hybrid works.** The VLM backbone provides:
- Semantic understanding of instructions involving novel objects, spatial relationships, and abstract concepts.
- Pre-trained visual features robust to lighting and background variation.

The flow matching head provides:
- Sub-millimeter action precision impossible with 256-bin discretization.
- Multimodal action distributions (multiple valid grasps) captured by the stochastic generation process.
- Smooth trajectory generation via the ODE integration.

**Generalization.** π₀ is trained across multiple robot platforms (Franka, xArm, UR5) and supports dexterous manipulation tasks (folding laundry, assembling boxes, making sandwiches) that require the combination of semantic understanding and fine motor control that neither VLA-style discrete models nor ACT-style deterministic policies alone can provide.

### 4.5 Octo (Ghosh et al., 2024)

> **Reference:** [arXiv:2405.12213](https://arxiv.org/abs/2405.12213)

Octo is a fully open-source generalist robot policy designed to be easily fine-tunable for new robots and tasks, addressing the gap between large proprietary VLAs and single-task policies.

**Architecture.** Octo follows a modular tokenization approach:
1. **Observation tokenizer:** RGB images (one or more cameras) are encoded with a small ViT patch encoder. Proprioceptive states are embedded via linear projection. Language goals are encoded with a pre-trained language model (T5-small).
2. **Transformer backbone:** All tokens (image patches, proprioception, language, task context) are concatenated and processed by a standard transformer encoder with 27 layers and 12 heads (~93M parameters).
3. **Diffusion action head:** The transformer output tokens serve as conditioning for a small diffusion network (U-Net or transformer-based) that generates the action chunk. This decouples the representation learning (transformer) from the action generation (diffusion), enabling the same backbone to support different action spaces.

**Training data.** Octo is trained on 800,000 trajectories from the Open X-Embodiment dataset, covering 9 different robot embodiments and over 25 tasks.

**Fine-tuning.** Because the backbone and action head are modular, fine-tuning Octo to a new robot requires only:
- Replacing or re-initializing the proprioception embedding layer (new joint dimensions).
- Optionally adding a new camera embedding.
- Fine-tuning on as few as 100–500 demonstrations for a specific task.

This process takes a few GPU-hours on a single A100, making Octo accessible to research labs without large compute budgets. Octo outperforms prior generalist policies (RT-1-X, RT-2-X) on the majority of BridgeData V2 and RT-2 benchmark tasks when fine-tuned on target-domain data.

---

## 5. Comparison Table

| Model | Year | Backbone | Action Head | Parameters | Training Data | Key Innovation | Best Result |
|---|---|---|---|---|---|---|---|
| [ACT](https://arxiv.org/abs/2304.13705) | 2023 | ResNet-18 + CVAE Transformer | L1 regression (chunk) | ~80M | 50 human demos / task | Action chunking + temporal ensembling | 80–90% on 6 precision tasks |
| [Diffusion Policy](https://arxiv.org/abs/2303.04137) | 2023 | ResNet / Transformer | DDPM denoising | ~300M (Transformer) | Task-specific demos | Multimodal diffusion over action chunks | State-of-the-art on 11/15 benchmarks |
| [RT-1](https://arxiv.org/abs/2212.06817) | 2022 | EfficientNet-B3 + TokenLearner | 256-bin discrete head | 35M | 130k demos, 700+ tasks, 13 robots | First large-scale robot transformer | 97% success, 700+ tasks |
| [RT-2](https://arxiv.org/abs/2307.15818) | 2023 | PaLI-X (55B) / PaLM-E | Text token actions | 55B | RT-1 + VQA co-training | VLM co-fine-tuning, emergent generalization | 62% novel objects (vs. 32% RT-1) |
| [OpenVLA](https://arxiv.org/abs/2406.09246) | 2024 | Llama 2 + DINOv2 + SigLIP | 256-bin discrete head | 7B | 970k demos, OXE dataset | Open-source VLA, LoRA fine-tuning | +16.5% vs RT-2-X with 7× fewer params |
| [π₀](https://arxiv.org/abs/2410.24164) | 2024 | PaliGemma VLM | Flow matching (continuous) | ~3B | Multi-robot proprietary | Flow matching for continuous dexterous actions | SOTA on dexterous manipulation |
| [Octo](https://arxiv.org/abs/2405.12213) | 2024 | ViT + T5 | Diffusion head | ~93M | 800k demos, 9 embodiments | Open generalist policy, few-GPU fine-tuning | Best open generalist on BridgeData V2 |

---

## Summary

This chapter surveyed the principal architectures for imitation learning at the level of precision required for real-world robot manipulation:

- **Behavioral cloning** provides a clean supervised learning baseline but suffers from compounding errors, multimodality, and distributional shift — problems that worsen with task horizon and precision requirements.

- **ACT** ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)) addresses compounding errors and temporal coherence through action chunking and resolves multimodality via a CVAE latent variable, with temporal ensembling providing the best of both reactive and predictive control.

- **Diffusion Policy** ([Chi et al., 2023](https://arxiv.org/abs/2303.04137)) replaces the CVAE with a DDPM that naturally handles multimodality and high-dimensional continuous actions through iterative denoising.

- **VLA models** (RT-1, RT-2, OpenVLA, π₀, Octo) scale imitation learning to internet-pretrained backbones and massive multi-task datasets. They demonstrate that language-conditioned generalization to novel objects and instructions emerges from co-training VLMs on robot data. The discrete-action (RT-1, RT-2, OpenVLA) and continuous-action (π₀ with flow matching, Octo with diffusion head) paradigms trade off precision versus generalization, with hybrid approaches like π₀ offering the most promising path forward.

The field is converging on a shared architecture pattern: a pre-trained vision-language encoder for semantic grounding, a large transformer backbone for context integration, and a generative action head (diffusion or flow matching) for continuous, multimodal action generation.

---

*Chapter 3 will address data collection, annotation, and dataset curation — the upstream bottleneck for all models described here.*
