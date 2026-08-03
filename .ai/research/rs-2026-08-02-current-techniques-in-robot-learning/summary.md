# Current Techniques in Robot Learning: A Comprehensive Survey

**Compiled:** August 2, 2026  
**Scope:** Techniques active in 2024–2026 literature, covering reinforcement learning, imitation learning, world models, foundation models, sim-to-real transfer, 3D/geometric methods, hierarchical planning, data collection, and emerging architectural innovations.

---

## Table of Contents

1. [Reinforcement Learning for Locomotion](#1-reinforcement-learning-for-locomotion)
2. [Imitation Learning: Action Generation](#2-imitation-learning-action-generation)
3. [Vision-Language-Action (VLA) Foundation Models](#3-vision-language-action-vla-foundation-models)
4. [World Models for Robot Learning](#4-world-models-for-robot-learning)
5. [Sim-to-Real Transfer](#5-sim-to-real-transfer)
6. [3D and Equivariant Policy Learning](#6-3d-and-equivariant-policy-learning)
7. [Hierarchical Planning and Skill Chaining](#7-hierarchical-planning-and-skill-chaining)
8. [Data Collection and Cross-Embodiment Transfer](#8-data-collection-and-cross-embodiment-transfer)
9. [Affordance and Grasp Learning](#9-affordance-and-grasp-learning)
10. [Emerging Architectural Innovations](#10-emerging-architectural-innovations)
11. [Cross-Cutting Themes and Open Challenges](#11-cross-cutting-themes-and-open-challenges)

---

## 1. Reinforcement Learning for Locomotion

RL remains the dominant paradigm for robot locomotion, particularly for legged and humanoid robots. Policies are trained in massively parallel simulation and transferred to hardware.

### Core Algorithms

- **PPO (Proximal Policy Optimization):** The de facto standard for locomotion policy training. Actor-critic with clipped surrogate objective. Used in Humanoid-Gym, Isaac Gym, and most humanoid locomotion frameworks. Typically with 4096+ parallel environments.
- **Off-policy RL (SAC, TD3):** Gaining traction for faster iteration. FastSAC and FastTD3 enable humanoid locomotion training in ~15 minutes on a single RTX 4090, vs. multi-hour PPO training. Better sample efficiency for iterative sim-to-real cycles.
- **Teacher-Student Distillation:** Two-stage approach where a "teacher" policy is trained with privileged simulator information (terrain friction, contact forces, body dynamics), then distilled into a "student" policy that uses only deployable proprioception. The student infers latent environment parameters from observation history.
- **RMA (Rapid Motor Adaptation):** Online adaptation module that infers environment parameters from recent observation history and adjusts the policy in real-time. Enables robust deployment without explicit system identification.

### Key Techniques

- **Domain Randomization:** Randomizing physics parameters (mass, friction, actuator delays, motor properties), sensory noise, and terrain during training to produce policies robust to sim-to-real gaps. Can be excessive — too much randomization yields overly conservative policies.
- **Adversarial Motion Priors (AMP):** Discriminator-based reward that encourages policies to match reference motion distributions. Selective AMP applies the prior only to stability-critical gaits (walking, stair climbing) and omits it for dynamic gaits (running, jumping) where it would over-constrain.
- **Reference Motion Tracking:** Training policies to track human motion capture data retargeted to robot morphology. Enables naturalistic and dynamic locomotion.
- **Massively Parallel Simulation:** Frameworks like Isaac Gym, Isaac Lab, and MuJoCo MJX enable thousands of parallel environments, reducing training from days to minutes.
- **Contrastive Representation Learning:** Forces the actor's latent state to encode privileged environmental information (available only to the critic in simulation) through contrastive losses. Enables "proactive" proprioceptive policies that anticipate terrain without exteroceptive sensors at deployment.
- **Reward Shaping:** Minimalist reward functions (tracking + smoothness + stability) are preferred over complex multi-term rewards, enabling faster hyperparameter sweeps.

### Representative Systems

| System | Key Contribution |
|--------|-----------------|
| Humanoid-Gym | Standardized humanoid RL training framework |
| Booster Gym | End-to-end training-to-deployment pipeline with domain randomization |
| RMA | Online motor adaptation from proprioceptive history |
| CRA-Loco | Contrastive learning for proactive proprioceptive locomotion |
| FastSAC/FastTD3 | 15-minute humanoid locomotion training |

---

## 2. Imitation Learning: Action Generation

Imitation learning (behavior cloning) has become the primary paradigm for manipulation, driven by generative models that can capture multimodal action distributions.

### Action Representation Paradigms

#### Diffusion Policy
- Formulates action generation as conditional denoising diffusion
- Predicts action chunks (short trajectory segments) rather than single actions
- Captures multimodal action distributions naturally
- **Limitation:** High inference latency from iterative denoising (typically 10-50 denoising steps)
- Widely adopted as a strong baseline; used in LIBERO, MetaWorld, and real-robot experiments

#### Flow Matching Policy
- Learns a continuous vector field (ODE) that transports a simple distribution (Gaussian) to the action distribution
- Enables near-single-step inference — much faster than diffusion
- Adopted by π0 and subsequent state-of-the-art methods
- **Variants:**
  - **Streaming Flow Policy:** Treats action trajectories as flow trajectories, enabling on-the-fly action streaming during sampling. Starts from a narrow Gaussian around the last action rather than pure noise.
  - **LAFM (Latent Action Guided Flow Matching):** Replaces the fixed Gaussian prior with an adaptive library of learned prior distributions indexed by latent motion primitives. Reduces vector field entanglement, +23.4% real-world success.
  - **CoLA-Flow:** Performs flow matching in a continuous latent action space, decoupling global motion structure from control noise. +93.7% trajectory smoothness improvement.
  - **CoF (Conditional Flow) Policy:** Models a continuous flow from suboptimal to expert actions. First to conclusively show flow matching outperforming diffusion in imitation learning.

#### ACT (Action Chunking with Transformers)
- CVAE-based architecture that predicts action chunks using a transformer
- Faster inference than diffusion (single forward pass)
- Strong on narrow, single-task settings
- Often outperforms foundation models on task-specific fine-tuning with 500-1000 demonstrations

#### Consistency Policy
- Uses consistency models for fast (1-2 step) action generation
- Bridges the speed gap between ACT and diffusion policy
- Maintains multimodal expressiveness

#### VQ-VAE / Discrete Latent Actions
- Encodes action sequences into discrete tokens via vector quantization
- Enables autoregressive generation and cross-embodiment transfer
- Used as pretraining signal or as structural bias for flow matching

### Key Trends

- **Flow matching is replacing diffusion** as the preferred action generation method due to faster inference
- **Action chunking** (predicting short trajectories rather than single steps) is universal — reduces high-frequency inference demands
- **Receding horizon execution:** Execute first few actions of a predicted chunk, then re-plan. Standard practice across all generative policy methods.
- **Latent action representations** are increasingly used as structural priors rather than just pretraining signals

---

## 3. Vision-Language-Action (VLA) Foundation Models

VLA models extend vision-language models to directly output robot actions, enabling language-conditioned generalist policies.

### Architectural Paradigms

#### Single-Tower (Monolithic) Models
- VLM backbone processes vision, language, and action tokens together in one transformer
- Actions tokenized as text (discretized into bins) and generated autoregressively
- **Examples:** RT-2 (55B, PaLI-X backbone), OpenVLA (7B, LLaMA-2 + DINOv2 + SigLIP)
- **Pros:** Simple architecture, leverages VLM's semantic knowledge
- **Cons:** Inference speed limited to VLM's autoregressive generation (~1-10 Hz)

#### Dual-Tower Models
- Slow VLM backbone handles language and scene understanding at low frequency
- Fast action expert cross-attends to VLM tokens and outputs motor commands at high frequency (50-500 Hz)
- **Examples:** π0 (PaLI-Gemma + flow matching head), GR00T (NVIDIA)
- **Pros:** Decoupled reasoning and control, high-frequency actions
- **Cons:** More complex architecture, careful training needed

#### Hybrid Architectures
- **Octo:** Smaller transformer (~93M) with diffusion action head, trained entirely on robot data (no web data). Fast fine-tuning (<1 hour on single A100). Edge-deployable on Jetson.
- **SmolVLA:** Lightweight VLA with separable proprioceptive conditioning. Compatible with PRISM-style polynomial modules.

### Training Data

- **Open X-Embodiment (OXE):** 1M+ robot trajectories from 22 embodiments, 21 institutions, 527 skills. The standard cross-embodiment dataset.
- **DROID:** Large-scale diverse robot manipulation dataset for Franka platforms.
- **Internet-scale pretraining:** VLM backbones (LLaMA, PaLI, SigLIP, DINOv2) pretrained on web image-text data provide semantic generalization.

### Fine-Tuning Strategies

- **LoRA (Low-Rank Adaptation):** Parameter-efficient fine-tuning. ~4-8 hours on single A100 with 1000 demonstrations.
- **Full fine-tuning:** 4-8× A100s, better performance but more expensive.
- **Action expert freezing:** Freeze VLM backbone, train only action head (SmolVLA pattern).

### Performance Trade-offs

| Factor | Foundation Model | Task-Specific Policy |
|--------|-----------------|---------------------|
| Best when | Many tasks, few demos, language conditioning | Single task, 500-1000 demos, precision |
| Inference speed | 1-10 Hz (single-tower), 50-500 Hz (dual-tower) | 30-100+ Hz |
| Generalization | Strong semantic/visual generalization | Limited to training distribution |
| Fine-tuning cost | $500-800 (LoRA) | Minimal (train from scratch) |

---

## 4. World Models for Robot Learning

World models learn predictive representations of environment dynamics, supporting planning, simulation, data generation, and policy evaluation.

### Architectural Families

- **State-space / Recurrent:** Learn latent dynamics models (e.g., PlaNet, Dreamer family). Predict next latent state from current state + action, then decode. Used for model-based RL and planning.
- **Transformer-based:** Autoregressive prediction of future states/frames. Scale well with data.
- **Diffusion-based generators:** Video diffusion models conditioned on actions/frames. High visual fidelity but slow generation.
- **JEPA (Joint-Embedding Predictive Architecture):** Predict in latent space rather than pixel space (V-JEPA 2). Avoids generative overhead, focuses on predictive representations.
- **Language-augmented:** Use language as an intermediate representation for world dynamics (PhiZero's physical language).

### Functional Roles

- **Policy learning:** World model provides imagined rollouts for model-based RL (Dreamer) or serves as a learned simulator for policy training.
- **Planning:** Model-predictive control using learned dynamics for action selection.
- **Data generation:** Generate synthetic trajectories for data augmentation, especially for rare scenarios.
- **Evaluation:** Use world model as a learned simulator for policy evaluation without real-world deployment.
- **Cross-embodiment transfer:** Disentangle dynamics from appearance to transfer motion patterns across embodiments (PhiZero's zero-shot human-to-robot transfer).

### Representative Systems

| System | Key Contribution |
|--------|-----------------|
| DreamerV3 | Latent imagination-based model-based RL |
| V-JEPA 2 (Meta) | Self-supervised world simulator from video |
| Genie (DeepMind) | Interactive environment generation |
| Cosmos (NVIDIA) | Foundation-scale physical AI world model |
| PhiZero | Physical language for reason-then-render world modeling |
| Sora (OpenAI) | Large-scale video generation (not robot-specific) |

### Key Trend: Reason-Then-Render

PhiZero introduces a paradigm shift: instead of predicting future video directly in pixel space, first reason about state transitions in a compact representation (physical language), then render. This separates dynamics reasoning from visual synthesis, making world evolution an explicit reasoning target.

---

## 5. Sim-to-Real Transfer

The reality gap — discrepancies between simulation and real-world physics, sensing, and dynamics — remains a central challenge.

### Techniques

#### Reducing the Gap (Improving Simulation)
- **High-fidelity rendering:** Geometry-aware Gaussian Splatting for photorealistic simulation environments (HyperSim)
- **System identification:** Calibrate simulator parameters to match real-world dynamics
- **Better physics engines:** Improved contact, friction, and deformation modeling

#### Overcoming the Gap (Robustness to Discrepancies)
- **Domain Randomization:** Train across broad distributions of simulated parameters (masses, frictions, delays, noise, textures, lighting). Most widely used approach.
- **Adversarial Training:** Train adversary agent to apply disturbances, forcing main policy to be robust
- **Meta-Learning:** Train policy across distribution of simulator parameters, enabling fast adaptation to real dynamics
- **Teacher-Student Distillation:** Train teacher with privileged info, distill to student with deployable sensors only

#### Bridging with Real Data
- **Sim-and-Real Co-Training:** Train single policy on mixed sim + real data. Enhanced with Optimal Transport (OT) alignment to learn domain-invariant feature space. Up to 30% improvement in real-world success.
- **Human-in-the-Loop Correction:** TRANSIC — humans observe and correct robot execution; residual policies learned from corrections. Scales with human effort.
- **Continual Cross-Task Transfer:** GeCo-SRT — accumulate transfer knowledge across iterative sim-to-real transfers using geometry-aware mixture-of-experts. 52% improvement over baseline, 6× data efficiency for new tasks.
- **Real-to-Sim-to-Real:** Transform real-world first frames to simulation, execute simulated policy, transform back. PhiZero demonstrates this via physical language.

#### Adversarial Trajectory Generation
- Generate challenging trajectories in simulation that include failures and perturbations
- HyperSim shows 35% higher completion rate under physical perturbations with adversarial trajectories

---

## 6. 3D and Equivariant Policy Learning

3D point cloud observations provide geometry-aware, appearance-invariant representations for manipulation policies.

### SE(3)-Equivariant Policy Learning

Exploiting the rotational and translational symmetries inherent in manipulation tasks: if the scene is rotated, the action should rotate correspondingly.

- **EquiForm:** Noise-robust SE(3)-equivariant framework with geometric denoising module and contrastive equivariant alignment. +17.2% simulation, +28.1% real-world improvement.
- **EquAct:** SE(3)-equivariant multi-task transformer with spherical Fourier features and invariant FiLM layers for language conditioning.
- **Canonical Policy:** Groups point clouds to a canonical representation for principled equivariant mappings. +18.0% simulation, +39.7% real-world.
- **RiEMann:** Near real-time (5.4 fps) SE(3)-equivariant manipulation without point cloud segmentation. 5-10 demonstrations sufficient.
- **E3Flow:** Unifies efficient rectified flow with stable equivariant learning. 7× inference speedup over Spherical Diffusion Policy.
- **Spherical Diffusion Policy (SDP):** Diffusion-based policy with spherical harmonic representations for SO(3) equivariance.

### 3D Scene Flow as Intermediate Representation

- **3D Flow Diffusion Policy (3D FDP):** Predicts temporal trajectories of sampled 3D query points (scene flow), then conditions action generation on these flows. Captures fine-grained local motion cues for contact-rich manipulation. SOTA on MetaWorld (50 tasks).

### Key Insights

- Equivariance provides strong inductive bias for data efficiency (5-10 demos sufficient in some cases)
- Noise robustness is critical — real point clouds have depth noise, occlusions, missing regions
- Combining equivariance with fast sampling (flow matching) is an active frontier
- 3D representations complement but don't replace 2D/image-based methods — hybrid approaches are emerging

---

## 7. Hierarchical Planning and Skill Chaining

For long-horizon, multi-step tasks, hierarchical architectures decompose the problem into planning and execution layers.

### Three-Layer Architectures

1. **High-level planner:** LLM/VLM generates task decomposition and skill sequences from language instructions and visual observations
2. **Mid-level skills:** Imitation-learned skill policies (each trained from demonstrations) produce motion targets
3. **Low-level controller:** RL-trained tracking controller executes joint-level commands

### LLM/VLM-Based Planning

- **ReAct-style prompting:** Interleave reasoning with environmental feedback for dynamic re-planning
- **Chain-of-Thought (CoT):** Break down manipulation tasks into step-by-step logic
- **Dual-LLM modules:** Separate high-level planning from low-level spatial reasoning to prevent "token exhaustion"
- **VLM-based skill monitoring:** Continuously verify skill completion using pretrained VLMs, orchestrating transitions
- **PDDL + LLM hybrid:** HSP-Plan combines LLM task decomposition with symbolic PDDL planning for verifiable plans

### Task and Motion Planning (TAMP) Extensions

- **Task and Skill Planning (TASP):** Extends TAMP to integrate closed-loop motor controllers (not just kinematic motion planning) using Composable Interaction Primitives
- **Capability-driven planning (RoboAgent):** Define vision-language capabilities that produce intermediate reasoning or atomic actions; central scheduler invokes appropriate capabilities

### Key Challenges

- Error compounding across hierarchy levels
- Skill library coverage — can't plan if required skills don't exist
- Real-time skill monitoring and failure recovery
- Bridging abstract plans to physical execution

---

## 8. Data Collection and Cross-Embodiment Transfer

Data is the primary bottleneck for robot learning at scale. Multiple paradigms address this.

### Teleoperation

- **Leader-follower setups:** ALOHA, GELLO — operator controls a leader arm, follower arm replicates
- **VR-based teleoperation:** Apple Vision Pro for humanoid upper-body control via wrist pose retargeting
- **Spacemouse / gamepad / keyboard:** Lower-fidelity but accessible teleoperation
- **Isomorphic exoskeletons:** SuperSuit — wearable arm mirrors robot kinematics for natural whole-body mobile manipulation demonstration

### Human Video-Based Data

- **Hand-Object Interaction (HOI) videos:** RoboWheel pipeline — reconstruct hand-object interactions from monocular RGB(D), enforce physical plausibility, retarget to robot embodiments. First quantitative evidence that HOI videos serve as effective supervision.
- **Human motion retargeting:** Human2LocoMan — human pretraining + robot fine-tuning for cross-embodiment manipulation
- **Internet video mining:** Large-scale human manipulation videos as auxiliary supervision for VLA training

### Cross-Embodiment Transfer

- **Open X-Embodiment:** 22 embodiments, 1M+ trajectories. Shows positive transfer across robots.
- **Functional similarity (CEI):** Quantify shared interaction behaviors across end-effectors using Directional Chamfer Distance. Transfer between parallel-jaw grippers and dexterous hands (82.4% transfer ratio).
- **Unified I/O frameworks (RIO):** Abstract robot control, teleoperation, and data formatting across diverse hardware platforms and morphologies
- **Action representation unification:** Standardize action spaces (relative EEF, joint deltas) across embodiments

### Automated Data Generation

- **Simulated demonstration generation:** MimicGen — automatically generate demonstrations across novel scene configurations
- **Trajectory transformations:** Offline augmentation of existing datasets through geometric transforms
- **Adversarial trajectory generation:** Include failures and perturbations for robustness training

---

## 9. Affordance and Grasp Learning

Affordance prediction bridges visual perception and manipulation by identifying how objects can be interacted with.

### Affordance Representations

- **Static contact points:** Where to grasp/interact (2D heatmaps or 3D regions)
- **Dynamic action directions:** How to move after contact (post-contact motion vectors)
- **Affordance maps:** Dense pixel/voxel-level predictions of interactable regions

### Methods

- **VLM-based affordance prediction:** Use large multimodal models as semantic oracles for zero-shot grasp selection (ORACLE-Grasp). Dual-prompt strategy: extract semantics, then identify graspable regions.
- **Reasoning-based segmentation:** AffordanceGrasp-R1 — CoT cold-start SFT + RL for reasoning-driven affordance segmentation. Chain-of-thought reasoning enhances spatial grounding.
- **Retrieval-augmented prediction (RAAP):** Combine dense correspondence for contact localization with retrieval-augmented alignment for action direction. Works with tens of samples per task.
- **Cross-modal diffusion (AffordGrasp):** Latent diffusion with dual-conditioning (physical plausibility + semantic guidance) for affordance-aware grasp synthesis.
- **Large-scale benchmarks:** RAGNet — 273K images, 180 categories, 26K reasoning instructions for affordance segmentation.

### Grasp Generation

- **6-DOF grasp pose prediction:** From point clouds or depth images, generate grasp configurations
- **Affordance-conditioned grasping:** Generate grasp candidates from full scene, filter using affordance masks (preserves global geometry)
- **Semantic grasping:** Grasp location depends on functional intent (handle vs. rim of a cup), not just geometry

---

## 10. Emerging Architectural Innovations

### Polynomial Interaction Representations (PRISM)

- Factorized polynomial module that exposes multiplicative interactions among proprioceptive variables
- Replaces linear conditioning with learnable quadratic/higher-order interactions
- Yields sensorless compliance without force/tactile sensors
- Backbone-agnostic: works with PPO, Diffusion Policy, SmolVLA

### Physical Language (PhiZero)

- Discrete, compact representation of world-state transitions learned self-supervised from video
- 256 tokens represent 4 seconds of video (175× compression vs. dense VAE)
- Enables zero-shot cross-embodiment motion transfer and sim-to-real appearance transfer

### Latent Action Representations

- Discrete or continuous latent codes that capture motion primitives
- Used as structural priors for flow matching (LAFM), pretraining signals, or cross-embodiment transfer media
- Enable trajectory-level reasoning rather than per-timestep generation

### Dual-System Designs

- System 1 (fast, reactive): Action expert running at high frequency
- System 2 (slow, deliberative): VLM backbone for reasoning and planning
- Mirrors cognitive science dual-process theory
- Adopted by π0, GR00T, and hierarchical humanoid frameworks

### Efficient Inference Techniques

- **Action trajectory polynomials:** FLASH — Legendre polynomial trajectory representation for single-inference coverage of extended action horizons. 175× faster than diffusion policies.
- **Consistency models:** 1-2 step generation for real-time control
- **Streaming inference:** Actions streamed during generative sampling process
- **VLM token compression:** Reduce visual token count for faster VLA inference

---

## 11. Cross-Cutting Themes and Open Challenges

### Themes

1. **Generative models are central:** Diffusion and flow matching dominate action generation; diffusion also used for world models and grasp synthesis.
2. **Foundation model scaling:** VLA models follow LLM/VLM scaling laws — more data, more parameters, more embodiments → better generalization. But inference speed remains a bottleneck.
3. **Simulation is necessary but insufficient:** Almost all methods train in simulation; real-world validation is often limited to small pilots. Sim-to-real remains the critical gap.
4. **Inductive biases matter:** Equivariance (SE(3)), polynomial interactions (PRISM), physical language (PhiZero) — structured representations outperform pure scaling.
5. **Data diversity > data volume:** Cross-embodiment, cross-task, and cross-environment diversity drive generalization more than raw dataset size.

### Open Challenges

- **Real-world validation:** Most published results are in simulation. Comprehensive real-robot benchmarks are rare and expensive.
- **Long-horizon tasks:** Hierarchical planning helps but error compounding across layers remains unsolved.
- **Contact-rich and deformable manipulation:** Most methods excel at rigid object pick-and-place; deformables, tight tolerances, and sustained contact remain hard.
- **Inference speed vs. expressiveness:** Generative policies (diffusion, flow matching) are expressive but slow; fast alternatives (ACT, consistency) sacrifice some expressiveness.
- **Safety and robustness:** Deploying learned policies in safety-critical settings requires guarantees that current methods don't provide.
- **Data efficiency:** Despite progress, most methods still need 50-1000+ demonstrations per task. Human video data and cross-embodiment transfer are promising but not solved.
- **Benchmark standardization:** Inconsistent evaluation protocols, metrics, and task definitions make cross-paper comparison difficult.
- **Sim-to-real for manipulation:** Locomotion sim-to-real is relatively mature (domain randomization works well); manipulation sim-to-real is harder due to contact dynamics, visual fidelity, and object diversity.

---

## Key References

### Surveys
- Robot Learning Tutorial (arXiv 2510.12403) — comprehensive tutorial with LeRobot examples
- World Model for Robot Learning: A Comprehensive Survey (arXiv 2605.00080)
- Robotic Manipulation via Imitation Learning: Taxonomy, Evolution, Benchmark (arXiv 2508.17449)
- Large VLM-based VLA Models for Robotic Manipulation: A Survey (arXiv 2508.13073)
- VLA Models for Robotics: A Review Towards Real-World Applications (IEEE Access 2025)
- The Reality Gap in Robotics: Challenges, Solutions, and Best Practices (arXiv 2510.20808)
- Efficient VLA Models: A Systematic Survey (arXiv 2510.17111)

### Foundation Models
- RT-2 (Google DeepMind, 2023) — 55B VLA, actions as text tokens
- OpenVLA (Stanford + Berkeley, 2024) — 7B open-weight VLA, OXE training
- Octo (Berkeley, 2024) — 93M transformer + diffusion head
- π0 (Physical Intelligence, 2024) — VLM + flow matching, bimanual dexterity
- Open X-Embodiment (22 institutions, 2023) — cross-embodiment dataset

### Action Generation
- Diffusion Policy (Chi et al., 2023) — diffusion-based action chunks
- ACT (Zhao et al., 2023) — CVAE + transformer action chunks
- Flow Matching / π0 (2024) — ODE-based action generation
- Streaming Flow Policy (Jiang et al., 2025) — streaming during sampling
- LAFM (2026) — latent action guided flow matching priors
- FLASH (2026) — Legendre polynomial trajectory representation

### World Models
- PhiZero (arXiv 2607.28624) — physical language world model
- DreamerV3 — latent imagination model-based RL
- V-JEPA 2 (Meta) — self-supervised predictive representations
- Genie (DeepMind) — interactive environment generation

### Sim-to-Real
- TRANSIC (Jiang et al., 2025) — human-in-the-loop correction
- HyperSim (2026) — holistic sim-to-real with Gaussian Splatting
- GeCo-SRT (CVPR 2026) — continual cross-task transfer
- Generalizable Domain Adaptation (arXiv 2509.18631) — OT-based co-training

### 3D / Equivariant
- EquiForm (arXiv 2601.17486) — noise-robust SE(3)-equivariant policy
- EquAct (arXiv 2505.21351) — SE(3)-equivariant multi-task transformer
- Canonical Policy (IEEE TRO 2026) — canonical 3D representation
- 3D Flow Diffusion Policy (arXiv 2509.18676) — 3D scene flow intermediate
- E3Flow (arXiv 2603.23227) — equivariant + flow matching

### Architectural Innovations
- PRISM (arXiv 2607.23473) — polynomial interaction representations
- PRISM (Performer RS-IMLE, arXiv 2602.02396) — single-pass multisensory imitation (different work, same name)

### Data Collection
- RoboWheel (CVPR 2026) — HOI video to cross-embodiment data
- RIO (arXiv 2605.11564) — flexible cross-embodiment robot I/O
- Human2LocoMan (RSS 2025) — human pretraining for quadrupedal manipulation
- SuperSuit (arXiv 2603.06280) — bimodal wearable data collection

### Hierarchical Planning
- VLP-Humanoid (arXiv 2506.22827) — VLM planning + IL skills + RL control
- RoboAgent (CVPR 2026) — capability-driven embodied task planning
- TASP (arXiv 2504.17901) — task and skill planning with composable primitives

### Affordance / Grasping
- RAGNet (ICCV 2025) — large-scale affordance segmentation benchmark
- AffordanceGrasp-R1 (arXiv 2602.03547) — reasoning-based affordance with RL
- RAAP (arXiv 2603.29419) — retrieval-augmented affordance prediction
- ORACLE-Grasp (2026) — zero-shot LMM-guided grasping
