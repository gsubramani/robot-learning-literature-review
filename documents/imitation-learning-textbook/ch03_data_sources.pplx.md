# Chapter 3: Data Sources and Collection Approaches

> **Chapter Prerequisites:** Familiarity with behavior cloning (Chapter 1) and the action chunking formulation (Chapter 2). Readers should be comfortable with basic PyTorch and NumPy.

---

## 3.1 The Data Bottleneck in Imitation Learning

Language models and vision systems are data-hungry but data-rich: GPT-3 trained on roughly 300 billion tokens drawn from the web, a corpus whose cost of collection is essentially zero beyond the compute to crawl and filter it. Robotics is the inverse: data is expensive to generate, heterogeneous in format, and physically bound to real hardware and real time.

### Why Robot Data Is Scarce

A single demonstration of picking up a coffee cup takes roughly 5–10 seconds of real-world interaction. You cannot download 100,000 coffee-cup pick demonstrations from the internet, because humans rarely hold joysticks while doing household tasks. Every trajectory must be staged, recorded, and quality-checked. A full ALOHA bimanual dataset for one task might represent two hours of a researcher's time to collect 50 high-quality episodes — 10 minutes of robot interaction time spread across setup, retries, and labeling.

The scarcity problem compounds across three axes:

| Axis | Vision/NLP | Robotics |
|------|-----------|---------|
| **Data source** | Passive web crawl | Active human teleoperation |
| **Marginal cost per sample** | ~\$0 | \$1–\$10 per episode |
| **Embodiment portability** | Universal (pixels/text) | Hardware-specific (joint angles, EEF poses) |
| **Label density** | Self-supervised (next token) | 50 Hz action labels required |

### Dimensionality of a Demonstration

A single ALOHA-style demonstration episode at 50 Hz is a sequence of tuples:

$$
\tau = \bigl\{(o_t,\, a_t)\bigr\}_{t=0}^{T}
$$

where each observation $o_t$ and action $a_t$ contains:

- **Images:** 4 cameras × 3 channels × 480 × 640 pixels → 3,686,400 uint8 values per timestep
- **Joint positions:** 14-DoF (7 joints per arm) → 14 float32 values  
- **Actions:** 14-DoF joint targets → 14 float32 values

At 50 Hz and a 10-second episode, that is 500 timesteps. The raw image volume alone is ~1.8 GB per episode before compression. Compare this to a single NLP training token: 2 bytes.

### Data Efficiency: What Good IL Can Achieve

Despite this scarcity, well-designed imitation learning algorithms are surprisingly data-efficient. The [ALOHA system](https://arxiv.org/abs/2304.13705) using Action Chunking with Transformers (ACT) achieves 80–90% success rates on precision manipulation tasks — threading cable ties, slotting batteries, opening translucent condiment cups — with roughly 50 demonstrations per task, or about 10 minutes of total demonstration time.

This stands in stark contrast to the data regimes of other ML domains:

```
GPT-3 (language): 300,000,000,000 tokens
ImageNet (vision):  14,197,122     labeled images
RT-1 (robotics):       130,000     robot episodes
ACT (robotics):             50     episodes per task  ← works!
```

The implication for practitioners: **smart data collection and smart algorithms can substitute for brute-force scale, at least within narrow task distributions.** This chapter explains how to build and exploit that data efficiently.

---

## 3.2 Teleoperation Systems

Teleoperation — a human controlling a robot in real time — is the dominant source of high-quality robot demonstrations. The key design challenge is maximizing human comfort and intuitive control while minimizing the embodiment gap between the human's motions and the robot's actions.

### 3.2.1 ALOHA: A Low-Cost Open-Source Hardware System

[ALOHA (A Low-cost Open-source Hardware System for Bimanual Teleoperation)](https://arxiv.org/abs/2304.13705) is the most widely adopted open-source teleoperation platform for IL research as of 2024. Its design philosophy is to democratize robot learning by keeping total system cost under \$20,000 — one to two orders of magnitude cheaper than dexterous hand systems.

#### Hardware Architecture

ALOHA uses a **leader-follower** configuration with two pairs of robot arms:

| Component | Model | Role |
|-----------|-------|------|
| Follower arms (2×) | Trossen ViperX 300 S (6-DoF each) | Execute the task |
| Leader arms (2×) | Trossen WidowX 250 S (6-DoF each) | Held by human operator |
| Cameras | 4× Intel RealSense D405 (848×480 RGB) | Visual observations |
| Control computer | Single workstation | Records all data at 50 Hz |

The WidowX leader arms share the same kinematic structure as the ViperX follower arms in a smaller form factor. The operator physically "puppeteers" the leader arms by backdriving them — no motors resist the human — and the follower arms mirror the joint angles via a PID controller running at >1 kHz.

```
Human hand → WidowX leader joints (θ_L)
                    ↓ PID at >1 kHz
           ViperX follower joints (θ_F)
```

#### Camera Configuration

Four cameras provide overlapping visual coverage:

1. **Overhead (top):** Wide-angle workspace view, useful for spatial reasoning  
2. **Front (wrist-eye):** Forward-facing scene context  
3. **Left wrist:** Close-up of left end-effector during manipulation  
4. **Right wrist:** Close-up of right end-effector during manipulation

Wrist cameras are particularly important for contact-rich tasks, where the global views cannot resolve the millimeter-scale details needed by the policy.

#### Data Format

Each episode is recorded as a fixed-length array of timesteps at 50 Hz:

```
Episode:
  images:     [T, 4, 3, 480, 640]  uint8
  qpos:       [T, 14]              float32   # joint positions (state)
  qvel:       [T, 14]              float32   # joint velocities
  action:     [T, 14]              float32   # recorded leader positions
  is_pad:     [T]                  bool      # padding mask for variable-length chunks
```

#### The Key Insight: Leader Positions as Actions

A subtle but important design choice: **the recorded actions are the leader joint positions, not the follower joint positions.** Because PID tracking is imperfect, the difference between leader and follower positions encodes implicit force information — when the robot is pushing against an object, the follower lags the leader, and this lag is captured in the data. A policy trained on leader positions learns to command with that implicit force intent baked in.

**Cost Breakdown (approximate, 2023):**

| Component | Cost |
|-----------|------|
| 2× ViperX 300 follower arms | ~\$10,000 |
| 2× WidowX 250 leader arms | ~\$6,000 |
| Cameras, mounts, compute | ~\$4,000 |
| **Total** | **~\$20,000** |

This contrasts sharply with Shadow Robot Hand setups (>\$100,000 per hand) or purpose-built bimanual systems (>\$500,000).

### 3.2.2 Other Teleoperation Approaches

The teleoperation design space spans a wide range of cost, dexterity, and operator burden:

**Shadow Robot Teleoperation System**  
The Shadow Dexterous Hand has 24 joints per hand and uses tendons driven by pneumatic muscles, achieving near-human dexterity. Teleoperation uses a CyberGlove or exoskeleton to capture the operator's hand configuration. At \$100,000+ per hand, this system targets research on dexterous manipulation where ALOHA-style parallel-jaw grippers are insufficient. The dataset volume is correspondingly small due to cost and operator fatigue.

**DexPilot (Handa et al., 2020)**  
DexPilot uses a commodity depth camera to track the operator's hand pose in real time, retargeting the resulting keypoints to a multi-fingered robot hand via inverse kinematics. This eliminates wearable hardware entirely. The accuracy is lower than exoskeleton-based methods but the setup cost is dramatically reduced. Operators can collect data without calibration per session, enabling faster iteration.

**UMI (Universal Manipulation Interface)**  
UMI takes a different architectural approach: the "robot" during data collection *is* a hand-held gripper. A 3D-printed parallel-jaw gripper equipped with a GoPro wrist camera becomes the demonstration device. The operator carries it through the task in the real environment, and a separate retargeting pipeline converts the gripper's trajectory (via SLAM-based pose estimation) to robot arm commands for a downstream arm like a UR5. Because collection happens without any robot arm present, demonstrations can be gathered in any location at about 30 seconds per episode. The wrist camera view directly matches the robot's deployment camera view, eliminating the camera calibration gap.

**VR-Based Teleoperation (Zhang et al., 2018)**  
VR headsets (HTC Vive, Meta Quest) track the operator's head and hand poses at low latency and high precision. The 6-DoF hand controllers map to robot end-effector targets, with IK solving for joint angles in real time. VR naturally captures wrist orientation, which is often missing from joystick-based interfaces. The immersive first-person view can improve task performance for cluttered or occluded scenarios.

**HoloDex (Arunachalam et al., 2022)**  
HoloDex combines an Oculus Quest hand tracking (no controllers, bare hand) with a mixed reality display that overlays the robot's workspace. The operator sees their hand superimposed on the robot's camera feed, providing intuitive visual feedback about contact and workspace constraints. Dexterous hand retargeting is performed via a learned mapping from human finger keypoints to robot joint angles.

### 3.2.3 Data Quality Considerations

Not all demonstrations are equally useful for learning. Several factors determine whether a dataset will yield a high-performing policy:

**Non-Stationarity of Human Demonstrations**  
Human operators do not execute the same motion twice. Given 50 demonstrations of the same task, you will observe at least 10–20 qualitatively distinct strategies (approach angle, grasp point, trajectory shape). This multimodality is a fundamental challenge for behavior cloning with MSE regression losses, which average over modes and produce blend trajectories that fail at all of them. ACT's CVAE architecture specifically addresses this by sampling a latent style variable $z$ that conditions the policy on a particular demonstration mode:

$$
\pi_\theta(a_{t:t+H} \mid o_t, z), \quad z \sim q_\phi(z \mid a_{t:t+H}, o_t)
$$

During training, the CVAE encoder infers $z$ from the demonstration; at test time, $z \sim \mathcal{N}(0, I)$ samples a mode.

**Demonstration Diversity vs. Consistency**  
There is a tension between two desiderata:

- **Coverage:** The dataset should cover the space of initial conditions (object positions, lighting, background clutter) the robot will encounter at test time.  
- **Consistency:** Within a given context, demonstrations should agree on a coherent strategy, so the policy can learn a well-defined behavior rather than an average.

The practical resolution is to collect demonstrations from a small number of skilled operators who develop consistent personal styles, while varying the *environment* (object placement, lighting) across episodes.

**What Makes a Good Demonstration Dataset**

| Property | Bad | Good |
|----------|-----|------|
| **Coverage** | All episodes start from the same object position | Object positions sampled from a distribution |
| **Episode length** | Uniform padding to max length | Variable-length with padding masks |
| **Success rate** | Include all attempts | Filter to successful completions only |
| **Operator quality** | Multiple untrained operators | 1–2 trained operators with consistent style |
| **Noise level** | Erratic, high-variance motions | Smooth, deliberate trajectories |

A practical rule of thumb: **50 high-quality demonstrations from one skilled operator** typically outperforms **200 demonstrations from many casual operators** on precision tasks.

---

## 3.3 Large-Scale Datasets

The field has progressively moved from task-specific datasets toward large shared corpora. This shift mirrors the ImageNet era in computer vision and enables pre-training of general robot policies.

### 3.3.1 Open X-Embodiment (OXE)

[Open X-Embodiment](https://arxiv.org/abs/2310.08864) is the most ambitious effort to aggregate robot learning data across institutions and embodiments. Assembled through a collaboration between 21 research institutions, it standardizes data from 22 different robot platforms into a single training corpus demonstrating 527 distinct skills across 160,266 tasks.

**Scale Summary:**

| Metric | Value |
|--------|-------|
| Robot embodiments | 22 |
| Contributing institutions | 21 |
| Distinct skills | 527 |
| Total task episodes | 160,266 |
| Data format | RLDS (TFRecord) |

**Key Finding: Data Diversity > Data Quantity**  
The RT-X models trained on OXE demonstrate *positive transfer* — policies trained on the combined dataset outperform policies trained only on data from the evaluation robot. More strikingly, the improvement is primarily explained by behavioral diversity, not raw episode count. A subset of OXE data representing diverse tasks and embodiments provides more generalization benefit than a larger homogeneous subset from a single robot.

This has a practical implication for practitioners: **when constructing a training mixture, prioritize breadth of skills over depth of demonstration count for any single skill.**

### 3.3.2 DROID Dataset

[DROID (Distributed Robot Interaction Dataset)](https://arxiv.org/abs/2403.12945) contains 76,000 demonstration trajectories — approximately 350 hours of interaction data — collected across 564 scenes and 86 tasks by 50 data collectors in North America, Asia, and Europe over 12 months. All demonstrations use the Franka Emika Panda arm with various end-effectors.

DROID's defining feature is its *in-the-wild* diversity: scenes span labs, offices, and household environments with real-world clutter, varied lighting, and non-standard object arrangements. Policies co-trained on DROID show 22% absolute improvement in in-distribution task success and 17% improvement in out-of-distribution scenarios compared to co-training with OXE alone.

### 3.3.3 Bridge Data (Ebert et al., 2021)

The [Bridge Dataset](https://arxiv.org/abs/2109.13396) was an early demonstration of the value of multi-domain, multi-task data for cross-domain generalization. It contains 7,200 demonstrations spanning 71 tasks across 10 tabletop environments, collected on a WidowX robot.

The central finding: jointly training on Bridge data plus 50 target-task demonstrations yields a **2× improvement** in success rate compared to using target-domain data alone. This established the template for the co-training paradigm now used throughout the field — collect a large diverse base dataset, then fine-tune or co-train on smaller in-domain data.

### 3.3.4 RT-1 Dataset

The [RT-1 (Robotics Transformer 1)](https://arxiv.org/abs/2212.06817) dataset was collected by Google using 13 robot arms over 17 months, producing 130,000+ episodes spanning 700 distinct natural-language-specified tasks. The breadth of tasks ranges from picking and placing objects to opening drawers, wiping surfaces, and knocking objects over.

RT-1's scaling study provides some of the clearest empirical data on the data-performance relationship: moving from 10,000 to 130,000 demonstrations shows consistent improvement in generalization to novel objects and environments, with no sign of saturation at the collected scale. This dataset established that **internet-scale imitation learning for robotics requires at minimum ~100k diverse episodes** to achieve meaningful zero-shot generalization.

### 3.3.5 LIBERO

[LIBERO](https://arxiv.org/abs/2306.03310) is a simulation benchmark specifically designed for the lifelong learning (LLDM) setting, where a robot must acquire a sequence of tasks without forgetting previous ones. It uses a procedural generation pipeline to construct task suites with varying objects, spatial arrangements, and instruction phrasings.

The standard LIBERO suite provides:
- 4 task suites (LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-Long)  
- 10 tasks per suite  
- 50 high-quality demonstrations per task (scripted oracle)  
- MuJoCo physics simulation

LIBERO is useful as a **controlled benchmark** where dataset size, task ordering, and domain distribution can be precisely controlled — making it ideal for ablation studies that would be prohibitively expensive to run on real hardware.

### 3.3.6 Data Format: RLDS (Robot Learning Dataset Specification)

RLDS is the de facto standard format for sharing robot learning datasets. Developed by Google DeepMind and adopted by OXE, it provides a consistent schema built on TensorFlow Datasets (TFDS) and serialized as TFRecord files.

**Schema Structure:**

```
Dataset
└── Episodes (tf.data.Dataset of episodes)
    └── Steps (tf.data.Dataset per episode)
        ├── observation/
        │   ├── image            [H, W, 3]   uint8     (or JPEG bytes)
        │   ├── wrist_image      [H, W, 3]   uint8
        │   ├── joint_pos        [D]          float32
        │   └── language_instruction  str
        ├── action              [A]           float32
        ├── reward              scalar        float32
        ├── discount            scalar        float32
        ├── is_first            bool
        ├── is_last             bool
        └── is_terminal         bool
```

**Loading RLDS Data in Python:**

```python
import tensorflow_datasets as tfds
import tensorflow as tf

# Load Bridge Data V2 in RLDS format
dataset = tfds.load('bridge_dataset', split='train', data_dir='/data/rlds')

# Iterate over episodes and steps
for episode in dataset:
    steps = episode['steps']
    for step in steps:
        obs_image = step['observation']['image']      # tf.Tensor [480, 640, 3]
        action    = step['action']                    # tf.Tensor [7]
        is_last   = step['is_last']                  # bool

# For PyTorch users: use the dlimp bridge library
# pip install dlimp
from dlimp.dataset import DLataset

dl_dataset = DLataset.from_rlds('bridge_dataset', data_dir='/data/rlds')
# Returns standard torch DataLoader-compatible interface
```

The key advantage of RLDS is **schema interoperability**: any model trained on OXE can load any OXE-compatible dataset without format-specific parsing code, because the episode-of-steps nesting and field names are standardized.

---

## 3.4 Simulation Data Generation

Simulation complements real data collection by enabling safe exploration, automatic task resets, and massive parallelism — all without robot hardware.

### 3.4.1 Why Simulation

**Advantages:**
- **Safe exploration:** Policies can fail catastrophically without hardware damage  
- **Automatic resets:** A scripted reset oracle returns the environment to a canonical initial state in milliseconds, enabling thousands of episodes per hour  
- **Parallelism:** Modern GPU simulators (IsaacGym) can run 4,096+ environments simultaneously on a single GPU  
- **Ground-truth labels:** Object poses, contact forces, and gripper state are available without sensors  
- **Counterfactual generation:** The same scene can be re-simulated with perturbed parameters (mass, friction, appearance) for domain randomization

**Sim-to-Real Gap:**  
The fundamental limitation of simulation is the *sim-to-real gap* — discrepancies between simulated and real dynamics that cause policies to fail when deployed. The gap manifests in three forms:

| Gap Type | Examples | Mitigation |
|----------|----------|------------|
| **Visual** | Unrealistic textures, lighting, shadows | Domain randomization, photorealistic rendering |
| **Physical** | Incorrect contact stiffness, friction, deformability | System identification, domain randomization |
| **Sensor** | Noiseless joint encoders, ideal cameras | Adding sensor noise models |

Closing the visual gap has become tractable with photorealistic renderers; the contact physics gap remains the hardest challenge for manipulation.

### 3.4.2 Key Simulators for IL

**MuJoCo (Multi-Joint dynamics with Contact)**  
MuJoCo remains the most widely used simulator for IL/RL research. It uses a reduced-coordinate dynamics formulation that is fast and numerically stable, and its contact model is well-understood if not physically perfect. The `dm_control` library provides standard task suites. MuJoCo 3.x supports GPU-accelerated batch simulation via MJX (JAX-based), enabling hundreds of parallel environments.

**IsaacGym / IsaacLab**  
NVIDIA's IsaacGym (now superseded by [IsaacLab](https://research.nvidia.com/publication/2025-09_isaac-lab-gpu-accelerated-simulation-framework-multi-modal-robot-learning)) runs the full physics simulation on the GPU, eliminating CPU-GPU transfer overhead. This enables 2,000–4,096 parallel environments on a single A100, reducing wall-clock training time from days to hours. IsaacLab adds photorealistic RTX rendering, tactile sensors, and first-class imitation learning support (SpaceMouse/VR teleoperation, demonstration replay).

**RLBench (James et al., 2020)**  
[RLBench](https://claru.ai/benchmarks/rlbench) provides 100 structured manipulation tasks (pick-place, drawer opening, peg insertion, knob turning) implemented in CoppeliaSim (V-REP) via the PyRep API. Each task includes scripted oracle demonstrations for automatic dataset generation — run the oracle, collect trajectories, train a policy. The structured task API and consistent scripted demonstrations make RLBench the standard benchmark for multi-task IL, despite its older rendering quality.

**SAPIEN**  
SAPIEN specializes in articulated object simulation — doors, drawers, fridges, dishwashers with accurate joint physics. It is the backend for PartNet-Mobility and ManiSkill benchmarks, which focus on generalizing across object instances within categories.

**Genesis**  
[Genesis](https://genesis-world.readthedocs.io) (released December 2024) is a new Python-native physics engine that claims 10–80× faster simulation than IsaacGym/MJX for single-environment scenarios, reaching 43 million FPS for a Franka manipulation scene on a single RTX 4090. It integrates rigid body, soft body, fluid, and cloth solvers in a unified framework and includes differentiable physics for gradient-based planning. A generative framework for automatic scene and task construction from natural language prompts is planned.

**Simulator Comparison:**

| Simulator | Physics Speed | GPU Parallel | Rendering | Differentiable | Primary Use |
|-----------|--------------|--------------|-----------|----------------|-------------|
| MuJoCo | Medium | Via MJX | Basic | Via MJX/JAX | RL/IL research |
| IsaacLab | High | ✓ (4096+) | Photorealistic (RTX) | Partial | Large-scale IL/RL |
| RLBench | Medium | ✗ | Basic | ✗ | Multi-task benchmarks |
| SAPIEN | Medium | ✗ | Good | ✗ | Articulated objects |
| Genesis | Very high | ✓ | Photorealistic | Partial (MPM) | General-purpose, new |

### 3.4.3 Scripted Policy Data

The fastest way to generate large simulation datasets is via **scripted (oracle) policies**: hard-coded controllers that solve tasks using ground-truth simulator state (object poses, robot kinematics). Because the oracle has access to information unavailable at test time, scripted policies are not directly deployable — but their trajectories are high-quality demonstrations.

A typical scripted demonstration pipeline for a pick-and-place task:

```python
def collect_scripted_episode(env, rng):
    """Generate one oracle demonstration."""
    obs = env.reset(seed=rng.integers(1e9))
    object_pose = env.get_ground_truth_pose('target_object')  # privileged info
    
    # Phase 1: reach
    pre_grasp = object_pose.position + np.array([0, 0, 0.1])
    for waypoint in linear_interpolate(env.robot.eef_pos, pre_grasp, steps=20):
        obs, _, done, _ = env.step(ik_solve(waypoint))
    
    # Phase 2: grasp
    grasp_pose = object_pose.position
    for waypoint in linear_interpolate(pre_grasp, grasp_pose, steps=10):
        obs, _, done, _ = env.step(ik_solve(waypoint))
    env.step(close_gripper())
    
    # Phase 3: transport and place
    ...
    
    return episode_buffer
```

RLBench and LIBERO both provide scripted demonstration generators. OXE contributors from simulation environments typically use this approach — the oracle policy is tuned once, then run for thousands of episodes with randomized initial conditions.

### 3.4.4 Human Video Data

Human video — YouTube cooking demonstrations, Epic-Kitchens, Ego4D — represents a massive latent data source: billions of hours of humans manipulating objects in realistic environments. The challenge is that video provides no direct robot action signal.

**EgoVLA: Bridging Egocentric Video and Robot Actions**

[EgoVLA](https://arxiv.org/abs/2507.12440) trains a Vision-Language-Action model on egocentric human videos to predict *human wrist and hand actions*. These are then retargeted to robot joint angles via inverse kinematics:

$$
\underbrace{
  \text{Egocentric video} \xrightarrow{\text{VLA}} \hat{p}^{\text{wrist}}_{t+1:t+H}
}_{\text{human action prediction}}
\xrightarrow{\text{IK + retargeting}}
\underbrace{
  q^{\text{robot}}_{t+1:t+H}
}_{\text{robot joint targets}}
$$

The model is then fine-tuned on a small number of robot demonstrations (~50 per task) to bridge the remaining embodiment gap. Significant improvements over baselines on bimanual manipulation benchmarks show that human video provides a meaningful initialization for dexterous skills.

**The Embodiment Gap:**  
Human hands differ fundamentally from robot grippers:
- 21 DoF vs. 1 DoF (parallel-jaw)  
- Compliance and tactile sensing vs. rigid fingers  
- Unobserved contact forces in video  

**Retargeting via IK:**  
A standard retargeting pipeline tracks hand keypoints $\{p_i^{\text{human}}\}$ (wrist, fingertips) and solves:

$$
q^* = \arg\min_q \sum_i \bigl\| \text{FK}(q)_i - \mathcal{T}(p_i^{\text{human}}) \bigr\|^2 + \lambda \|q - q_{\text{prev}}\|^2
$$

where $\text{FK}(q)_i$ is the forward kinematics of the robot's $i$-th relevant joint, and $\mathcal{T}$ is a learned or geometric mapping from human to robot keypoint space. The regularization term $\lambda \|q - q_\text{prev}\|^2$ enforces trajectory smoothness.

**Limitations:**  
- Unobserved forces cannot be recovered from RGB video  
- Camera egomotion entangles with hand motion  
- Retargeting errors accumulate over long horizons  
- Success depends on task similarity to the human video domain

---

## 3.5 Data Augmentation

When real data is scarce, augmentation extends its effective coverage. Robotics augmentation must preserve action-observation consistency: augmenting an image without a corresponding adjustment to the action labels produces inconsistent data.

### 3.5.1 Visual Augmentation

**Standard Pixel-Level Augmentation**  
The simplest augmentations apply standard computer vision transforms:

```python
import torchvision.transforms as T

visual_augmentations = T.Compose([
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.RandomCrop(size=(460, 620), padding=None),   # then resize back to 480×640
    T.RandomHorizontalFlip(p=0.0),  # only if action is in image space; 
                                    # NEVER flip joint-space actions
])
```

> **Warning:** Geometric augmentations (flips, rotations) are only valid when the action space is defined relative to the augmented image. For absolute joint-position actions, only color/appearance augmentations are safe.

**Novel-View Synthesis (NeRF-based)**  
Scene-level augmentation reconstructs a 3D NeRF representation from collected demonstrations, then re-renders observations from camera poses not present in the original dataset. This provides geometrically consistent multi-view data without additional collection. Zhou et al. (2023) demonstrate that NeRF-augmented datasets improve generalization to novel camera placements at test time.

**RoVi-Aug: Robot and Viewpoint Augmentation**  
[RoVi-Aug](https://arxiv.org/abs/2409.03403) takes a more aggressive approach: it uses image-to-image generative models to synthesize demonstrations showing *a different robot* (embodiment swap) or *from a different camera angle* (viewpoint swap). Actions are copied from the original demonstration; only the visual observation is changed.

Results from RoVi-Aug (CoRL 2024):
- **Zero-shot transfer** to an unseen robot with significantly different camera angles (no test-time processing required)  
- Up to **30% improvement in success rate** when co-training on original + augmented data  
- Enables multi-robot, multi-task policy training from single-robot datasets

The method requires no assumed knowledge of test-time camera angles and allows policy fine-tuning, distinguishing it from retargeting approaches like Mirage.

### 3.5.2 Action Space Augmentation: DART

[DART (Disturbances for Augmenting Robot Trajectories)](http://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf) (Laskey et al., 2017) directly addresses the covariate shift problem in behavior cloning: a policy trained on expert trajectories never sees the off-nominal states it will encounter when its own imperfect actions compound.

DART's solution is to inject optimized noise into the *supervisor's* demonstrations during collection, forcing the supervisor to demonstrate corrective behavior from near-trajectory states:

$$
\tilde{a}_t = a_t^{\text{expert}} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \Sigma^*)
$$

where $\Sigma^*$ is optimized to approximate the error distribution of the trained policy. The supervisor naturally compensates for the injected perturbations, producing a richer demonstration that covers the "recovery funnel" around the nominal trajectory.

**Algorithm (simplified):**

```
1. Collect N demonstrations with initial noise parameter Σ₀
2. Train policy π̂_θ on collected data
3. Estimate policy error covariance: Σ* ≈ E[‖π̂(o) - π_expert(o)‖²]
4. Collect N more demonstrations with noise Σ*
5. Train final policy on aggregated dataset
```

In a grasping-in-clutter task, DART achieves a **62% performance increase** over standard behavior cloning by providing corrective examples at the boundary of the expert's distribution. Unlike DAgger (which requires querying the expert at test-time states), DART is entirely offline after the noise-injection phase.

### 3.5.3 Data Mixing Strategies

When combining multiple data sources — task-specific demos, large-scale pre-training corpora, and internet video — the mixing ratio significantly affects policy performance.

**Task-Proportional Sampling:**  
Sample each task with probability proportional to its number of demonstrations. This preserves the relative frequency of skills in the original dataset but can undersample rare tasks.

**Uniform Task Sampling:**  
Sample each task with equal probability, regardless of demonstration count. This overrepresents rare tasks and can prevent catastrophic forgetting in multi-task settings.

**Co-Training Ratios for VLAs:**  
Vision-Language-Action models that co-train on robot data and internet (image, text) data must balance the two regimes. Typical ratios used in practice:

| Data source | Typical ratio |
|-------------|--------------|
| In-domain robot demos | 50–70% |
| Large robot corpus (OXE/DROID) | 20–40% |
| Internet image-text data | 5–15% |

The right ratio depends on the degree of domain shift between in-domain and co-training data; more similar corpora warrant higher co-training weight.

**Temperature-Based Resampling:**  
To interpolate between task-proportional and uniform sampling, use a temperature $T$:

$$
p_k \propto n_k^{1/T}
$$

where $n_k$ is the number of demonstrations for task $k$. $T=1$ is proportional sampling; $T \to \infty$ is uniform sampling. Values around $T = 2$ are common in practice.

---

## 3.6 Data Collection Pipeline: End-to-End

A complete data pipeline for an ALOHA-style real-robot IL system involves six stages:

```
1. Hardware Setup
        ↓
2. Teleoperation Interface
        ↓
3. Recording System
        ↓
4. Data Preprocessing
        ↓
5. Dataset Format Conversion
        ↓
6. Training
```

### Stage 1: Hardware Setup

Calibrate camera extrinsics and intrinsics. Verify joint limits. Set up e-stop. Ensure the control loop (PID follower) is running at >1 kHz before any recording begins. A typical ROS 2 node structure:

```
/leader_joint_states     → joint_controller → /follower_cmd
/camera_{top,front,left_wrist,right_wrist}/image_raw
/recording_manager       → HDF5 writer
```

### Stage 2: Teleoperation Interface

The operator grips the leader arms and performs the task. A foot pedal or keyboard shortcut triggers episode start/stop. A display shows live camera feeds and episode count. Quality filters (minimum episode length, success confirmation by operator) gate which episodes are saved.

### Stage 3: Recording System

Data is written to HDF5 files in real time. The file structure follows ALOHA conventions:

```
episode_0042.hdf5
├── action            [500, 14]   float32
├── observations/
│   ├── images/
│   │   ├── cam_high      [500, 480, 640, 3]   uint8
│   │   ├── cam_low       [500, 480, 640, 3]   uint8
│   │   ├── cam_left_wrist  [500, 480, 640, 3] uint8
│   │   └── cam_right_wrist [500, 480, 640, 3] uint8
│   └── qpos              [500, 14]   float32
└── metadata
    ├── task_name       "transfer_cube"
    ├── operator_id     "op_02"
    └── success         true
```

### Stage 4: Data Preprocessing

Before training, apply:

1. **Filtering:** Remove failed episodes, episodes shorter than 50 steps, or outlier episodes (e.g., joint limit violations)  
2. **Normalization:** Compute per-dimension mean and standard deviation of `action` across the dataset; normalize to zero mean, unit variance  
3. **Image compression:** Re-encode images to JPEG at quality 90 to reduce disk I/O during training  
4. **Chunking:** Pre-compute action chunks of length $H$ (e.g., $H=100$) with padding masks for episodes shorter than $H$

### Stage 5: Dataset Format Conversion

For integration with the OXE ecosystem, convert HDF5 to RLDS. A minimal converter:

```python
import tensorflow_datasets as tfds
import tensorflow as tf
import h5py
import glob

class AlohaDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for ALOHA HDF5 data → RLDS format."""
    
    VERSION = tfds.core.Version('1.0.0')
    
    def _info(self):
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(shape=(480, 640, 3)),
                        'wrist_image': tfds.features.Image(shape=(480, 640, 3)),
                        'joint_pos': tfds.features.Tensor(shape=(14,), dtype=tf.float32),
                    }),
                    'action': tfds.features.Tensor(shape=(14,), dtype=tf.float32),
                    'is_first': tf.bool,
                    'is_last': tf.bool,
                    'is_terminal': tf.bool,
                }),
            })
        )
    
    def _generate_examples(self, paths):
        for hdf5_path in paths:
            with h5py.File(hdf5_path, 'r') as f:
                T = f['action'].shape[0]
                steps = []
                for t in range(T):
                    steps.append({
                        'observation': {
                            'image': f['observations/images/cam_high'][t],
                            'wrist_image': f['observations/images/cam_right_wrist'][t],
                            'joint_pos': f['observations/qpos'][t],
                        },
                        'action': f['action'][t],
                        'is_first': t == 0,
                        'is_last': t == T - 1,
                        'is_terminal': t == T - 1,
                    })
                yield hdf5_path, {'steps': steps}
```

### Stage 6: Training

The PyTorch `Dataset` class that feeds the ACT policy:

```python
import torch
import h5py
import numpy as np
from pathlib import Path


class AlohaDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for ALOHA-style HDF5 demonstrations.
    
    Each item returns a fixed-length action chunk suitable for ACT/Diffusion Policy.
    """

    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 100,
        camera_names: list[str] | None = None,
        norm_stats: dict | None = None,
    ):
        """
        Args:
            data_dir:     Directory containing episode_XXXX.hdf5 files.
            chunk_size:   Length H of the action chunk (timesteps).
            camera_names: Which cameras to include.
            norm_stats:   {'action': {'mean': ..., 'std': ...}, 'qpos': {...}}
                          Precomputed from the training set.
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.camera_names = camera_names or ['cam_high', 'cam_low',
                                              'cam_left_wrist', 'cam_right_wrist']
        self.norm_stats = norm_stats

        # Discover all episode files and build an index of (file, timestep) pairs
        episode_files = sorted(Path(data_dir).glob('episode_*.hdf5'))
        assert len(episode_files) > 0, f"No HDF5 files found in {data_dir}"

        self._index: list[tuple[Path, int]] = []
        for ep_path in episode_files:
            with h5py.File(ep_path, 'r') as f:
                T = f['action'].shape[0]
            # Each valid start index t ∈ [0, T-1] produces a chunk [t, t+H)
            for t in range(T):
                self._index.append((ep_path, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        ep_path, t_start = self._index[idx]

        with h5py.File(ep_path, 'r') as f:
            T = f['action'].shape[0]

            # --- Joint position at query timestep ---
            qpos = f['observations/qpos'][t_start]          # (14,)

            # --- Action chunk [t_start, t_start + chunk_size) ---
            t_end   = min(t_start + self.chunk_size, T)
            chunk_len = t_end - t_start
            action = f['action'][t_start:t_end]             # (chunk_len, 14)

            # Pad with last action if episode ends before chunk_size
            is_pad = np.zeros(self.chunk_size, dtype=bool)
            if chunk_len < self.chunk_size:
                pad = np.tile(action[-1:], (self.chunk_size - chunk_len, 1))
                action = np.concatenate([action, pad], axis=0)
                is_pad[chunk_len:] = True                   # mask padding

            # --- Images at query timestep ---
            # Shape: (num_cameras, H, W, 3) → (num_cameras, 3, H, W) float32
            images = []
            for cam in self.camera_names:
                img = f[f'observations/images/{cam}'][t_start]  # (480, 640, 3)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))              # CHW
                images.append(img)
            images = np.stack(images, axis=0)               # (4, 3, 480, 640)

        qpos   = qpos.astype(np.float32)
        action = action.astype(np.float32)

        # --- Normalize (if stats provided) ---
        if self.norm_stats is not None:
            qpos   = (qpos   - self.norm_stats['qpos']['mean'])   \
                   / (self.norm_stats['qpos']['std'] + 1e-8)
            action = (action - self.norm_stats['action']['mean']) \
                   / (self.norm_stats['action']['std'] + 1e-8)

        return {
            'images':   torch.from_numpy(images),    # (4, 3, H, W)
            'qpos':     torch.from_numpy(qpos),      # (14,)
            'actions':  torch.from_numpy(action),    # (chunk_size, 14)
            'is_pad':   torch.from_numpy(is_pad),    # (chunk_size,)
        }


def compute_norm_stats(data_dir: str) -> dict:
    """Compute per-dimension mean/std of actions and qpos across the dataset."""
    all_actions, all_qpos = [], []
    for ep_path in Path(data_dir).glob('episode_*.hdf5'):
        with h5py.File(ep_path, 'r') as f:
            all_actions.append(f['action'][:])
            all_qpos.append(f['observations/qpos'][:])

    actions = np.concatenate(all_actions, axis=0)   # (N_total, 14)
    qpos    = np.concatenate(all_qpos,    axis=0)   # (N_total, 14)

    return {
        'action': {'mean': actions.mean(0), 'std': actions.std(0)},
        'qpos':   {'mean': qpos.mean(0),    'std': qpos.std(0)},
    }


# --- Usage example ---
if __name__ == '__main__':
    norm_stats = compute_norm_stats('/data/aloha/transfer_cube')
    
    dataset = AlohaDataset(
        data_dir='/data/aloha/transfer_cube',
        chunk_size=100,
        norm_stats=norm_stats,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    for batch in loader:
        # batch['images']:  (32, 4, 3, 480, 640)
        # batch['qpos']:    (32, 14)
        # batch['actions']: (32, 100, 14)
        # batch['is_pad']:  (32, 100)
        print(batch['images'].shape, batch['actions'].shape)
        break
```

---

## 3.7 Data Scaling Laws

A central practical question for any IL project is: *how many demonstrations do I need?* The empirical evidence, while still sparse compared to language model scaling laws, reveals a consistent qualitative picture.

### Per-Task Data Scaling

For precision manipulation tasks with a well-designed policy architecture:

| Demonstrations | Expected Success Rate (ACT-style) | Notes |
|---------------|----------------------------------|-------|
| 5–10 | 20–40% | Underfits; poor generalization |
| 20–30 | 50–70% | Viable for simple tasks |
| 50 | **80–90%** | Sweet spot for most tasks |
| 200+ | 85–95% | Diminishing returns within a fixed task distribution |

The [ALOHA paper](https://arxiv.org/abs/2304.13705) demonstrates 80–90% success with 50 demonstrations for six distinct precision tasks, including battery insertion and cable tie threading. The key enabling factor is ACT's action chunking, which reduces effective sequence length and the CVAE's ability to handle multimodal demonstrations.

### Cross-Task and Large-Scale Scaling

For generalist policies trained across many tasks, the relationship is different:

**RT-1 scaling (10k → 130k episodes):** [RT-1](https://arxiv.org/abs/2212.06817) shows consistent improvement in zero-shot generalization to novel instructions and objects as the training set grows from 10,000 to 130,000 episodes. Unlike per-task scaling, cross-task performance continues improving past 50k episodes, suggesting that novel instruction generalization requires substantially more data than in-distribution task performance.

**OXE diversity effect:** The [OXE study](https://arxiv.org/abs/2310.08864) shows that data *diversity* (across tasks, embodiments, environments) predicts cross-robot generalization better than raw data volume. An RT-X model trained on a diverse 160k-episode mixture outperforms a model trained on a 10× larger single-robot dataset on held-out tasks.

**Practical scaling guidelines:**

```
Per-task performance goal:
  ├── 80-90% success:  ~50 demos with ACT/Diffusion Policy
  ├── 90-95% success:  ~100-200 demos + stronger augmentation
  └── 95%+:            Consider policy architecture before adding more data

Cross-task generalization goal:
  ├── Novel positions/objects:  50 demos × 10 tasks = 500 total
  ├── Novel environments:       Co-train with Bridge/DROID (~70k scale)
  └── Novel skills zero-shot:   OXE-scale co-training (160k+)
```

**The data flywheel:** Systems that achieve good performance from 50 demos can bootstrap a *data flywheel*: deploy the policy, use it to collect more data semi-automatically (human in the loop for failures only), and iterate. RT-2 and π₀ both exploit this loop to grow their datasets significantly beyond their initial collection phases.

---

## Chapter Summary

This chapter has established the data landscape for imitation learning:

1. **Robot data is scarce but surprisingly efficient to use.** ACT achieves 80–90% success from 50 demonstrations; GPT-3 required 300B tokens. The difference is that robot demonstrations are dense, task-relevant, and at exactly the right level of abstraction.

2. **Teleoperation is the primary data source.** ALOHA's leader-follower design, 4-camera setup, 14-DoF 50Hz recording, and ~\$20k cost make it the reference platform for open-source IL research. Design choices — recording leader positions as actions, using wrist cameras, filtering for operator quality — have outsized impact on downstream policy performance.

3. **Large-scale datasets enable generalization.** OXE (160k tasks, 22 robots), DROID (76k trajectories, in-the-wild), and RT-1 (130k episodes, 700 tasks) provide the pre-training substrate for generalist policies. The key empirical finding: diversity beats quantity for cross-embodiment and cross-domain transfer.

4. **Simulation complements real data.** Scripted oracle policies, GPU-parallelized simulators (IsaacLab, Genesis), and differentiable physics enable data generation at scales impossible in the real world. The sim-to-real gap remains the binding constraint, particularly for contact-rich manipulation.

5. **Augmentation is a force multiplier.** RoVi-Aug provides embodiment transfer; DART addresses covariate shift at collection time; NeRF-based novel-view synthesis extends camera coverage. Visual augmentations must be applied carefully — geometric transforms are invalid for joint-space action data.

6. **Scaling laws are emerging.** Per-task performance saturates around 50–200 demonstrations with modern architectures. Cross-task generalization continues to improve with data volume and diversity well past the per-task saturation point.

The next chapter will take this data and examine how policy architectures consume it — from behavior cloning with visual backbones to transformer-based sequence models and diffusion policies.

---

## References

- Zhao, T.Z., Kumar, V., Levine, S., & Finn, C. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. *arXiv:2304.13705*. <https://arxiv.org/abs/2304.13705>
- Embodiment Collaboration, O'Neill, A., et al. (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. *arXiv:2310.08864*. <https://arxiv.org/abs/2310.08864>
- Ebert, F., Yang, Y., Schmeckpeper, K., et al. (2021). Bridge Data: Boosting Generalization of Robotic Skills with Cross-Domain Datasets. *arXiv:2109.13396*. <https://arxiv.org/abs/2109.13396>
- Brohan, A., Brown, N., Carbajal, J., et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. *arXiv:2212.06817*. <https://arxiv.org/abs/2212.06817>
- Khazatsky, A., Pertsch, K., et al. (2024). DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset. *arXiv:2403.12945*. <https://arxiv.org/abs/2403.12945>
- Liu, B., et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. *arXiv:2306.03310*. <https://arxiv.org/abs/2306.03310>
- Chen, L.Y., Xu, D., Karthik, I., et al. (2024). RoVi-Aug: Robot and Viewpoint Augmentation for Cross-Embodiment Robot Learning. *arXiv:2409.03403*. <https://arxiv.org/abs/2409.03403>
- Yang, R., Yu, Q., Wu, Y., et al. (2025). EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos. *arXiv:2507.12440*. <https://arxiv.org/abs/2507.12440>
- Laskey, M., Lee, J., Fox, R., Dragan, A., & Goldberg, K. (2017). DART: Noise Injection for Robust Imitation Learning. *CoRL 2017*. <http://proceedings.mlr.press/v78/laskey17a/laskey17a.pdf>
- Chi, C., Feng, S., Du, Y., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. *RSS 2023*.
- James, S., Ma, Z., Arrojo, D.R., & Davison, A.J. (2020). RLBench: The Robot Learning Benchmark & Learning Environment. *IEEE Robotics and Automation Letters*.
- Handa, A., et al. (2020). DexPilot: Vision-Based Teleoperation of Dexterous Robotic Hand-Arm System. *ICRA 2020*.
- Chi, C., et al. (2024). Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots. *RSS 2024*. <https://arxiv.org/abs/2402.10329>
