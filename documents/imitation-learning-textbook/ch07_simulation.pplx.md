# Chapter 7: The Scope of Simulations

Simulation plays a dual role in imitation learning: as a **data factory** (generating demonstrations at a scale impossible to collect physically) and as a **safe evaluation arena** (measuring policy performance without risking hardware). This chapter gives practitioners a working map of the simulators, benchmarks, and sim-to-real transfer methods that define current IL practice.

---

## 1. Role of Simulation in Imitation Learning

Simulation is not a second-class substitute for real data — it is a qualitatively different tool with distinct strengths and failure modes. The main use cases are:

### Pre-training before Real-World Fine-tuning

Training a policy entirely in simulation and then transferring to real hardware is the classic sim-to-real pipeline. Recent large-scale examples:
- **OpenVLA** was pre-trained on the Open X-Embodiment dataset, which includes both real and simulated demonstrations ([Kim et al., 2024](https://arxiv.org/abs/2406.09246))
- **Octo** incorporates simulated data from multiple benchmarks alongside real robot demonstrations

The value here is scale: a simulated environment can generate millions of demonstrations overnight; a human teleoperator can generate perhaps a thousand per day.

### Generating Diverse Demonstrations via Scripted Oracles

In simulation, a scripted oracle with access to ground-truth state (object positions, contact forces, end-effector pose) can produce near-perfect demonstrations. This bypasses the human bottleneck in data collection and enables:
- **Systematic coverage** of object positions, orientations, and configurations
- **Automatic perturbation** of initial conditions for data augmentation
- **Language annotation** via procedural generation of task descriptions

The limitation: scripted oracle demonstrations may not match the distribution of human demonstrations. Scripted trajectories tend to be more direct and less variable than human teleop, which can hurt generalization to human-like task execution styles.

### Safe Evaluation and Benchmarking

Evaluating a manipulation policy requires resetting the environment to a known state, which is time-consuming and wear-inducing on physical hardware. Simulation enables:
- Hundreds of evaluation episodes in minutes
- Reproducible initial conditions (exact object poses, same randomization seed)
- Metrics that require ground-truth state (contact forces, grasp success classified by physics)

### Curriculum Design

Simulation allows programmatic control over task difficulty — a capability that is very hard to replicate in the real world:
- Start with objects placed directly in the gripper; gradually increase the reaching distance
- Begin with a single object; add distractors progressively
- Reduce oracle guidance over training iterations

---

## 2. Physics Simulators

### 2.1 MuJoCo

MuJoCo (Multi-Joint dynamics with Contact) is the de facto standard physics engine for robot learning research. Originally developed by Todorov et al. and now maintained by Google DeepMind.

**Core design philosophy:** Accuracy of contact dynamics at high simulation speed.

**Performance:** ~10,000 simulation steps per second single-threaded; ~100,000 fps with batching on modern CPUs. This speed comes from a particular contact model (smooth, analytic) that approximates real friction and contact well for rigid bodies.

**Ecosystem:** Gym (now Gymnasium), dm\_control, and ManiSkill all expose MuJoCo environments through standardized APIs. RLBench internally uses CoppeliaSim but many derived IL tasks are benchmarked on equivalent MuJoCo setups.

**Key limitation for vision-based policies:** MuJoCo's rendering quality is functional but not photorealistic. The visual gap between MuJoCo renders and real-world camera images is large enough to require separate visual encoders or domain randomization to bridge.

### 2.2 IsaacGym / IsaacLab (NVIDIA)

NVIDIA IsaacGym (now superseded by IsaacLab built on IsaacSim 4.x) runs the physics simulation entirely on the GPU, enabling massive parallelism.

**Performance:** 1,000–10,000 parallel environments on a single GPU. Training data that would take weeks in MuJoCo can be generated in hours.

**GPU-sim2GPU-policy pipeline:** Because both simulation and the policy training loop run on the same GPU, there is no PCIe bottleneck for observation transfer. This is critical for RL-based data generation for IL:

```python
# IsaacGym-style parallel environment loop (pseudocode)
envs = isaacgym.VecEnv(
    num_envs=4096,
    sim_device="cuda:0",
    physics_engine=gymapi.SIM_PHYSX,
)

# All operations on GPU tensors — no CPU transfer
obs = envs.reset()  # shape: (4096, obs_dim) on CUDA

for step in range(num_steps):
    with torch.no_grad():
        actions = policy(obs)          # (4096, action_dim) — GPU
    obs, rewards, dones, info = envs.step(actions)   # GPU
    
    # Store in replay buffer on GPU
    replay_buffer.add(obs, actions, rewards, dones)
```

**IL use case:** IsaacGym is primarily used for generating large-scale scripted or RL-optimized demonstrations that are then used to train IL policies offline. Direct behavioral cloning from IsaacGym data is common when high-quality scripted oracles are available.

**Limitation:** PhysX (the underlying engine) handles simple rigid body contact well but is less accurate than MuJoCo for complex multi-contact scenarios (e.g., deformable grasping, tight insertion). The contact model is not analytically smooth, making gradient-based policy optimization harder.

### 2.3 RLBench

RLBench is a large-scale robot learning benchmark built on CoppeliaSim (formerly V-REP). It is primarily a benchmark rather than a standalone simulator, but it has become a standard evaluation suite for IL.

**Scale:** 100 structured manipulation tasks organized into families (reach, grasp, open/close, press, etc.)

**Key features:**
- Ground-truth waypoints and scripted oracle demonstrations for all 100 tasks
- Multi-camera setups (front, wrist, overhead)
- Language instructions for each task
- Standardized train/test splits

**Who uses it:** PerceiverActor, RVT (Robotic View Transformer), RVT-2, and many other IL papers report results on RLBench, typically on a 18-task subset with 10 demonstrations per task — this is the de facto comparison point for manipulation IL research as of 2024.

### 2.4 SAPIEN / ManipulateNow

SAPIEN is a physically accurate simulator specifically designed for articulated object manipulation. Its key asset is the **PartNet-Mobility** dataset: 2,346 articulated objects with annotated part structure (doors, drawers, handles, buttons, faucets).

**Why it matters for IL:** Most real-world manipulation involves articulated objects (opening cabinets, operating appliances). SAPIEN provides the only large-scale benchmark for this class of tasks with physically accurate joint dynamics.

**ManiSkill** (built on SAPIEN) packages a benchmark suite for manipulation with standardized APIs compatible with Gym.

### 2.5 Genesis

Genesis is an emerging differentiable physics simulator combining:
- GPU-accelerated rigid body and soft body simulation
- Differentiable dynamics (gradients flow through simulation steps)
- Photorealistic neural rendering via Gaussian splatting or NeRF-based representations

**Relevance to IL:** The combination of differentiable physics and photorealistic rendering opens a path to gradient-based policy optimization that operates directly in pixel space — closing the loop between simulation and visual observation without domain randomization. Still early-stage (2024–2025) but worth tracking.

### 2.6 Simulator Comparison Table

| Simulator | Physics Engine | Speed (FPS) | Rendering | Parallelism | Key IL Use |
|---|---|---|---|---|---|
| MuJoCo | MuJoCo (analytic contact) | ~10,000 (single thread) | Functional; not photorealistic | CPU batch, limited GPU | dm\_control, Gym tasks, ACT benchmarks |
| IsaacGym / IsaacLab | PhysX (GPU) | 100,000+ (4096 envs) | Decent; path tracing in IsaacSim | 4,000–10,000 envs/GPU | Large-scale scripted demo generation |
| RLBench | Bullet via CoppeliaSim | ~100 | Good; configurable cameras | Limited | Structured task benchmarking (PerceiverActor, RVT) |
| SAPIEN | PhysX | ~1,000 | Raytracing supported | CPU | Articulated object manipulation (ManiSkill) |
| Gazebo / PyBullet | Bullet | ~500 | Basic | Limited | Legacy; still used in ROS ecosystems |
| Genesis | Custom differentiable | ~5,000 (early) | Photorealistic (NeRF/3DGS) | GPU | Gradient-based policy learning (emerging) |

---

## 3. Benchmark Tasks

A benchmark is more than a simulator — it is a standardized evaluation protocol that enables fair comparison across methods. This section describes the main IL benchmarks in current use.

### 3.1 ACT Benchmarks

The ACT paper introduced two simulation tasks and six real-world tasks on the ALOHA dual-arm setup.

**Simulation tasks (MuJoCo):**
- **Transfer Cube:** Pick up a small red cube from one side of a workspace and transfer it to the other
- **Bimanual Insertion:** Insert a peg into a socket using two arms

Both tasks run at **50 Hz** for **300–1000 timesteps**. Success is binary per episode (object reaches goal region). These tasks are deceptively hard because bimanual coordination requires temporally precise synchronization between two arms.

**Real-world tasks (ALOHA):**
- Open a two-liter bottle
- Pick up a battery and insert it into a slot
- Tape a cable to a surface
- Slot a CPU cooling fan
- Thread a Ziploc bag
- Move a toy block

The real tasks involve contact-rich manipulation that simulation cannot fully replicate, making the real performance numbers the primary benchmark.

### 3.2 RLBench

RLBench's 100 tasks span six families:

| Family | Examples | Difficulty |
|---|---|---|
| Reach | ReachTarget, PickAndLift | Low |
| Push | PushButton, SlidePot | Low–Medium |
| Grasp and Place | PickUpCup, PlaceWineBottle | Medium |
| Open/Close | OpenDoor, OpenDrawer | Medium–High |
| Tool use | ScoopWithSpatula, WaterPlants | High |
| Dexterous | RotateTap, TakePlateOffColoredDishRack | High |

**Standard IL evaluation protocol:** 18-task subset, 10 demonstrations per task, evaluated over 25 episodes. PerceiverActor established this protocol; RVT and RVT-2 follow it. This is intentionally low-data to test sample efficiency rather than asymptotic performance.

### 3.3 LIBERO

LIBERO is a benchmark specifically designed for **lifelong imitation learning** — evaluating a policy's ability to learn new tasks without forgetting previous ones ([Liu et al., 2023](https://arxiv.org/abs/2306.03310)).

**Four task suites:**

| Suite | Focus | Tasks | Demos/Task |
|---|---|---|---|
| LIBERO-Spatial | Object position variation | 10 | 50 |
| LIBERO-Object | Object category variation | 10 | 50 |
| LIBERO-Goal | Task goal variation | 10 | 50 |
| LIBERO-Long | Long-horizon tasks (5+ subtasks) | 10 | 50 |

Total: 130 tasks across all suites with high-quality human-teleoperated demonstrations.

**Key design decisions:** Each suite isolates a single axis of variation, making it possible to attribute failure modes to specific generalization challenges. LIBERO-Long is particularly valuable as a stress test for methods that use action chunking or recurrent representations.

### 3.4 MetaWorld

MetaWorld provides 50 robot manipulation tasks with a shared workspace, robot, and object set. This standardization is intentional: it separates task diversity from embodiment diversity.

**Structure:**
- **MT10:** 10 tasks for multi-task training
- **MT50:** All 50 tasks simultaneously
- **ML1/ML10/ML45:** Meta-learning splits (train on \(n\) tasks, generalize to held-out tasks)

MetaWorld was originally designed for RL; it is now widely used for IL comparison because the same environment supports both paradigms. Success metrics are per-task binary success rates averaged over 50 episodes.

### 3.5 SIMPLER

SIMPLER (Simulated Manipulation Policy and Language Evaluation for Real Robots) is Google DeepMind's effort to build simulation benchmarks that are **predictive of real-world performance**.

**Key innovation:** SIMPLER reconstructs real-world BridgeV2 and RT-X evaluation environments in simulation with photorealistic rendering that closely matches actual camera images. This allows sim-to-real correlation to be measured directly.

**Purpose:** If SIMPLER performance correlates with real-robot performance, it becomes a fast proxy for expensive real-robot evaluation. Early results show strong correlation for tasks in the training distribution but weaker correlation for OOD configurations — suggesting that visual fidelity alone is not sufficient to close the sim-to-real gap entirely.

---

## 4. Sim-to-Real Transfer

The core challenge is that a policy trained in simulation will encounter a distribution shift when deployed on real hardware. This shift has three sources.

### 4.1 The Reality Gap

**Visual domain gap:** The most significant gap for vision-based policies.
- Texture and material differences: simulated objects lack the micro-texture of real surfaces
- Lighting: real environments have complex, time-varying illumination; simulated lighting is static
- Depth: real cameras introduce noise, blur, and distortion absent from simulated renders
- Shadows and reflections: rare in simulation, ubiquitous in real environments

**Physical gap:** Contact dynamics in simulation are approximations.
- Friction coefficients: hard to calibrate without extensive real-world measurement
- Deformable objects: most simulators treat soft bodies poorly
- Contact instabilities: real grasps involve subtle slipping and restabilization that simulators miss

**Sensor gap:** Real sensors are noisy and latent.
- Proprioceptive noise: joint encoders have quantization error and backlash
- Camera latency: typically 30–60 ms on real hardware; simulation is instantaneous
- Calibration drift: real camera-to-robot extrinsic calibration is imperfect and drifts

### 4.2 Domain Randomization

Domain Randomization (DR) is the most widely used sim-to-real strategy: randomize simulation parameters during training so that the real world looks like just another sample from the training distribution.

**Parameters to randomize for manipulation IL:**

```python
# Domain randomization during demo generation (pseudocode)
randomization_config = {
    # Visual
    "object_texture":     UniformSampler(texture_library),
    "table_texture":      UniformSampler(texture_library),
    "lighting_intensity": UniformSampler(0.5, 2.0),
    "lighting_direction": UniformSphereSampler(),
    "camera_pose":        GaussianSampler(mean=nominal_pose, std=0.02),
    "camera_fov":         UniformSampler(60, 80),
    
    # Physical
    "object_mass":        UniformSampler(0.05, 0.5),   # kg
    "friction_coeff":     UniformSampler(0.3, 1.2),
    "object_position":    UniformSampler(workspace_bounds),
    "object_orientation": UniformSphereSampler(),
    
    # Sensor
    "observation_noise_std": UniformSampler(0.0, 0.01),
    "action_latency_steps":  UniformSampler(0, 3),
}

for episode in range(num_episodes):
    env.apply_randomization(randomization_config)
    demo = scripted_oracle.collect_demo(env)
    dataset.add(demo)
```

**The limits of domain randomization:** There is a Goldilocks zone. Too little randomization and the policy does not transfer. Too much randomization and the policy cannot learn fine-grained manipulation — if the object could be anywhere in a 50 cm cube, the policy receives no useful signal about where to actually reach. In practice, DR works well for reach-and-grasp but poorly for tight insertions (peg-in-hole, USB ports) where the required visual precision exceeds what a randomized policy can achieve.

### 4.3 Photorealistic Rendering

An alternative to randomization is closing the visual gap directly by improving simulation rendering quality.

**Neural Radiance Fields (NeRF) / 3D Gaussian Splatting (3DGS):**
- Scan a real workspace with a camera array or a moving camera
- Fit a NeRF or 3DGS model to the real images
- Use the reconstructed scene as the simulation background

This gives photorealistic backgrounds while maintaining physics simulation for object dynamics. The approach is used in:
- **RealDreamer / SpeedFolding:** NeRF backgrounds for cloth folding policies
- **SIMPLER:** Static scene reconstruction to match BridgeV2 environments

**Limitation:** NeRF/3DGS reconstruction is view-interpolation — it captures the scene as it was during scanning. Dynamic changes (moved objects, different lighting conditions) require re-scanning or augmentation.

### 4.4 Domain Adaptation

Where DR applies at the input distribution level, domain adaptation operates at the feature level.

**CycleGAN / image-to-image translation:**
- Train a CycleGAN to translate simulated images to appear as real images (and vice versa)
- Apply the sim-to-real translation to policy observations at inference
- Challenge: CycleGAN introduces unpredictable artifacts in task-relevant regions (object textures)

**Feature-level adaptation:**
- Train a visual encoder with a domain adversarial loss: encoder features should be indistinguishable between sim and real
- Requires paired or unpaired real data during training
- More stable than pixel-level translation for manipulation-relevant features

In practice, most IL systems today use a combination of **domain randomization** (broad coverage) + **real fine-tuning data** (domain adaptation via gradient descent on a small real dataset) rather than explicit domain adaptation methods. This is simpler and generally more reliable.

---

## 5. Simulation for Benchmark Development

Well-designed IL benchmarks share a common structure that separates signal from noise. When evaluating or creating benchmarks, apply the following criteria:

### Five Properties of a Good IL Benchmark

**1. Diverse tasks with different contact requirements**

A benchmark that only tests pick-and-place favors policies with strong spatial reasoning but weak contact handling. A complete benchmark should span:
- Non-contact reaching and tracking
- Rigid body grasping (power grasp, pinch grasp)
- Tool-use (pushing with a spatula, pouring)
- Articulated object interaction (doors, drawers)
- Bimanual coordination

**2. Clear, quantitative success metrics**

Binary per-episode success is the standard, but the threshold matters:
- "Object in target region within 5 cm" is clearer than "task completed"
- Multi-step tasks should report both final success and intermediate milestone completion
- Metrics should be reproducible across labs (simulator version, randomization seed fixed)

**3. Standardized data format (RLDS)**

The Robot Learning Dataset Specification (RLDS) / TFDS-based format used by Open X-Embodiment enables:
- Cross-benchmark dataset mixing
- Standardized episode structure: `{observation, action, reward, discount, metadata}`
- Compatible with existing data loaders (LeRobot, OpenVLA fine-tuning scripts)

**4. Realistic visual conditions**

Benchmarks with blank white backgrounds do not test visual generalization. Good benchmarks include:
- Textured tables and backgrounds
- Realistic object materials
- Variable lighting

**5. Scalable scripted oracle**

A scripted oracle that can generate demonstrations automatically is essential for benchmarks that need more than ~100 episodes. The oracle should use only information available to the real robot (no cheating on contact forces if the real robot has no force sensor).

---

## 6. Limitations of Simulation

Despite progress in simulator quality and rendering, several categories of real-world manipulation remain poorly served by current simulators.

### Contact-Rich Tasks with Deformable Objects

Cloth folding, cable routing, bag manipulation, and food handling all involve deformable objects that require finite-element or particle-based simulation. Current FEM simulations are too slow for the parallelism that makes IL tractable, and particle-based methods (FLEX, SPH) are not yet accurate enough for tight cable routing or thin fabric. This is why:
- ACT's best-performing real tasks involve rigid objects
- Diffusion Policy cloth-folding experiments require real demonstrations, not simulated ones

### Long-Horizon Tasks Require Non-Trivial Reset Logic

A 10-step task (open drawer → find object → grasp → close drawer → place in container → ...) requires a valid reset to an exact initial state after each episode. In simulation, this requires:
- Exact object placement with all constraints satisfied
- Scripted intermediate state initialization (if evaluating from step 3 onwards)
- Stable simulation initialization without physics instabilities

This engineering burden is underestimated; it can take weeks to build a robust reset system for a 10-step task.

### Distribution Gap Between Scripted and Human Demonstrations

A policy trained on scripted oracle trajectories may learn a different strategy than a human would use. Key differences:

| Property | Scripted Oracle | Human Teleop |
|---|---|---|
| Path efficiency | Near-optimal (A* or waypoint) | Suboptimal, winding |
| Grasp selection | Single deterministic policy | Variable, human-preferred |
| Recovery behavior | No recovery (reset on failure) | Natural recovery attempts |
| Speed profile | Constant velocity | Accelerate / decelerate |
| Contact preference | Minimal contact | Natural support contacts |

Policies trained on scripted data may be brittle to the perturbations that humans naturally handle via recovery behaviors. This is an active research area: generating diverse, human-like scripted demonstrations that include random perturbations and recovery motions is one direction ([Liu et al., 2023](https://arxiv.org/abs/2306.03310)).

### The Fundamental Limitation: Irreducible Reality Gap

Even with photorealistic rendering, GPU-accelerated parallelism, and domain randomization, there is a class of tasks — involving friction, deformability, and precision — for which simulation fidelity is insufficient for zero-shot real-world transfer. For these tasks, simulation is valuable as a **pre-training substrate** that initializes a policy in a reasonable part of weight space, but real-world fine-tuning data (even a small amount: 10–50 demonstrations) is still necessary for reliable deployment.

The practical guideline: use simulation to learn **what to do** (task structure, object affordances, language grounding) and real data to learn **how to do it** (precise contact strategies, friction-dependent grasp adjustments).

---

## Summary

Simulation in imitation learning is not a monolith. Different simulators serve different purposes:
- **MuJoCo** for accurate single-environment physics and standard benchmark compatibility
- **IsaacGym / IsaacLab** for scale: generating millions of demonstrations in hours
- **RLBench / LIBERO / MetaWorld** for standardized evaluation with fair comparison across methods
- **SIMPLER** for sim-to-real correlation measurement

The sim-to-real gap has three components (visual, physical, sensor) and is best addressed through a combination of domain randomization (broadening coverage), photorealistic rendering (narrowing the visual gap), and real fine-tuning data (closing the residual gap). No simulator fully replaces real-world data for contact-rich tasks, but simulation dramatically reduces the amount of real data required.
