# Chapter 8: Evaluation and Benchmarking

> **Who this chapter is for:** Anyone who needs to evaluate an IL policy rigorously—whether for a paper submission, a product milestone, or an internal research review. We cover what metrics to measure, what benchmarks exist, and the pitfalls that make evaluation results in the IL literature hard to compare.

---

## 1. What Does It Mean for a Policy to Be "Good"?

The first step in evaluation is deciding what "good" means for your task. There is no single universal metric; the right choice depends on your application.

### 1.1 Success Rate

The most common metric in manipulation IL:

$$
\text{SR} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\text{episode } i \text{ succeeded}]
$$

Success is binary per episode: either the robot completed the task or it did not. This requires a **task-specific success criterion** that must be defined before running any experiment (to avoid post-hoc cherry-picking).

**Examples of success criteria:**
- *Pick-and-place:* object is within 3 cm of target position at episode end
- *Insertion:* peg is fully inserted (contact sensor triggered, or visual check)
- *Drawer opening:* drawer displaced by at least 80% of its full range

**Statistical considerations:** With $N = 20$ rollouts, a 95% confidence interval for a true success rate of 80% spans approximately $\pm 18\%$ (Wilson interval). Report confidence intervals, not just point estimates.

### 1.2 Task Completion Rate

For long-horizon tasks with natural subtask boundaries, a binary success metric discards useful information. Instead, score each stage:

$$
\text{TCR} = \frac{1}{N \cdot K} \sum_{i=1}^{N} \sum_{k=1}^{K} \mathbf{1}[\text{stage } k \text{ completed in episode } i]
$$

where $K$ is the number of stages. This metric rewards partial progress and is especially useful when comparing two policies where neither achieves >50% binary success.

### 1.3 Recovery Rate

Measures robustness to perturbation:

1. Run the policy until it reaches a fixed waypoint (e.g., object grasped)
2. Apply a perturbation (e.g., nudge the object 3 cm)
3. Measure whether the policy recovers and completes the task

Recovery rate is a leading indicator of real-world robustness. A policy with 95% nominal success rate and 0% recovery rate will fail frequently in deployment.

### 1.4 Generalization Metrics

Split your evaluation objects/positions/backgrounds into:

- **In-distribution (ID):** Same objects and positions seen during training
- **Out-of-distribution (OOD):** Novel objects, new positions, different lighting, cluttered scenes

Report both. A policy that achieves 90% ID and 10% OOD has not generalized. The gap $\text{SR}_{\text{ID}} - \text{SR}_{\text{OOD}}$ quantifies overfitting to the training setup.

---

## 2. Real-Robot Evaluation Protocols

### 2.1 Number of Rollouts

Most IL papers report 20–50 rollouts per task condition. This is not enough to estimate success rates below ~60% with confidence, but it is a practical constraint given the time cost of real-robot evaluation (a 30-second task with robot reset takes ~5 minutes per rollout including setup).

**Recommended minimum:** 20 rollouts for preliminary results, 50 for paper submission, 100 for deployment decisions.

### 2.2 Randomization

Specify exactly how the evaluation environment is randomized:

- **Object position:** random within a fixed region (e.g., 15 cm × 15 cm grid)
- **Object orientation:** random yaw only, or full SO(3)?
- **Initial robot configuration:** fixed home position or random within joint limits?
- **Background clutter:** none, fixed set, or random objects?

Randomization should be documented well enough that another lab can reproduce the evaluation setup from your paper.

### 2.3 Independence

Held-out evaluation objects must never appear in the training dataset. For generalization tests:
- Use a different color/brand of the same object category (e.g., red can instead of silver can)
- Use visually distinct objects from the same semantic category (e.g., different cup shapes)
- Document the specific objects used in training vs. evaluation

### 2.4 Video Recording

Record every evaluation rollout. Videos allow:
- Qualitative failure mode analysis ("the policy consistently misses the left side of the table")
- Reviewer scrutiny (reviewers can spot cherry-picked results)
- Post-hoc analysis when quantitative metrics disagree with intuition

**Checklist for video documentation:**
- Record at least 5 representative successes and all failure modes
- Show the camera views that the policy actually sees (not just a third-person perspective)
- Do not edit out recovery attempts or false starts

---

## 3. Standard Benchmarks

### 3.1 ALOHA Tasks (ACT Benchmark)

Introduced in [Zhao et al., 2023](https://arxiv.org/abs/2304.13705), the ALOHA benchmark uses a low-cost bimanual robot system with six real-world manipulation tasks:

| Task | Description | ACT Success Rate | Demos |
|---|---|---|---|
| Inserting battery | Insert an AA battery into a toy | 80% | 50 |
| Opening translucent cup | Unscrew translucent cup lid | 50% | 50 |
| Threading Velcro | Thread Velcro through a loop | 92% | 50 |
| Slotting battery | Slot battery into a case | 74% | 50 |
| Opening tape dispenser | Open a spring-loaded tape dispenser | 85% | 50 |
| Assembling a pen | Disassemble and reassemble a pen | 35% | 50 |

**Success metric:** Binary, evaluated over 20 rollouts.  
**Randomization:** Object position randomized along a 15 cm line.  
**Key result:** ACT achieves 80–96% on four of six tasks with only 50 demonstrations each—a demonstration efficiency benchmark that subsequent methods are compared against.

**Limitations:** Fixed table, controlled lighting, no background clutter. Results may not transfer to messier environments.

### 3.2 Google Robot / BridgeV2

The BridgeV2 dataset ([Walke et al., 2023](https://arxiv.org/abs/2308.12952)) contains ~60,000 demonstrations across 13 environments and 24 tasks collected on WidowX robots. It is the primary evaluation benchmark for foundation models such as RT-2 and OpenVLA.

**Task categories:**
- Pick-and-place (various object/container combinations)
- Drawer opening/closing
- Object manipulation (stacking, sweeping)

**Evaluation split:** In-distribution vs. out-of-distribution objects, and in-distribution vs. novel task instructions.

**Usage:** [OpenVLA](https://arxiv.org/abs/2406.09246) evaluates on 29 BridgeV2-derived tasks; [RT-2](https://arxiv.org/abs/2307.15818) uses Google's internal robot fleet on related tasks.

### 3.3 SIMPLER

[SIMPLER](https://arxiv.org/abs/2405.05941) (Simulated Manipulation Procedures for Language Evaluation in Robotics) is a simulated version of the BridgeV2 evaluation environments built in MuJoCo.

**Why it matters:** Running 29 tasks × 50 rollouts each on a physical BridgeV2 robot takes days. SIMPLER reduces this to hours on a GPU cluster.

**Correlation with real-world:** SIMPLER shows Spearman rank correlation of ~0.9 with physical BridgeV2 evaluation across multiple methods—high enough to use for architecture search and hyperparameter tuning before committing to physical evaluation.

**Caveat:** Sim-to-real gaps remain. A method that performs well in SIMPLER may still fail in the real lab due to visual domain shift, contact dynamics mismatch, or sensor noise.

### 3.4 OpenVLA Benchmark

[OpenVLA](https://arxiv.org/abs/2406.09246) defines a standardized evaluation of 29 tasks across multiple robot embodiments (WidowX, Google Robot). The key comparison:

| Model | Parameters | BridgeV2 Tasks Success Rate |
|---|---|---|
| RT-2-X | 55B | ~35% |
| OpenVLA-7B | 7B | ~51.5% |
| **Improvement** | **8× smaller** | **+16.5 pp absolute** |

This result demonstrates that scale is not the primary driver of IL performance—data quality, fine-tuning strategy, and action tokenization choices matter more than raw model size.

**Evaluation protocol:** Each task evaluated over 50 rollouts with randomized object positions; success rate averaged over 3 seeds.

### 3.5 RLBench

[RLBench](https://arxiv.org/abs/1909.12271) (James et al., 2020) is a simulation benchmark with 100 tasks built on CoppeliaSim/PyRep. Typically, 18 tasks are selected for evaluation in papers.

**Strengths:**
- Large task diversity (open drawer, put item in drawer, reach for target, etc.)
- Built-in camera configurations (front, left, right, wrist, overhead)
- Keyframe extraction for goal-conditioned policies

**Commonly reported metric:** Per-task success rate over 25 episodes, averaged across the 18-task subset.

**Representative results:**
- PerAct ([Shridhar et al., 2023](https://arxiv.org/abs/2209.05451)): 49.4% mean success over 18 tasks (100 demos/task)
- RVT ([Goyal et al., 2023](https://arxiv.org/abs/2306.14896)): 62.9% mean success over 18 tasks (100 demos/task), 6× faster inference

### 3.6 LIBERO

[LIBERO](https://arxiv.org/abs/2306.03310) (Liu et al., 2023) provides a lifelong learning evaluation framework with four task suites:

| Suite | Focus | Tasks |
|---|---|---|
| LIBERO-Spatial | Spatial reasoning ("left of the bowl") | 10 |
| LIBERO-Object | Object diversity (20+ objects) | 10 |
| LIBERO-Goal | Goal conditioning | 10 |
| LIBERO-Long | Long-horizon (4+ stages) | 10 |

**Evaluation protocol:** Each suite evaluated sequentially—the policy must learn tasks in order without forgetting previous ones. Metrics include both **within-task accuracy** (forward transfer) and **retention** (backward transfer).

**Why it matters:** Most IL benchmarks assume i.i.d. training. LIBERO tests continual adaptation, which is critical for real deployment where you add new tasks over time.

---

## 4. Evaluation Pitfalls

### 4.1 Overfitting to the Evaluation Setup

**Symptom:** The policy achieves 90% success on the reported evaluation but 10% in any other lab.

**Cause:** Implicit overfitting to the specific table, lighting, camera position, and object set used during both data collection and evaluation.

**Detection:** Ask a collaborator to set up an independent evaluation environment without consulting your team's setup.

**Mitigation:** Randomize lighting (change overhead lights, add/remove lamps), vary camera extrinsics slightly between training and evaluation, use a different table surface.

### 4.2 Insufficient Rollouts

**Symptom:** Two methods are reported as 80% vs. 85% success rate—but neither difference is statistically meaningful.

With $N = 20$ rollouts, the standard error of a binomial proportion is:

$$
\text{SE} = \sqrt{\frac{p(1-p)}{N}} \approx \sqrt{\frac{0.5 \times 0.5}{20}} \approx 11\%
$$

A 5-percentage-point difference is noise, not signal.

**Mitigation:** Report Wilson confidence intervals. Use $N \geq 50$ for comparisons you want to claim as statistically significant.

### 4.3 Cherry-Picked Videos

**Symptom:** The paper shows 5 success videos; the failure modes are not discussed.

**Best practice:**
- Show failure videos in the supplementary
- Report which failure modes are responsible for each percentage point of success rate loss
- If you are releasing a video for a conference, show the failure-rate denominator ("4 of 5 attempts shown succeeded")

### 4.4 Simulation-Only Evaluation That Does Not Transfer

**Symptom:** 95% success in simulation, 30% on the physical robot.

**Causes:**
- Visual domain gap (simulated textures vs. real materials)
- Contact dynamics mismatch (simulated friction vs. real rubber gripper)
- Sensor noise not modeled in simulation

**Mitigation:** Use SIMPLER only for ablations and architecture search. Always validate final results on the physical system. When reporting sim results, always note the sim-to-real gap explicitly.

---

## 5. Metrics Beyond Success Rate

Success rate is coarse. For production systems and nuanced research comparisons, complement it with:

### 5.1 Contact Quality

Force profiles during manipulation tasks capture whether the robot is being appropriately gentle or applying dangerous force.

**Metrics:**
- Peak contact force (Newtons) during task execution
- Time-averaged force integral (impulse) at contact events
- Number of force limit violations ($F > F_{\max}$)

Requires a wrist-mounted force-torque sensor or joint torque estimation.

### 5.2 Task Completion Time

For real deployments, speed matters.

$$
\bar{t}_{\text{success}} = \frac{1}{N_{\text{success}}} \sum_{i : \text{success}} t_i
$$

Report mean and standard deviation over successful episodes. A policy that succeeds in 12 seconds is preferable to one that succeeds in 45 seconds, all else equal.

### 5.3 Trajectory Smoothness

Jerky trajectories stress hardware and indicate unstable policies.

**Metrics:**
- **Jerk:** $\dddot{\mathbf{q}}_t$, computed as third derivative of joint angles
- **Mean absolute jerk:** $\frac{1}{T}\sum_t \|\dddot{\mathbf{q}}_t\|_2$
- **Spectral analysis:** Power spectral density of joint velocity; high-frequency content indicates oscillation

```python
import numpy as np

def compute_mean_jerk(joint_positions: np.ndarray, dt: float) -> float:
    """
    joint_positions: (T, n_joints) array of joint angles in radians
    dt: timestep in seconds
    Returns: mean absolute jerk in rad/s^3
    """
    velocity = np.diff(joint_positions, axis=0) / dt      # (T-1, n_joints)
    acceleration = np.diff(velocity, axis=0) / dt          # (T-2, n_joints)
    jerk = np.diff(acceleration, axis=0) / dt              # (T-3, n_joints)
    return np.mean(np.linalg.norm(jerk, axis=1))
```

### 5.4 Safety Metrics

- **Number of collisions per episode** (requires contact detection)
- **Number of joint limit violations**
- **Number of emergency stops triggered**
- **Maximum force applied to object** (if force sensing available)

---

## 6. Ablation Study Design for IL Papers

Ablations are the mechanism by which the research community understands *why* a method works. A weak ablation table undermines an otherwise strong paper.

### 6.1 Key Variables to Ablate for ACT-Style Models

| Variable | Typical range | What you are testing |
|---|---|---|
| Chunk size $k$ | 1, 10, 50, 100 | Value of action chunking; effective horizon reduction |
| CVAE weight $\beta$ | 0, 0.1, 1, 10 | Role of latent variable; $\beta=0$ reduces to pure BC |
| Temporal ensembling | on / off | Value of multi-chunk averaging |
| Number of cameras | 1 (wrist only), 2, 4 | Information contribution of each camera |
| Observation space | images only, qpos only, images + qpos | Value of proprioception |

### 6.2 How to Report Ablations

**Minimum:** Mean success rate over 20 rollouts, 3 random seeds. Report as $\bar{p} \pm \sigma$ where $\sigma$ is the standard deviation across seeds.

**Better:** Wilson 95% confidence interval per condition. Flag pairwise differences that are not statistically significant.

**Example ablation table format:**

| Configuration | Task A SR (%) | Task B SR (%) | Mean SR (%) |
|---|---|---|---|
| Full model | 85 ± 4 | 78 ± 6 | 81.5 |
| No action chunking ($k=1$) | 42 ± 8 | 38 ± 9 | 40.0 |
| No temporal ensembling | 80 ± 5 | 71 ± 7 | 75.5 |
| Single camera (wrist only) | 65 ± 6 | 55 ± 8 | 60.0 |
| No proprioception | 72 ± 5 | 68 ± 7 | 70.0 |
| $\beta = 0$ (pure BC) | 50 ± 7 | 45 ± 8 | 47.5 |

### 6.3 Common Ablation Mistakes

**Ablating only on easy tasks:** If your baseline achieves 95% on Task A, removing a component drops it to 90%—a 5-point drop that is not statistically meaningful with $N=20$. Ablate on tasks where the full model scores 50–80% so there is room to detect degradation.

**Running only one seed:** Random initialization and demonstration order can swing success rates by 10–15 points on small datasets. Three seeds is a minimum; five is better.

**Not ablating the most important component:** If your paper's main contribution is X, you must ablate X. Ablating only secondary components invites reviewer skepticism.

---

## Summary

Rigorous evaluation requires care at every step:

1. **Define metrics before running experiments** to avoid post-hoc selection bias.
2. **Standard benchmarks** (ALOHA, BridgeV2, SIMPLER, RLBench, LIBERO) enable comparison across papers, but each has specific limitations.
3. **Statistical power** requires at least 20–50 rollouts; report confidence intervals alongside point estimates.
4. **Generalization evaluation** must use held-out objects, positions, and environments—not just held-out rollouts.
5. **Ablations** should vary one factor at a time over 3+ seeds and should target the method's core claims.
6. **Simulation** is useful for iteration but must be validated on physical hardware before publication.

**Next:** Chapter 9 covers deploying a trained IL policy on a physical robot system—hardware, latency, failure modes, and production considerations.

---

*Sources: [ACT (Zhao et al., 2023)](https://arxiv.org/abs/2304.13705) · [OpenVLA (Kim et al., 2024)](https://arxiv.org/abs/2406.09246) · [BridgeV2 (Walke et al., 2023)](https://arxiv.org/abs/2308.12952) · [SIMPLER (Li et al., 2024)](https://arxiv.org/abs/2405.05941) · [RLBench (James et al., 2020)](https://arxiv.org/abs/1909.12271) · [LIBERO (Liu et al., 2023)](https://arxiv.org/abs/2306.03310) · [PerAct (Shridhar et al., 2023)](https://arxiv.org/abs/2209.05451)*
