# Chapter 9: Deployment and Real-World Considerations

> **Who this chapter is for:** Engineers who have a trained IL policy and need to run it on a physical robot. We cover the full deployment stack—hardware selection, control loop design, latency budgets, failure modes, and continuous improvement. Theory is kept to a minimum; operational detail is the focus.

---

## 1. The Real-World Deployment Gap

Every IL practitioner eventually discovers that a policy achieving 85% success in the training environment can fall to 20% in a slightly different room. This gap has several named sources:

### 1.1 Visual Distribution Shift

The policy's vision backbone was trained on images from a specific camera, lighting setup, and background. Even small changes break performance:

| Change | Typical SR drop | Mitigation |
|---|---|---|
| Different lighting (overhead vs. daylight) | 20–40 pp | Diverse lighting during data collection |
| Novel object color/texture | 30–60 pp | VLA backbone with pretrained features |
| Background clutter | 10–30 pp | Augmentation (random background, ColorJitter) |
| Camera position drift (~1 cm) | 5–20 pp | Camera calibration maintenance + recalibration protocol |

### 1.2 Mechanical Variation

Physical robots drift. Joints develop backlash, gripper compliance changes as rubber wears, motor offsets shift after collisions. A policy trained before a maintenance cycle may fail after.

**Mitigation:** Collect a small set of fresh demonstrations (5–10) after any hardware change and fine-tune the policy. Keep a calibration object in the workspace to detect drift early.

### 1.3 Demonstration Nonstationarity

The people demonstrating a task in deployment may use different styles than those in the original training set. This is particularly acute when scaling to many operators.

**Mitigation:** Screen demonstrators before data collection, provide standardized task instructions, and monitor data quality with held-out evaluation rollouts after each new batch.

---

## 2. Hardware Stack

### 2.1 Robot Arms

The right arm choice depends on your budget, required payload, and community support:

| Robot | Cost (USD) | DoF | Payload | IL community support | Notes |
|---|---|---|---|---|---|
| WidowX 250 | `~$3,000` | 6 | 250 g | High (BridgeV2) | Low-cost research standard |
| ViperX 300 | `~$5,500` | 6 | 750 g | High (BridgeV2) | More payload than WidowX |
| Franka Panda | `~$25,000` | 7 | 3 kg | Very high (research standard) | Torque control, force sensing built-in |
| Universal Robots UR5 | `~$35,000` | 6 | 5 kg | Moderate | Industrial reliability, no direct torque control |
| UFACTORY xArm 6 | `~$8,000` | 6 | 5 kg | Growing | Good value; xArm SDK well documented |
| ALOHA (custom) | `~$20,000` | 14 (bimanual) | 2×500 g | High (ACT papers) | Low-Dynamixel-based bimanual system |

**Key recommendation:** For new projects, Franka Panda is the research standard if budget allows. For low-cost work, WidowX + BridgeV2 data gives access to a large pretrained-model ecosystem.

### 2.2 Camera Placement

Camera placement is among the most consequential decisions in your hardware design. Once the robot is assembled and data collection begins, changing cameras means recollecting all demonstrations.

**Standard configuration for ACT-class tasks:**
- **Top-down camera** (mounted on overhead arm or ceiling): provides workspace overview
- **Front-facing camera** (eye-level, 60–90 cm away): captures approach trajectories
- **Wrist camera(s)**: critical for fine manipulation; captures the gripper–object interface at close range

```
        [Top camera]
              |
    _______[Front camera]________
   |                              |
   |      [Robot arm]             |
   |         |                   |
   |      [Wrist camera]         |
   |_____________________________| 
```

**Wrist cameras** are the highest-information source for dexterous tasks. If you can only use one camera, make it a wrist camera.

### 2.3 Real-Time Control Stack

Your software stack must meet hard real-time constraints. The three main options:

**ROS2 (recommended for new projects)**
- Industry-standard middleware
- Real-time capable with DDS configured for low-latency transport
- Extensive driver ecosystem (Franka ROS2, Universal Robots ROS2 driver, Dynamixel SDK)
- Overhead: ~2–5 ms per message; sufficient for 50 Hz control

**Custom UDP**
- Lowest latency, no middleware overhead
- Used by high-performance systems where ROS2 latency is unacceptable
- Requires you to implement reliability, time synchronization, and introspection yourself

**Dynamixel SDK (for Dynamixel-based systems like ALOHA)**
- Direct USB-to-Dynamixel communication at up to 1000 Hz
- ACT uses this at 50 Hz (20 ms control period)

### 2.4 Compute Options

| Platform | GPU | Typical IL inference latency | Use case |
|---|---|---|---|
| NVIDIA RTX 4090 workstation | RTX 4090 (24 GB) | 5–15 ms (ACT/diffusion) | Lab research, co-located with robot |
| NVIDIA Jetson AGX Orin | Ampere (64 GB) | 20–80 ms | Edge deployment, on-robot |
| Cloud GPU (A100) | A100 (80 GB) | 50–200 ms + network | Not recommended for real-time control |
| MacBook M2 Pro | Neural Engine | 40–120 ms (small models) | Prototyping only |

**Practical guidance:** For ACT and diffusion-based policies (< 1B parameters), an RTX 3090 or 4090 is sufficient and cost-effective. For VLA inference (7B+ parameters), a 4090 or A100 is required.

---

## 3. The ROS2 Integration

Here is a minimal but production-quality ROS2 inference node for an ACT-style policy:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import torch
import numpy as np
from collections import deque


class ACTPolicyNode(Node):
    def __init__(self):
        super().__init__('act_policy')

        # ── Subscribers ──────────────────────────────────────────────────────
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        # 4-camera setup: top, front, left_wrist, right_wrist
        self.bridge = CvBridge()
        self.camera_topics = [
            '/camera_top/image_raw',
            '/camera_front/image_raw',
            '/camera_wrist_left/image_raw',
            '/camera_wrist_right/image_raw',
        ]
        self.image_subs = [
            self.create_subscription(
                Image, topic, 
                lambda msg, i=i: self.image_callback(msg, i), 
                10)
            for i, topic in enumerate(self.camera_topics)
        ]

        # ── Publisher ─────────────────────────────────────────────────────────
        self.publisher = self.create_publisher(
            JointTrajectory, '/target_joints', 10)

        # ── Policy ────────────────────────────────────────────────────────────
        self.policy = ACTPolicy.load('model.ckpt')
        self.policy.eval()
        self.device = torch.device('cuda')
        self.policy.to(self.device)

        # ── State ─────────────────────────────────────────────────────────────
        self.current_qpos = None
        self.current_images = [None] * len(self.camera_topics)
        self.action_queue = deque()  # temporal ensembling buffer

        # ── Control loop at 50 Hz ─────────────────────────────────────────────
        self.timer = self.create_timer(0.02, self.inference_step)
        self.get_logger().info('ACT policy node initialized at 50 Hz.')

    def joint_callback(self, msg: JointState):
        self.current_qpos = np.array(msg.position)

    def image_callback(self, msg: Image, camera_idx: int):
        self.current_images[camera_idx] = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='rgb8')

    def _ready(self) -> bool:
        return (self.current_qpos is not None and 
                all(img is not None for img in self.current_images))

    def inference_step(self):
        if not self._ready():
            return

        if len(self.action_queue) == 0:
            # Inference: predict a new action chunk
            with torch.no_grad():
                obs = self._prepare_obs()
                action_chunk = self.policy(obs)  # shape: (chunk_size, action_dim)
            self.action_queue.extend(action_chunk.cpu().numpy())

        # Execute next action from queue
        action = self.action_queue.popleft()  # shape: (action_dim,)
        self._publish_action(action)

    def _prepare_obs(self) -> dict:
        """Convert raw sensor data to policy input tensors."""
        images = torch.stack([
            self._preprocess_image(img) 
            for img in self.current_images
        ]).unsqueeze(0).to(self.device)  # (1, N_cams, C, H, W)

        qpos = torch.tensor(
            self.current_qpos, dtype=torch.float32
        ).unsqueeze(0).to(self.device)  # (1, n_joints)

        return {'images': images, 'qpos': qpos}

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        import torchvision.transforms.functional as TF
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = TF.resize(img_t, [224, 224])
        img_t = TF.normalize(img_t, 
                             mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        return img_t  # (3, 224, 224)

    def _publish_action(self, action: np.ndarray):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        point = JointTrajectoryPoint()
        point.positions = action.tolist()
        point.time_from_start.nanosec = 20_000_000  # 20 ms (50 Hz)
        msg.points.append(point)
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = ACTPolicyNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

**Key design decisions in the node above:**
- **Timer-driven control:** The 50 Hz timer ensures consistent control frequency regardless of message arrival times.
- **Queue-based action execution:** The action queue decouples inference (which may take 10–15 ms) from execution (which must happen every 20 ms).
- **Lazy inference:** A new chunk is only computed when the queue empties, minimizing unnecessary computation.

---

## 4. Latency and Control Loop Design

Latency determines your maximum control frequency, which determines how responsive the policy can be to real-world feedback.

### 4.1 Latency Budget by Policy Type

| Policy class | Single forward pass | Control frequency | Notes |
|---|---|---|---|
| ACT (CVAE + Transformer) | 8–15 ms | 50 Hz | Fits comfortably in a 20 ms control period |
| Diffusion (DDPM, 100 steps) | ~200 ms | ~5 Hz | Too slow for fast tasks; use DDIM |
| Diffusion (DDIM, 10 steps) | 15–25 ms | ~40 Hz | Practical for most manipulation tasks |
| VLA 7B (e.g., OpenVLA) | 200–500 ms | 2–5 Hz | Requires hierarchical decomposition |
| VLA 55B (e.g., RT-2-X) | ~2 s | < 1 Hz | Cloud inference only; not suitable for reactive tasks |

### 4.2 Hierarchical Decomposition for Slow Policies

For VLAs that run at 2–5 Hz, a two-level architecture separates slow high-level reasoning from fast low-level execution:

```python
class HierarchicalController:
    """
    High-level VLA plan at 2 Hz; low-level primitive executes at 50 Hz.
    """
    def __init__(self, vla_policy, low_level_controller):
        self.vla = vla_policy
        self.ll = low_level_controller
        self.current_subgoal = None

    def run(self, obs_stream):
        for step, obs in enumerate(obs_stream):
            # VLA plans every 25 steps (= 0.5 s at 50 Hz)
            if step % 25 == 0:
                self.current_subgoal = self.vla.plan(obs)  # slow, 200–500 ms

            # Low-level executes at 50 Hz toward current subgoal
            action = self.ll.step_toward(obs, self.current_subgoal)
            yield action
```

### 4.3 Measuring Your Latency

Always profile your actual deployment stack, not just the model's forward pass:

```python
import time

def measure_inference_latency(policy, obs, n_trials=100):
    """Measure wall-clock latency including GPU synchronization."""
    latencies = []
    for _ in range(n_trials):
        torch.cuda.synchronize()  # ensure previous ops are done
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = policy(obs)
        torch.cuda.synchronize()  # wait for this op to finish
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    latencies = latencies[10:]  # discard warmup
    print(f"Latency: {np.mean(latencies):.1f} ± {np.std(latencies):.1f} ms "
          f"(p95: {np.percentile(latencies, 95):.1f} ms)")
```

**Gotcha:** A model with a mean latency of 15 ms but a p95 of 35 ms will miss its 20 ms deadline 5% of the time, causing observable stuttering. Always check percentiles, not just means.

---

## 5. Failure Modes and Mitigations

Understanding failure modes *before* deployment lets you design mitigations in advance rather than debugging them under time pressure.

### 5.1 Distribution Shift Failure

**Symptom:** Policy works reliably in the training setup but fails immediately when any environmental variable changes—lighting, object color, table surface, camera angle.

**Diagnosis:** Compare the feature activations (or action outputs) on training images vs. deployment images. If they differ substantially, distribution shift is the cause.

**Mitigations:**
- **Diverse data collection:** Collect demonstrations under multiple lighting conditions, with multiple object instances, at multiple table positions.
- **Visual augmentation:** Apply random color jitter, brightness, contrast, and background replacement during training (not just during evaluation).
- **VLA backbone:** Use a pretrained vision-language backbone (ResNet-50 → ViT or SigLIP) that has seen many more visual conditions than your robot dataset.

### 5.2 Compounding Errors on Long Horizons

**Symptom:** Policy executes the first stage correctly, then drifts during the second stage, and has completely lost track by the third stage.

**Diagnosis:** Monitor the per-stage success rate. If stage 1 = 90%, stage 2 = 50%, stage 3 = 20%, compounding error is the pattern.

**Mitigations:**
- **Action chunking:** Increase chunk size $`k`$ to reduce decision frequency.
- **Temporal ensembling:** Smooth actions across overlapping chunks.
- **Hierarchical IL:** Train a separate high-level policy for stage transitions and per-stage low-level policies. Each low-level policy operates over a shorter horizon.
- **Reset policy:** Train a recovery policy that can re-establish a canonical state (e.g., put the object back in the tray) when a stage fails.

### 5.3 Contact and Grasp Failures

**Symptom:** The robot consistently misses the object by a few millimeters, or grasps at the wrong orientation, or drops objects after picking.

**Diagnosis:** Look at wrist camera frames at the moment of contact. If the camera shows the object clearly but the gripper approaches from the wrong angle, the policy is using insufficient depth information.

**Mitigations:**
- **Wrist cameras:** Add or improve wrist-mounted cameras. They provide the most precise spatial signal for grasp.
- **Targeted demonstrations:** Collect extra demonstrations specifically at the contact moments (approach and insert). These are typically underrepresented relative to transit phases.
- **Force sensing:** If available, add wrist force/torque signals as an observation. Bi-ACT ([Chen et al., 2024](https://arxiv.org/abs/2408.05981)) shows this significantly improves contact-rich assembly.
- **Grasp verification:** Add a post-grasp verification step: after attempting a grasp, check a visual criterion (gripper closed, object visible in wrist camera) and retry if it fails.

### 5.4 Recovery Strategies

When a policy fails mid-task, you need a defined response:

**Option A: SAFE (Stop and Ask For Expert)**  
Detect out-of-distribution states using a learned uncertainty estimator. When uncertainty exceeds a threshold, halt and request human intervention. The SAFE framework ([SAFE, 2025](https://arxiv.org/abs/2506.09937)) formalizes this: a lightweight anomaly detector runs in parallel with the policy and triggers a safe stop when the policy is likely to compound an error.

**Option B: Safe DAgger**  
Run the policy but query the human expert whenever the policy's confidence is low (e.g., the action distribution has high entropy or the CVAE KL divergence is large). The human's actions are recorded and added to the training dataset.

**Option C: Language-Conditioned Retry**  
If the policy is language-conditioned (e.g., a VLA), issue a different instruction on failure: "Try again from the left side" or "Pick up the object more carefully." This requires a language-conditioned policy and a failure detector.

---

## 6. Continuous Improvement in Deployment

A deployed robot policy is not static. The data flywheel—systematic collection of new data from deployment, retraining, and redeployment—is how real-world IL systems improve over time.

### 6.1 The Data Flywheel

```
Deploy policy → Observe failures → Collect targeted demos →
Retrain or fine-tune → Redeploy → Repeat
```

**Practical cadence:** Run the policy for one week. Review failure videos. Identify the top 2–3 failure modes. Collect 20–50 demonstrations specifically targeting those modes. Fine-tune for 100–500 gradient steps. Redeploy and measure improvement.

### 6.2 Online DAgger

In online DAgger, a human expert monitors the policy in real time and can take over the robot to correct a mistake. The correction trajectory is recorded and added to the training dataset.

```python
class OnlineDAggerCollector:
    """
    Monitors policy actions; human can intervene at any time.
    Intervention trajectories are saved to the dataset.
    """
    def __init__(self, policy, robot, dataset_path):
        self.policy = policy
        self.robot = robot
        self.dataset_path = dataset_path
        self.is_human_controlling = False
        self.intervention_buffer = []

    def run_episode(self):
        obs = self.robot.reset()
        episode_data = []

        for t in range(MAX_EPISODE_LENGTH):
            if not self.is_human_controlling:
                action = self.policy(obs)
            else:
                action = self.robot.read_human_input()  # from leader arm
                self.intervention_buffer.append((obs, action))

            next_obs, done = self.robot.step(action)
            episode_data.append((obs, action))
            obs = next_obs
            if done:
                break

        # Save intervention data
        if self.intervention_buffer:
            self._save_to_dataset(self.intervention_buffer)
            self.intervention_buffer = []
```

### 6.3 Autonomous Improvement

For tasks where success/failure can be verified automatically (e.g., a sensor detects whether the peg was inserted), the policy can collect its own successful trajectories:

1. Run the policy autonomously for $`N`$ episodes
2. Detect success via sensor or visual check
3. Add successful trajectories to the training dataset
4. Retrain on the augmented dataset

This is a limited form of RL (using binary success as reward) without the instability of full RL. It works best when the policy already achieves >30% success, so successful trajectories accumulate at a reasonable rate.

---

## 7. Multi-Robot Deployment

### 7.1 Cross-Embodiment Challenges

Deploying a single policy across multiple robot types requires reconciling incompatible action and observation spaces:

| Difference | Challenge | Solution |
|---|---|---|
| Different DoF (6 vs. 7 joint) | Action vectors have different dimensions | Embodiment-specific output heads |
| Different camera configurations | Observation tensors differ in shape | Embodiment-specific image encoders |
| Different joint limits | Same absolute positions mean different poses | Normalize to [-1, 1] within each robot's limits |
| Different gripper designs | Gripper action semantics differ | Binary open/close abstraction |

### 7.2 OpenVLA and Octo Approaches

Both [OpenVLA](https://arxiv.org/abs/2406.09246) and [Octo](https://arxiv.org/abs/2405.12213) are designed for multi-embodiment deployment.

**OpenVLA** tokenizes actions as discrete string tokens (e.g., the action `[0.02, -0.01, 0.05]` becomes the string `"0.020 -0.010 0.050"`). Different robot embodiments can use different tokenization ranges mapped to the same token vocabulary.

**Octo** uses a transformer architecture with embodiment-specific input/output adapters. The core model weights are shared; per-embodiment projection layers are trained separately.

```python
# Octo-style embodiment-specific projection (pseudocode)
class OctoHead(nn.Module):
    def __init__(self, hidden_dim, action_dim, embodiment_id):
        super().__init__()
        # Shared backbone (frozen or lightly fine-tuned)
        self.backbone = OctoBackbone(hidden_dim)
        # Per-embodiment output projection
        self.projectors = nn.ModuleDict({
            'widowx': nn.Linear(hidden_dim, 6),    # 6-DoF arm
            'franka': nn.Linear(hidden_dim, 7),    # 7-DoF arm
            'aloha':  nn.Linear(hidden_dim, 14),   # bimanual
        })
        self.embodiment_id = embodiment_id

    def forward(self, obs_tokens):
        features = self.backbone(obs_tokens)
        return self.projectors[self.embodiment_id](features)
```

---

## 8. Production Deployment Checklist

Use this checklist before any deployment to a new environment, after any hardware change, or before any user-facing demonstration.

1. **Safety review**
   - Verify joint position limits are enforced in software (not just hardware stops)
   - Verify joint velocity limits are enforced (maximum joint speed in rad/s per joint)
   - Verify workspace bounds: robot cannot reach humans or fragile objects in the workspace
   - Verify force limits: emergency stop triggers if wrist force exceeds $`F_{\max}`$ (e.g., 30 N)
   - Test emergency stop button manually

2. **Camera calibration**
   - Verify intrinsic calibration (focal length, distortion coefficients) for each camera using a checkerboard
   - Verify extrinsic calibration (camera pose relative to robot base) using a known calibration object
   - Check that calibration is consistent across sessions (re-run calibration if camera was moved)

3. **Baseline evaluation**
   - Run ≥20 evaluation rollouts with the standard object configuration
   - Confirm success rate is within 5 pp of the last recorded baseline
   - If success rate has dropped, investigate before proceeding (likely hardware drift or calibration issue)

4. **Failure mode documentation**
   - Maintain a failure log: date, failure mode, object configuration, recovery action taken
   - After any change (new demos, fine-tuning), run a fresh baseline evaluation and compare failure modes

5. **Monitoring and telemetry setup**
   - Log all joint positions, actions, and camera images at the native control frequency
   - Set up alerts for: excessive force, joint limit approach (within 5% of limit), inference latency > 2× baseline
   - Store logs with timestamps for post-hoc debugging

6. **Human oversight protocol**
   - Define who is responsible for monitoring the robot during deployment
   - Define the intervention procedure (how to take manual control, how to emergency stop)
   - Define the escalation path if a novel failure mode is observed
   - Ensure all monitoring personnel have completed safety training before operating the system

---

## Summary

Deployment is where IL systems meet the real world—and where theoretical guarantees break down in favor of operational discipline:

1. **Distribution shift** is the primary failure mode; address it with diverse data, augmentation, and where possible, VLA backbones with broad visual priors.
2. **Hardware selection** (robot arm, cameras, compute) is a long-lived decision—get it right before data collection begins.
3. **Latency budgets** determine your viable control frequency; profile your full stack (not just model forward pass) and choose your policy architecture accordingly.
4. **Failure modes** should be anticipated and documented, with mitigations prepared before they occur in front of a user.
5. **Continuous improvement** via the data flywheel is how deployed systems get better over time; design your logging and data pipeline to support it from day one.
6. **The production checklist** (Section 8) should be reviewed before every deployment to a new environment.

---

*Sources: [ACT (Zhao et al., 2023)](https://arxiv.org/abs/2304.13705) · [SAFE (Bousmalis et al., 2025)](https://arxiv.org/abs/2506.09937) · [OpenVLA (Kim et al., 2024)](https://arxiv.org/abs/2406.09246) · [Octo (Team et al., 2024)](https://arxiv.org/abs/2405.12213)*
