# GUARDIAN Implementation Guide

## Phase 1: Foundation Setup (Weeks 1-4)

### Week 1: Environment Setup

```bash
# 1. Install Isaac Lab
cd ~/
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install nvidia-warp openfl ray[serve]
pip install opencv-python pillow scipy scikit-learn

# 3. Setup ROS2 (Ubuntu 22.04)
sudo apt install ros-humble-desktop
sudo apt install ros-humble-moveit ros-humble-ros2-control

# 4. Clone baseline VLA (TinyVLA or OpenVLA)
git clone https://github.com/TinyVLA/TinyVLA.git
cd TinyVLA
pip install -e .
```

### Week 2: Data Collection Infrastructure

**File: `guardian/data_collection/recorder.py`**

```python
import rospy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import WrenchStamped
import cv2
import h5py
import numpy as np
from datetime import datetime

class GuardianDataRecorder:
    """Records multi-modal sensor data for GUARDIAN training"""

    def __init__(self, output_dir="/data/guardian"):
        self.output_dir = output_dir
        self.buffer = {
            'rgb': [],
            'depth': [],
            'joints_pos': [],
            'joints_vel': [],
            'wrench': [],
            'timestamps': []
        }

        # ROS subscribers
        rospy.Subscriber('/camera/rgb/image', Image, self.rgb_callback)
        rospy.Subscriber('/camera/depth/image', Image, self.depth_callback)
        rospy.Subscriber('/joint_states', JointState, self.joints_callback)
        rospy.Subscriber('/wrist/wrench', WrenchStamped, self.wrench_callback)

        self.episode_count = 0

    def rgb_callback(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.buffer['rgb'].append(img)
        self.buffer['timestamps'].append(rospy.Time.now().to_sec())

    def depth_callback(self, msg):
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        self.buffer['depth'].append(depth)

    def joints_callback(self, msg):
        self.buffer['joints_pos'].append(np.array(msg.position))
        self.buffer['joints_vel'].append(np.array(msg.velocity))

    def wrench_callback(self, msg):
        force = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
        torque = [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]
        self.buffer['wrench'].append(force + torque)

    def save_episode(self, success, failure_type=None, failure_time=None):
        """Save episode to HDF5"""
        filename = f"{self.output_dir}/episode_{self.episode_count:05d}.h5"

        with h5py.File(filename, 'w') as f:
            # Save trajectories
            f.create_dataset('rgb', data=np.stack(self.buffer['rgb']))
            f.create_dataset('depth', data=np.stack(self.buffer['depth']))
            f.create_dataset('joints_pos', data=np.stack(self.buffer['joints_pos']))
            f.create_dataset('joints_vel', data=np.stack(self.buffer['joints_vel']))
            f.create_dataset('wrench', data=np.stack(self.buffer['wrench']))
            f.create_dataset('timestamps', data=np.array(self.buffer['timestamps']))

            # Save labels
            f.attrs['success'] = success
            if not success:
                f.attrs['failure_type'] = failure_type
                f.attrs['failure_time'] = failure_time

        # Clear buffer
        for key in self.buffer:
            self.buffer[key] = []

        self.episode_count += 1
        print(f"Saved episode {self.episode_count}")
```

**File: `guardian/data_collection/labeler.py`**

```python
import h5py
import cv2
import tkinter as tk
from tkinter import ttk

class FailureLabelingGUI:
    """GUI for labeling failure types and timestamps"""

    FAILURE_TYPES = ['collision', 'wrong_object', 'grasp_failure', 'goal_miss', 'timeout']

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.current_episode = 0

        # Create GUI
        self.root = tk.Tk()
        self.root.title("GUARDIAN Failure Labeler")

        # Video player
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        # Controls
        self.slider = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL,
                              command=self.on_slider)
        self.slider.pack(fill=tk.X)

        # Failure type selector
        self.failure_var = tk.StringVar()
        self.failure_menu = ttk.Combobox(self.root, textvariable=self.failure_var,
                                         values=self.FAILURE_TYPES)
        self.failure_menu.pack()

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Mark Failure", command=self.mark_failure).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Success", command=self.mark_success).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Next Episode", command=self.next_episode).pack(side=tk.LEFT)

        self.load_episode(0)

    def load_episode(self, idx):
        """Load episode from HDF5"""
        filename = f"{self.data_dir}/episode_{idx:05d}.h5"
        with h5py.File(filename, 'r') as f:
            self.rgb_frames = f['rgb'][:]
            self.timestamps = f['timestamps'][:]

        self.current_frame = 0
        self.slider.config(to=len(self.rgb_frames)-1)
        self.show_frame(0)

    def show_frame(self, idx):
        """Display frame on canvas"""
        frame = cv2.cvtColor(self.rgb_frames[idx], cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        # Convert to PhotoImage and display
        # ... (implementation details)

    def mark_failure(self):
        """Save failure label"""
        failure_type = self.failure_var.get()
        failure_time = self.timestamps[self.current_frame]

        filename = f"{self.data_dir}/episode_{self.current_episode:05d}.h5"
        with h5py.File(filename, 'a') as f:
            f.attrs['success'] = False
            f.attrs['failure_type'] = failure_type
            f.attrs['failure_time'] = failure_time

        print(f"Labeled episode {self.current_episode} as {failure_type} at t={failure_time:.2f}s")

    def on_slider(self, val):
        self.current_frame = int(val)
        self.show_frame(self.current_frame)
```

### Week 3: VLA Integration & Signal Extraction

**File: `guardian/models/vla_wrapper.py`**

```python
import torch
import torch.nn as nn
from tinyvla import TinyVLA  # or OpenVLA

class VLAWithInternals(nn.Module):
    """Wrapper around VLA that exposes internal signals for prediction"""

    def __init__(self, vla_checkpoint, ensemble_size=5):
        super().__init__()

        # Load ensemble of VLA models (for model uncertainty)
        self.ensemble = nn.ModuleList([
            TinyVLA.from_pretrained(vla_checkpoint) for _ in range(ensemble_size)
        ])

        # Freeze all models
        for model in self.ensemble:
            for param in model.parameters():
                param.requires_grad = False

        self.ensemble_size = ensemble_size

    def forward(self, obs, language, return_internals=False):
        """
        Args:
            obs: dict with 'rgb' (B, 3, H, W), 'proprio' (B, D)
            language: str or List[str]
            return_internals: bool - whether to return attention maps, hidden states

        Returns:
            actions: (B, ensemble_size, action_dim)
            internals: dict with 'attention', 'hidden', 'action_variance'
        """
        actions = []
        attentions = []
        hiddens = []

        for model in self.ensemble:
            # Forward pass with hooks to capture internals
            output = model(obs, language, output_attentions=True, output_hidden_states=True)
            actions.append(output.action)
            attentions.append(output.attentions[-1])  # Last layer attention
            hiddens.append(output.hidden_states[-2])  # Penultimate layer

        actions = torch.stack(actions, dim=1)  # (B, ensemble_size, action_dim)

        if return_internals:
            internals = {
                'attention': torch.stack(attentions, dim=1),  # (B, K, num_heads, seq_len, seq_len)
                'hidden': torch.stack(hiddens, dim=1),  # (B, K, seq_len, hidden_dim)
                'action_mean': actions.mean(dim=1),
                'action_variance': actions.var(dim=1)  # Model uncertainty
            }
            return actions, internals
        else:
            return actions.mean(dim=1)  # Return ensemble mean action
```

**File: `guardian/models/signal_extractor.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FailureSignalExtractor:
    """Extracts 4 signal modalities for failure prediction"""

    def __init__(self, vla_model, trajectory_encoder):
        self.vla = vla_model
        self.trajectory_encoder = trajectory_encoder

    def extract_signals(self, obs, language, history, depth_image):
        """
        Extract 12D feature vector for predictor

        Args:
            obs: Current observation
            language: Task description
            history: List of past (obs, action) tuples (last 10 steps)
            depth_image: (H, W) depth map

        Returns:
            features: (12,) tensor
        """
        # Get VLA internals
        actions, internals = self.vla(obs, language, return_internals=True)

        # 1. Model Uncertainty (2 features)
        sigma2_ensemble = internals['action_variance'].mean().item()
        sigma2_action = actions.var(dim=1).mean().item()

        # 2. Attention Degradation (2 features)
        attention = internals['attention'].mean(dim=1)  # Average over ensemble
        H_attention = self.compute_entropy(attention)
        d_attention = self.compute_misalignment(attention, obs)

        # 3. Trajectory Divergence (2 features)
        traj_embedding = self.trajectory_encoder(history)  # (D,)
        nominal_embedding = self.get_nominal_embedding()  # Cluster center from training
        d_traj = torch.norm(traj_embedding - nominal_embedding).item()

        # Placeholder for learned divergence (train encoder later)
        d_learned = 0.0  # Will be replaced after training

        # 4. Environmental Risk (4 features)
        d_obs = self.compute_obstacle_distance(depth_image)
        gripper_force = obs['proprio'][-6:]  # Last 6 dims are wrench
        F_gripper = torch.norm(gripper_force).item()
        occlusion = self.compute_occlusion(depth_image, obs['rgb'])

        # Concatenate all signals
        features = torch.tensor([
            sigma2_ensemble, sigma2_action,  # Epistemic
            H_attention, d_attention,  # Attention
            d_traj, d_learned,  # Trajectory
            d_obs, F_gripper, occlusion, 0, 0, 0  # Environment (padding to 12D)
        ], dtype=torch.float32)

        return features

    def compute_entropy(self, attention):
        """Compute entropy of attention distribution"""
        # attention: (num_heads, seq_len, seq_len)
        # Average over heads and normalize
        attn_avg = attention.mean(dim=0)  # (seq_len, seq_len)
        attn_probs = F.softmax(attn_avg, dim=-1)
        entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1).mean()
        return entropy.item()

    def compute_obstacle_distance(self, depth_image):
        """Compute minimum distance to obstacles from depth map"""
        # Assuming depth is in meters, robot mask is provided
        robot_mask = self.get_robot_mask(depth_image.shape)
        obstacle_depths = depth_image[~robot_mask]
        min_distance = obstacle_depths.min().item() if len(obstacle_depths) > 0 else 10.0
        return min_distance

    def compute_occlusion(self, depth_image, rgb_image):
        """Estimate occlusion fraction of target object"""
        # Requires object segmentation (use Mask R-CNN or SAM)
        # Placeholder: return random value for now
        return 0.0
```

### Week 4: Isaac Lab Simulation Setup

**File: `guardian/sim/isaac_tasks.py`**

```python
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
import torch

@configclass
class G1PickPlaceEnvCfg(DirectRLEnvCfg):
    """Configuration for Unitree G1 pick-and-place tasks"""

    # Environment settings
    episode_length_s = 120.0
    decimation = 4  # 50Hz control from 200Hz sim
    num_envs = 16

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=4.0
    )

    # Randomization (domain randomization)
    randomization = {
        'object_pose': {
            'position': [-0.1, 0.1],  # ±10cm
            'orientation': [-30, 30]   # ±30 degrees
        },
        'lighting': {
            'intensity': [100, 2000]  # lux
        },
        'friction': {
            'static': [0.3, 1.2],
            'dynamic': [0.3, 1.2]
        },
        'object_mass': {
            'scale': [0.8, 1.2]  # ±20%
        }
    }

class G1PickPlaceEnv(DirectRLEnv):
    """Isaac Lab environment for G1 humanoid pick-and-place"""

    cfg: G1PickPlaceEnvCfg

    def __init__(self, cfg: G1PickPlaceEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Track failure types
        self.failure_log = []

    def _reset_idx(self, env_ids):
        """Reset specific environments"""
        # Randomize object poses
        obj_pos = self.sample_uniform(
            len(env_ids), 3,
            self.cfg.randomization['object_pose']['position']
        )
        self.scene['object'].set_world_poses(obj_pos, env_ids=env_ids)

        # Randomize physics
        friction = self.sample_uniform(
            len(env_ids), 1,
            self.cfg.randomization['friction']['static']
        )
        self.scene['table'].set_friction(friction, env_ids=env_ids)

    def _get_observations(self):
        """Return observations dict"""
        return {
            'rgb': self.scene.camera.get_rgb(),
            'depth': self.scene.camera.get_depth(),
            'proprio': torch.cat([
                self.scene.robot.get_joint_positions(),
                self.scene.robot.get_joint_velocities(),
                self.scene.robot.get_end_effector_wrench()
            ], dim=-1)
        }

    def _compute_failure(self):
        """Check for failures and log type"""
        # Collision detection
        contact_forces = self.scene.robot.get_contact_forces()
        collision = (contact_forces > 10.0).any(dim=-1)  # 10N threshold

        # Wrong object detection
        gripper_obj_id = self.scene.robot.get_grasped_object_id()
        wrong_object = (gripper_obj_id != self.target_object_id) & (gripper_obj_id != -1)

        # Grasp failure (object dropped)
        object_pos = self.scene['object'].get_world_poses()[0]
        grasp_failure = (object_pos[:, 2] < 0.1) & self.was_grasped  # On floor

        failure = collision | wrong_object | grasp_failure

        # Log failure types
        failure_types = torch.zeros(len(failure), dtype=torch.long)
        failure_types[collision] = 0  # Collision
        failure_types[wrong_object] = 1  # Wrong object
        failure_types[grasp_failure] = 2  # Grasp failure

        return failure, failure_types
```

## Phase 2: Core Predictor (Weeks 5-8)

### Week 5-6: Multi-Horizon Predictor

**File: `guardian/models/predictor.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHorizonFailurePredictor(nn.Module):
    """
    Predicts failure type distribution across time horizons
    Output: (batch, num_failure_types, num_horizons)
    """

    FAILURE_TYPES = ['collision', 'wrong_object', 'grasp_failure', 'goal_miss']
    HORIZONS = [200, 300, 400, 500]  # milliseconds

    def __init__(self, input_dim=12, hidden_dim=128):
        super().__init__()

        # Shared feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Separate prediction head per failure type
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, len(self.HORIZONS))
            for _ in range(len(self.FAILURE_TYPES))
        ])

        # Temperature for calibration
        self.register_buffer('temperature', torch.ones(1))

    def forward(self, features):
        """
        Args:
            features: (B, 12) signal features

        Returns:
            probs: (B, num_types, num_horizons) failure probabilities
        """
        # Encode features
        h = self.encoder(features)  # (B, hidden_dim)

        # Per-type predictions
        logits = []
        for head in self.heads:
            type_logits = head(h)  # (B, num_horizons)
            logits.append(type_logits)

        logits = torch.stack(logits, dim=1)  # (B, num_types, num_horizons)

        # Apply temperature scaling
        logits = logits / self.temperature

        # Softmax over horizons for each type
        probs = torch.softmax(logits, dim=-1)

        return probs

    def get_intervention_signal(self, probs, threshold=0.7):
        """
        Determine if intervention needed based on probabilities

        Args:
            probs: (B, num_types, num_horizons)
            threshold: Intervention threshold (lower for collision)

        Returns:
            intervene: (B,) bool tensor
            failure_type: (B,) predicted failure type (0-3)
            horizon_idx: (B,) predicted horizon (0-3)
        """
        # Use adaptive thresholds per type
        thresholds = torch.tensor([
            0.5,  # Collision (lower threshold - critical)
            0.7,  # Wrong object
            0.7,  # Grasp failure
            0.8   # Goal miss (higher threshold - recoverable)
        ]).to(probs.device)

        # Find highest probability across all types/horizons
        max_probs = probs.max(dim=-1)[0]  # (B, num_types)

        # Check if any type exceeds its threshold
        intervene = (max_probs > thresholds.unsqueeze(0)).any(dim=-1)

        # Get failure type and horizon for intervention
        failure_type = max_probs.argmax(dim=-1)
        horizon_idx = probs.argmax(dim=-1).gather(1, failure_type.unsqueeze(1)).squeeze(1)

        return intervene, failure_type, horizon_idx
```

**File: `guardian/training/train_predictor.py`**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class FailureDataset(Dataset):
    """Dataset of failure episodes with temporal labels"""

    def __init__(self, data_dir, horizon_bins=[200, 300, 400, 500]):
        self.data_dir = data_dir
        self.horizon_bins = np.array(horizon_bins) / 1000.0  # Convert to seconds
        self.episodes = self._load_episodes()

    def _load_episodes(self):
        """Load all episodes and extract failure labels"""
        episodes = []

        for i in range(10000):  # Adjust based on actual count
            try:
                filename = f"{self.data_dir}/episode_{i:05d}.h5"
                with h5py.File(filename, 'r') as f:
                    if not f.attrs['success']:
                        failure_type = f.attrs['failure_type']
                        failure_time = f.attrs['failure_time']
                        features = f['features'][:]  # Pre-extracted 12D features
                        timestamps = f['timestamps'][:]

                        episodes.append({
                            'features': features,
                            'timestamps': timestamps,
                            'failure_type': failure_type,
                            'failure_time': failure_time
                        })
            except:
                break

        return episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]

        # Sample a random timestep before failure
        failure_idx = np.where(ep['timestamps'] >= ep['failure_time'])[0][0]
        # Sample from 1-5 seconds before failure
        t = np.random.randint(max(0, failure_idx - 50), failure_idx)

        features = torch.tensor(ep['features'][t], dtype=torch.float32)

        # Create multi-label target: which horizons will see this failure?
        time_to_failure = ep['failure_time'] - ep['timestamps'][t]
        target = torch.zeros(len(self.horizon_bins))

        for i, horizon in enumerate(self.horizon_bins):
            if time_to_failure <= horizon:
                target[i] = 1.0

        # Convert failure type to index
        type_map = {'collision': 0, 'wrong_object': 1, 'grasp_failure': 2, 'goal_miss': 3}
        failure_type_idx = type_map[ep['failure_type']]

        return features, target, failure_type_idx

def train_predictor(data_dir, epochs=100, batch_size=64):
    """Train multi-horizon predictor"""

    dataset = FailureDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MultiHorizonFailurePredictor()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0

        for features, targets, failure_types in dataloader:
            optimizer.zero_grad()

            # Forward pass
            probs = model(features)  # (B, num_types, num_horizons)

            # Multi-label loss: predict which horizons will see failure
            loss = 0
            for b in range(len(features)):
                type_idx = failure_types[b]
                pred = probs[b, type_idx]  # (num_horizons,)
                target = targets[b]  # (num_horizons,)

                # Binary cross-entropy for each horizon
                loss += F.binary_cross_entropy(pred, target)

            loss = loss / len(features)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")

    # Calibration (temperature scaling)
    calibrate_temperature(model, dataset)

    return model

def calibrate_temperature(model, dataset, num_samples=1000):
    """Fit temperature parameter for calibration"""
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Collect predictions
    all_probs = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for features, targets, failure_types in dataloader:
            if len(all_probs) * 32 >= num_samples:
                break
            probs = model(features)
            all_probs.append(probs)
            all_targets.append(targets)

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    # Optimize temperature
    temperature = nn.Parameter(torch.ones(1))
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy(all_probs / temperature, all_targets)
        loss.backward()
        return loss

    optimizer.step(eval)
    model.temperature.copy_(temperature.data)

    print(f"Calibrated temperature: {temperature.item():.3f}")
```

This is Part 1 of the implementation guide. Would you like me to continue with:

1. **Phase 3: Safety Manifold Implementation** (contrastive learning, VAE)
2. **Phase 4: MPC Synthesis** (NVIDIA Warp integration, parallel rollouts)
3. **Phase 5: Self-Validating Dynamics**
4. **Phase 6: Federated Learning** (OpenFL integration)
5. **Phase 7: Deployment** (Ray Serve, real robot integration)

And then move to **PART 2: IP PROTECTION STRATEGY**?
