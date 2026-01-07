# SALUS/GUARDIAN: Getting Started Guide

## You're Ready to Start NOW!

You have everything you need to begin:
- ✅ 4x 2080 Ti GPUs (perfect for the 4-GPU architecture)
- ✅ HPC cluster access (for heavy training)
- ⏳ Physical robot (NOT needed until Phase 2 - months 4-6)

## What You Can Do Without a Robot

**90% of SALUS development happens in simulation!** You'll build and train the entire system using Isaac Lab's virtual Unitree G1 robot before ever touching hardware.

---

## WEEK 1: Environment Setup (START HERE)

### Day 1-2: Core Dependencies

Run these commands on your 4x 2080 Ti machine:

```bash
# Navigate to aegis directory
cd ~/Desktop/aegis

# Create project structure
mkdir -p salus/{core,data,training,deployment,simulation,utils,scripts,tests,models,notebooks}
mkdir -p salus/core/{predictor,manifold,synthesis,vla}
mkdir -p salus/data
mkdir -p salus/models/{predictor,manifold,dynamics}

# Create Python package
cat > salus/__init__.py << 'EOF'
"""
SALUS/GUARDIAN: Predictive Runtime Safety for VLA Models
Copyright (c) 2025 - Proprietary and Confidential
"""
__version__ = "0.1.0"
EOF

# Create virtual environment
python3.10 -m venv venv_salus
source venv_salus/bin/activate

# Install PyTorch for CUDA 11.8 (2080 Ti support)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Core ML dependencies
pip install numpy scipy scikit-learn matplotlib seaborn
pip install opencv-python pillow h5py tqdm wandb tensorboard

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
# Should output: CUDA available: True, GPUs: 4
```

### Day 3: Isaac Lab Installation

Isaac Lab is NVIDIA's new robotics simulation framework (successor to Isaac Gym). It's GPU-accelerated and perfect for our needs.

```bash
# Install Isaac Sim dependencies (Ubuntu 22.04)
# Note: Isaac Sim requires NVIDIA driver 525+ for 2080 Ti
nvidia-smi  # Check driver version

# Install Isaac Lab
cd ~/
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Run installer (downloads Isaac Sim binaries, ~15GB)
./isaaclab.sh --install

# Activate Isaac Lab environment
source _isaac_sim/setup_conda_env.sh

# Test installation
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
# Should open a window with empty scene

# Install NVIDIA Warp (for physics)
pip install nvidia-warp
```

**Alternative if Isaac Lab doesn't work**: Use MuJoCo (lighter weight)
```bash
pip install mujoco
pip install dm_control
```

### Day 4: VLA Model Setup

Download and test baseline VLA models:

```bash
# Option 1: TinyVLA (1B params, faster, recommended for development)
cd ~/
git clone https://github.com/OpenDriveLab/TinyVLA.git
cd TinyVLA
pip install -e .

# Download pretrained weights
huggingface-cli download TinyVLA/tinyvla-1b --local-dir ./models/tinyvla-1b

# Test inference
python scripts/test_inference.py --model models/tinyvla-1b

# Option 2: OpenVLA (7B params, higher quality, use for final system)
cd ~/
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# Download pretrained weights (warning: 14GB)
huggingface-cli download openvla/openvla-7b --local-dir ./models/openvla-7b
```

### Day 5: ROS2 Setup (Optional for Week 1, needed later)

```bash
# ROS2 Humble (Ubuntu 22.04)
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl gnupg lsb-release

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop ros-humble-moveit ros-humble-ros2-control

# Add to .bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Test ROS2
ros2 run demo_nodes_cpp talker
# Should print "Hello World" messages
```

---

## WEEK 2: Simulation Environment (Critical!)

### Create Isaac Lab G1 Environment

Create file: `salus/simulation/g1_pick_place_env.py`

```python
"""
Unitree G1 Humanoid Pick-and-Place Environment
Simulates manipulation tasks with failure injection
"""

from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg
from omni.isaac.lab.utils import configclass
import torch
import numpy as np

@configclass
class G1PickPlaceEnvCfg(DirectRLEnvCfg):
    """Configuration for G1 pick-and-place with failure modes"""

    # Simulation settings
    episode_length_s = 60.0
    decimation = 4  # 50Hz control loop
    num_envs = 16  # Parallel environments

    # Observation space
    observation_space = 23  # joint_pos(13) + joint_vel(13) + wrench(6) + ...
    action_space = 13  # 7 DoF arm + 6 DoF base (simplified)

    # Failure modes to inject
    failure_injection = {
        'collision': 0.15,  # 15% of episodes
        'wrong_object': 0.10,
        'grasp_failure': 0.15,
        'goal_miss': 0.15
    }

    # Domain randomization
    randomization = {
        'object_pose_noise': 0.05,  # ±5cm
        'object_mass_range': [0.05, 0.5],  # kg
        'friction_range': [0.3, 1.2],
        'lighting_intensity': [100, 2000],  # lux
        'camera_noise_std': 0.01
    }

class G1PickPlaceEnv(DirectRLEnv):
    """Isaac Lab environment for G1 manipulation with failures"""

    cfg: G1PickPlaceEnvCfg

    def __init__(self, cfg: G1PickPlaceEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Track failure occurrences
        self.failure_log = {
            'episode_id': [],
            'failure_type': [],
            'failure_time': [],
            'signals_at_failure': []
        }

    def _setup_scene(self):
        """Load robot, objects, sensors"""
        # Load G1 humanoid (URDF from Unitree)
        self.robot = self._load_articulation("g1_humanoid.urdf")

        # Load manipulation objects (blocks, cups, etc.)
        self.objects = self._spawn_objects(num_objects=5)

        # Add cameras (RGB + Depth)
        self.cameras = self._add_cameras()

        # Add contact sensors for collision detection
        self.contact_sensors = self._add_contact_sensors()

    def _get_observations(self) -> dict:
        """Return current observation"""
        return {
            'rgb': self.cameras['wrist_cam'].get_rgb(),  # (num_envs, 3, H, W)
            'depth': self.cameras['wrist_cam'].get_depth(),  # (num_envs, H, W)
            'proprio': torch.cat([
                self.robot.get_joint_positions(),  # (num_envs, 13)
                self.robot.get_joint_velocities(),  # (num_envs, 13)
                self.robot.get_end_effector_wrench()  # (num_envs, 6)
            ], dim=-1)
        }

    def _inject_failures(self):
        """Randomly inject failure conditions"""
        # Example: Spawn obstacle for collision
        if np.random.rand() < self.cfg.failure_injection['collision']:
            obstacle_pos = self.robot.get_end_effector_pose() + torch.randn(3) * 0.1
            self._spawn_obstacle(obstacle_pos)

        # Example: Swap target object for wrong_object failure
        if np.random.rand() < self.cfg.failure_injection['wrong_object']:
            self._swap_target_object()

    def _check_termination(self) -> tuple:
        """Check for success or failure"""
        # Collision detection
        contact_forces = self.contact_sensors.get_forces()
        collision = (contact_forces > 10.0).any(dim=-1)  # 10N threshold

        # Wrong object detection
        grasped_obj_id = self._get_grasped_object_id()
        wrong_object = (grasped_obj_id != self.target_obj_id) & (grasped_obj_id != -1)

        # Grasp failure (object dropped)
        object_on_floor = self.objects[self.target_obj_id].get_position()[:, 2] < 0.05
        grasp_failure = object_on_floor & self.was_grasped

        # Goal miss (timeout without reaching goal)
        goal_miss = (self.episode_length > 55.0) & ~self.reached_goal

        # Combine failures
        failure = collision | wrong_object | grasp_failure | goal_miss

        # Log failure type
        failure_type = torch.zeros(self.num_envs, dtype=torch.long)
        failure_type[collision] = 0
        failure_type[wrong_object] = 1
        failure_type[grasp_failure] = 2
        failure_type[goal_miss] = 3

        return failure, failure_type

# Create environment
env_cfg = G1PickPlaceEnvCfg()
env = G1PickPlaceEnv(env_cfg)
```

### Test Simulation

```bash
# Run simulation test
cd ~/Desktop/aegis
python salus/simulation/g1_pick_place_env.py

# Should see:
# - 16 parallel G1 robots
# - Random object placements
# - Some episodes failing (collisions, etc.)
```

---

## WEEK 3: Data Collection in Simulation

### Implement Data Recorder

Create file: `salus/data/sim_recorder.py`

```python
"""
Data recorder for simulation episodes
Collects multi-modal sensor data + failure labels
"""

import torch
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime

class SimulationDataRecorder:
    """Records episodes from Isaac Lab simulation"""

    def __init__(self, output_dir="./data/raw_episodes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0

        # Buffers for current episode
        self.reset_buffers()

    def reset_buffers(self):
        """Clear episode buffers"""
        self.buffers = {
            'rgb': [],
            'depth': [],
            'proprio': [],
            'actions': [],
            'vla_internals': [],
            'timestamps': []
        }
        self.start_time = datetime.now()

    def record_step(self, obs, action, vla_internals):
        """Record single timestep"""
        self.buffers['rgb'].append(obs['rgb'].cpu().numpy())
        self.buffers['depth'].append(obs['depth'].cpu().numpy())
        self.buffers['proprio'].append(obs['proprio'].cpu().numpy())
        self.buffers['actions'].append(action.cpu().numpy())

        # Store VLA internal signals (for signal extraction later)
        self.buffers['vla_internals'].append({
            'action_variance': vla_internals['action_var'].cpu().numpy(),
            'attention_entropy': vla_internals.get('attention_entropy', 0.0)
        })

        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.buffers['timestamps'].append(elapsed)

    def save_episode(self, success, failure_type=None, failure_time=None):
        """Save episode to HDF5"""
        filename = self.output_dir / f"episode_{self.episode_count:05d}.h5"

        with h5py.File(filename, 'w') as f:
            # Save trajectories
            for key in ['rgb', 'depth', 'proprio', 'actions', 'timestamps']:
                if len(self.buffers[key]) > 0:
                    f.create_dataset(key, data=np.stack(self.buffers[key]))

            # Save VLA internals
            if len(self.buffers['vla_internals']) > 0:
                action_vars = np.array([x['action_variance'] for x in self.buffers['vla_internals']])
                f.create_dataset('action_variance', data=action_vars)

            # Save labels
            f.attrs['success'] = success
            f.attrs['episode_id'] = self.episode_count

            if not success and failure_type is not None:
                f.attrs['failure_type'] = failure_type
                f.attrs['failure_time'] = failure_time

        print(f"✅ Saved episode {self.episode_count}: {'SUCCESS' if success else f'FAILURE ({failure_type})'}")

        self.episode_count += 1
        self.reset_buffers()

# Usage
recorder = SimulationDataRecorder(output_dir="./data/raw_episodes")
```

### Run Data Collection

Create script: `scripts/collect_simulation_data.py`

```python
"""
Collect 500+ episodes in simulation with failures
"""

import torch
from salus.simulation.g1_pick_place_env import G1PickPlaceEnv, G1PickPlaceEnvCfg
from salus.data.sim_recorder import SimulationDataRecorder
from tinyvla import TinyVLA  # or OpenVLA

# Setup
env_cfg = G1PickPlaceEnvCfg()
env_cfg.num_envs = 8  # Collect 8 episodes in parallel
env = G1PickPlaceEnv(env_cfg)

# Load VLA model
vla = TinyVLA.from_pretrained("models/tinyvla-1b")
vla.eval()
vla.cuda()

# Recorder (will save 8 parallel episodes)
recorders = [SimulationDataRecorder(f"./data/raw_episodes/env_{i}") for i in range(8)]

# Data collection loop
num_episodes = 500
episodes_collected = 0

while episodes_collected < num_episodes:
    obs = env.reset()
    done = torch.zeros(env.num_envs, dtype=torch.bool)

    while not done.all():
        # VLA forward pass
        with torch.no_grad():
            action, vla_internals = vla(obs, language="pick the red block", return_internals=True)

        # Record step for each parallel env
        for i in range(env.num_envs):
            if not done[i]:
                recorders[i].record_step(
                    {k: v[i] for k, v in obs.items()},
                    action[i],
                    {k: v[i] for k, v in vla_internals.items()}
                )

        # Step environment
        obs, rewards, done, info = env.step(action)

    # Save episodes
    for i in range(env.num_envs):
        success = info['success'][i]
        failure_type = info.get('failure_type', [None]*env.num_envs)[i]
        failure_time = info.get('failure_time', [None]*env.num_envs)[i]

        recorders[i].save_episode(success, failure_type, failure_time)

    episodes_collected += env.num_envs
    print(f"Progress: {episodes_collected}/{num_episodes} episodes")

print(f"✅ Data collection complete! {episodes_collected} episodes saved.")
```

Run it:
```bash
cd ~/Desktop/aegis
python scripts/collect_simulation_data.py

# Expected output:
# ✅ Saved episode 0: FAILURE (collision)
# ✅ Saved episode 1: SUCCESS
# ✅ Saved episode 2: FAILURE (grasp_failure)
# ...
# Progress: 500/500 episodes
```

**This will take 6-12 hours on your 4x 2080 Ti** (8 parallel envs × 60s episodes).

**Pro tip**: Run overnight on your machine, or submit to HPC cluster as batch job.

---

## WEEK 4: Signal Extraction + Predictor Training

### Extract Failure Signals

Create file: `scripts/extract_signals.py`

```python
"""
Extract 12D failure signal vectors from recorded episodes
"""

import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from salus.core.vla.wrapper import SignalExtractor

def extract_signals_from_dataset(data_dir="./data/raw_episodes", output_dir="./data/processed"):
    """Process all episodes and extract signals"""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = SignalExtractor(device="cuda:1")

    episode_files = sorted(data_dir.glob("episode_*.h5"))
    print(f"Found {len(episode_files)} episodes to process")

    for ep_file in tqdm(episode_files):
        with h5py.File(ep_file, 'r') as f_in:
            # Load episode data
            rgb = torch.tensor(f['rgb'][:])
            depth = torch.tensor(f['depth'][:])
            proprio = torch.tensor(f['proprio'][:])
            action_variance = torch.tensor(f['action_variance'][:])

            T = len(rgb)  # Episode length
            signals = []

            # Extract signals for each timestep
            for t in range(T):
                obs = {'rgb': rgb[t:t+1], 'proprio': proprio[t:t+1]}
                vla_internals = {'action_var': action_variance[t:t+1]}
                history = []  # TODO: Add trajectory history

                signal = extractor.extract(obs, vla_internals, history, depth[t:t+1])
                signals.append(signal.cpu().numpy())

            signals = np.stack(signals)  # (T, 12)

            # Save processed episode
            output_file = output_dir / ep_file.name
            with h5py.File(output_file, 'w') as f_out:
                # Copy original data
                f_out.create_dataset('signals', data=signals)
                f_out.create_dataset('timestamps', data=f['timestamps'][:])

                # Copy labels
                f_out.attrs['success'] = f.attrs['success']
                if not f.attrs['success']:
                    f_out.attrs['failure_type'] = f.attrs['failure_type']
                    f_out.attrs['failure_time'] = f.attrs['failure_time']

    print(f"✅ Signal extraction complete! Processed {len(episode_files)} episodes")

if __name__ == "__main__":
    extract_signals_from_dataset()
```

Run signal extraction:
```bash
python scripts/extract_signals.py
# Expected: 2-4 hours on your machine
```

### Train Multi-Horizon Predictor

Use the implementation from `salus_implementation_guide.md`:

```bash
# Create training script
cat > scripts/train_predictor.py << 'EOF'
import sys
sys.path.append('./salus')

from training.train_predictor import train_predictor

# Train on processed data
model = train_predictor(
    data_dir="./data/processed",
    epochs=100,
    batch_size=64,
    device="cuda:0"  # Use GPU 0
)

# Save model
torch.save(model.state_dict(), "./models/predictor/multi_horizon_v1.pth")
print("✅ Predictor training complete!")
EOF

# Run training
python scripts/train_predictor.py

# Expected training time: 2-4 hours on single GPU
```

**For HPC cluster**: Create SLURM script:

```bash
cat > hpc_train_predictor.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=aegis_predictor
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/predictor_%j.out

module load cuda/11.8
source ~/Desktop/aegis/venv_salus/bin/activate

cd ~/Desktop/aegis
python scripts/train_predictor.py
EOF

# Submit to cluster
sbatch hpc_train_predictor.sh
```

---

## Your Immediate Action Plan (Start TODAY)

### ✅ DAY 1 (2-3 hours):
```bash
cd ~/Desktop/aegis

# 1. Install core dependencies
python3.10 -m venv venv_salus
source venv_salus/bin/activate
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn matplotlib h5py tqdm

# 2. Test GPU setup
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# Should output: GPUs: 4

# 3. Clone VLA model
cd ~/
git clone https://github.com/OpenDriveLab/TinyVLA.git
cd TinyVLA
pip install -e .
```

### ✅ DAY 2-3 (8-12 hours):
```bash
# 4. Install Isaac Lab
cd ~/
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install  # Takes 1-2 hours (downloads ~15GB)

# 5. Test Isaac Lab
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py
# Should open simulation window
```

### ✅ DAY 4-5 (4-6 hours):
```bash
# 6. Create G1 environment (use code from WEEK 2 above)
# 7. Test simulation with VLA
```

### ✅ WEEK 2 (OVERNIGHT RUN):
```bash
# 8. Collect 500 episodes (run overnight)
python scripts/collect_simulation_data.py
# 6-12 hours runtime
```

### ✅ WEEK 3-4:
```bash
# 9. Extract signals
python scripts/extract_signals.py

# 10. Train predictor
python scripts/train_predictor.py  # or submit to HPC
```

---

## What You're NOT Waiting For

### ❌ You DON'T need:
- ❌ Physical robot (use Isaac Lab virtual G1)
- ❌ Real camera (use simulated RGB/depth)
- ❌ Force sensors (simulated in Isaac Lab)
- ❌ ROS2 bridge (not needed until robot deployment)

### ✅ You DO have:
- ✅ 4x 2080 Ti GPUs (perfect for 4-GPU architecture)
- ✅ HPC cluster (for heavy training later)
- ✅ All the code (in implementation guide)
- ✅ Simulation environment (Isaac Lab)

---

## Timeline (Realistic with Your Resources)

| Week | Task | Output | GPU Usage |
|------|------|--------|-----------|
| **1** | Environment setup | Isaac Lab + VLA working | Testing only |
| **2** | Simulation environment | G1 virtual robot | All 4 GPUs |
| **3** | Data collection | 500+ episodes (~50GB) | All 4 GPUs (overnight) |
| **4** | Signal extraction | 12D features extracted | 1 GPU |
| **5-6** | Train predictor | Multi-horizon model | 1 GPU or HPC |
| **7-8** | Train manifold | Safety subspace learned | 1 GPU or HPC |
| **9-10** | MPC synthesis | Full pipeline working | 4 GPUs |
| **11-12** | Integration testing | End-to-end in simulation | 4 GPUs |

**After Week 12**: You have a fully functional SALUS system running in simulation. Only then do you need the physical robot for validation.

---

## Commands to Run RIGHT NOW

Copy-paste this entire block into your terminal:

```bash
# Create project directory
cd ~/Desktop/aegis
mkdir -p salus/{core,data,training,deployment,simulation,utils,scripts,tests,models,notebooks}

# Activate environment
python3.10 -m venv venv_salus
source venv_salus/bin/activate

# Install PyTorch (CUDA 11.8 for 2080 Ti)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy scipy scikit-learn matplotlib h5py tqdm wandb

# Test GPU
python << 'PYEOF'
import torch
print("="*50)
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print("="*50)
PYEOF

echo "✅ Core setup complete! Next: Install Isaac Lab (see Day 2-3)"
```

**You're NOT waiting for anything. Start TODAY!**
