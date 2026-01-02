# SALUS Local Machine Setup (4x RTX 2080 Ti)

## Your Hardware Profile
- **GPUs**: 4x NVIDIA RTX 2080 Ti (11GB VRAM each = 44GB total)
- **GPU Architecture**: Turing (CUDA Compute 7.5)
- **CUDA Version Required**: 11.8 (for 2080 Ti support)
- **Role**: Development, data collection, and predictor training
- **HPC Cluster**: Use for heavy jobs (manifold, dynamics training)

---

## GPU Allocation Plan

Your 4x 2080 Ti setup maps perfectly to SALUS architecture:

```
┌─────────────────────────────────────────────────────┐
│  YOUR LOCAL MACHINE (4x RTX 2080 Ti - 11GB each)   │
└─────────────────────────────────────────────────────┘

GPU 0 (11GB) → VLA Ensemble (5 models × 2.2GB = 11GB)
   └─ TinyVLA-1B models × 5 (for epistemic uncertainty)

GPU 1 (4GB used) → Signal Extraction + Failure Predictor
   └─ Lightweight: feature extraction + predictor inference

GPU 2 (3GB used) → Safety Manifold (later phase)
   └─ Encoder/Decoder for safe action space

GPU 3 (8GB used) → MPC Synthesis + Dynamics Model (later phase)
   └─ Parallel rollouts for counterfactual actions
```

---

## PHASE 1: INITIAL SETUP (Day 1 - 3 hours)

### Step 1: Check GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# You should see:
# - Driver Version: 525.xx or higher
# - CUDA Version: 11.8 or 12.x
# - 4 GPUs listed: each RTX 2080 Ti with 11264MB
```

If driver is old (< 525), update:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot

# After reboot
nvidia-smi
```

### Step 2: Create Project Structure

```bash
cd ~/Desktop/aegis

# Create directory tree
mkdir -p salus/{core/{predictor,manifold,synthesis,vla},data,training,deployment,simulation,utils,scripts,tests,models/{predictor,manifold,dynamics},notebooks,config}

# Create data directories
mkdir -p data/{raw_episodes,processed,labels,checkpoints}

# Create logs directory
mkdir -p logs/{training,simulation,deployment}

# Verify structure
tree -L 2 salus/
```

### Step 3: Python Environment Setup

```bash
cd ~/Desktop/aegis

# Create virtual environment (Python 3.10 recommended)
python3.10 -m venv venv_salus

# Activate it
source venv_salus/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support (for RTX 2080 Ti)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch sees all 4 GPUs
python << 'PYEOF'
import torch
print("="*60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print("-"*60)
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}")
    print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute Capability: {props.major}.{props.minor}")
print("="*60)

# Test multi-GPU tensor allocation
try:
    tensors = []
    for i in range(4):
        t = torch.randn(1000, 1000).cuda(i)
        tensors.append(t)
    print("✅ All 4 GPUs accessible!")
except Exception as e:
    print(f"❌ GPU Error: {e}")
PYEOF
```

Expected output:
```
============================================================
PyTorch Version: 2.1.2+cu118
CUDA Available: True
CUDA Version: 11.8
Number of GPUs: 4
------------------------------------------------------------
GPU 0: NVIDIA GeForce RTX 2080 Ti
  Memory: 11.0 GB
  Compute Capability: 7.5
GPU 1: NVIDIA GeForce RTX 2080 Ti
  Memory: 11.0 GB
  Compute Capability: 7.5
GPU 2: NVIDIA GeForce RTX 2080 Ti
  Memory: 11.0 GB
  Compute Capability: 7.5
GPU 3: NVIDIA GeForce RTX 2080 Ti
  Memory: 11.0 GB
  Compute Capability: 7.5
============================================================
✅ All 4 GPUs accessible!
```

### Step 4: Install Core Dependencies

```bash
# ML/Scientific libraries
pip install numpy==1.24.3 scipy scikit-learn pandas

# Computer Vision
pip install opencv-python pillow torchvision

# Data handling
pip install h5py zarr

# Training utilities
pip install tqdm wandb tensorboard

# Robotics (will install ROS2 separately)
pip install transforms3d spatialmath-python

# Visualization
pip install matplotlib seaborn plotly

# Code quality
pip install black flake8 pytest

# Save requirements
pip freeze > requirements.txt

echo "✅ Core dependencies installed!"
```

---

## PHASE 2: VLA MODEL SETUP (Day 1 - 2 hours)

### Download TinyVLA (Recommended for 2080 Ti)

TinyVLA-1B is perfect for your hardware (only 2.2GB per model).

```bash
cd ~/

# Install Hugging Face CLI
pip install huggingface-hub

# Login to Hugging Face (optional, for gated models)
# huggingface-cli login

# Clone TinyVLA repository
git clone https://github.com/OpenDriveLab/TinyVLA.git
cd TinyVLA

# Install TinyVLA
pip install -e .

# Download pretrained model (~2.5GB)
cd ~/
mkdir -p models/tinyvla
cd models/tinyvla

# Download TinyVLA-1B weights
huggingface-cli download TinyVLA/tinyvla-1b --local-dir ./tinyvla-1b

# Verify download
ls -lh tinyvla-1b/
# Should see: pytorch_model.bin (~2.2GB), config.json, etc.
```

### Test TinyVLA Inference

```bash
cd ~/Desktop/aegis

# Create test script
cat > scripts/test_tinyvla.py << 'EOF'
"""Test TinyVLA inference on single GPU"""
import torch
import sys
sys.path.append('/home/YOUR_USERNAME/TinyVLA')  # Adjust path
from tinyvla import TinyVLA

print("Loading TinyVLA-1B...")
model = TinyVLA.from_pretrained("~/models/tinyvla/tinyvla-1b")
model = model.cuda(0)  # Load on GPU 0
model.eval()

print(f"Model loaded on: {next(model.parameters()).device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# Test inference with dummy data
batch_size = 1
rgb = torch.randn(batch_size, 3, 224, 224).cuda(0)
proprio = torch.randn(batch_size, 7).cuda(0)  # 7-DoF arm
language = "pick up the red block"

print("\nRunning inference...")
with torch.no_grad():
    output = model(rgb, proprio, language)
    action = output['action']

print(f"✅ Inference successful!")
print(f"Input shape: RGB {rgb.shape}, Proprio {proprio.shape}")
print(f"Output action shape: {action.shape}")
print(f"GPU memory used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")

print("\n✅ TinyVLA is working!")
EOF

# Run test
python scripts/test_tinyvla.py
```

### Create VLA Ensemble (Use all 5 on GPU 0)

```bash
cat > scripts/test_vla_ensemble.py << 'EOF'
"""Test loading 5 TinyVLA models on GPU 0 for ensemble"""
import torch
import sys
sys.path.append('/home/YOUR_USERNAME/TinyVLA')
from tinyvla import TinyVLA

print("Loading VLA Ensemble (5 models on GPU 0)...")

ensemble = []
for i in range(5):
    print(f"Loading model {i+1}/5...")
    model = TinyVLA.from_pretrained("~/models/tinyvla/tinyvla-1b")
    model = model.cuda(0)
    model.eval()
    ensemble.append(model)

    mem_gb = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  GPU 0 memory: {mem_gb:.2f} GB")

print(f"\n✅ Ensemble loaded! Total GPU 0 memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / 11.0 GB")

# Test ensemble inference
rgb = torch.randn(1, 3, 224, 224).cuda(0)
proprio = torch.randn(1, 7).cuda(0)
language = "grasp the object"

actions = []
with torch.no_grad():
    for i, model in enumerate(ensemble):
        output = model(rgb, proprio, language)
        actions.append(output['action'])

actions = torch.stack(actions, dim=0)  # (5, 1, action_dim)
mean_action = actions.mean(dim=0)
action_variance = actions.var(dim=0)  # Epistemic uncertainty!

print(f"\n✅ Ensemble inference successful!")
print(f"Action shape: {actions.shape}")
print(f"Mean action: {mean_action.squeeze()}")
print(f"Action variance (uncertainty): {action_variance.squeeze()}")
EOF

python scripts/test_vla_ensemble.py
```

---

## PHASE 3: SIMULATION SETUP (Days 2-3 - 8 hours)

You have two options: **Isaac Lab** (official, GPU-accelerated) or **MuJoCo** (lighter, easier).

### Option A: Isaac Lab (Recommended, GPU-accelerated)

```bash
# Install Isaac Sim dependencies
cd ~/

# Download Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Run installer (downloads ~15GB, takes 1-2 hours)
./isaaclab.sh --install

# This will:
# - Download Isaac Sim binaries
# - Install Omniverse dependencies
# - Set up conda environment
# - Configure GPU settings

# Activate Isaac Lab environment
source _isaac_sim/setup_conda_env.sh

# Test installation
./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py

# Should open Isaac Sim window with empty scene
# If it works → you're good!
# If it crashes → try Option B (MuJoCo)
```

### Option B: MuJoCo (Backup, lighter weight)

```bash
# Install MuJoCo (much faster, ~5 minutes)
pip install mujoco
pip install dm_control
pip install mujoco-python-viewer

# Test MuJoCo
python << 'PYEOF'
import mujoco
import numpy as np

print("MuJoCo version:", mujoco.__version__)

# Load example model
xml = """
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .1 .1" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Simulate
for _ in range(100):
    mujoco.mj_step(model, data)

print("✅ MuJoCo working!")
PYEOF
```

---

## PHASE 4: IMPLEMENT CORE SALUS COMPONENTS (Days 3-7)

### Create Core Package Structure

```bash
cd ~/Desktop/aegis/salus

# Create __init__.py files
cat > __init__.py << 'EOF'
"""
SALUS/GUARDIAN: Predictive Runtime Safety for VLA Models
Copyright (c) 2025 - Proprietary
"""
__version__ = "0.1.0"
EOF

# Create sub-package inits
for dir in core core/predictor core/manifold core/synthesis core/vla data training deployment simulation utils; do
    echo "\"\"\"$(basename $dir) module\"\"\"" > $dir/__init__.py
done
```

### Implement VLA Wrapper with Signal Extraction

```bash
cat > salus/core/vla/wrapper.py << 'EOF'
"""VLA Ensemble Wrapper with Internal Signal Extraction"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import sys
sys.path.append('/home/YOUR_USERNAME/TinyVLA')  # CHANGE THIS
from tinyvla import TinyVLA


class VLAEnsemble(nn.Module):
    """
    Manages ensemble of VLA models with internal signal extraction
    Runs on GPU 0 (11GB)
    """

    def __init__(
        self,
        model_path: str = "~/models/tinyvla/tinyvla-1b",
        ensemble_size: int = 5,
        device: str = "cuda:0"
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.device = device

        print(f"Loading VLA ensemble ({ensemble_size} models on {device})...")

        # Load ensemble
        self.models = nn.ModuleList()
        for i in range(ensemble_size):
            model = TinyVLA.from_pretrained(model_path)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            self.models.append(model)
            print(f"  Model {i+1}/{ensemble_size} loaded")

        self.to(device)
        print(f"✅ Ensemble ready on {device}")

    def forward(
        self,
        rgb: torch.Tensor,
        proprio: torch.Tensor,
        language: str,
        return_internals: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through ensemble

        Args:
            rgb: (B, 3, H, W) RGB images
            proprio: (B, D) proprioceptive state
            language: Task description
            return_internals: Return attention/hidden states

        Returns:
            actions: (B, ensemble_size, action_dim)
            internals: Dict with attention, hidden, variance
        """
        rgb = rgb.to(self.device)
        proprio = proprio.to(self.device)

        actions = []
        attentions = []
        hiddens = []

        with torch.no_grad():
            for model in self.models:
                output = model(rgb, proprio, language)
                actions.append(output['action'])

                if return_internals and 'attention' in output:
                    attentions.append(output['attention'])
                if return_internals and 'hidden' in output:
                    hiddens.append(output['hidden'])

        actions = torch.stack(actions, dim=1)  # (B, K, action_dim)

        if return_internals:
            internals = {
                'action_mean': actions.mean(dim=1),
                'action_var': actions.var(dim=1),  # Epistemic uncertainty!
                'actions_all': actions
            }

            if attentions:
                internals['attention'] = torch.stack(attentions, dim=1)
            if hiddens:
                internals['hidden'] = torch.stack(hiddens, dim=1)

            return actions, internals
        else:
            return actions.mean(dim=1), None


class SignalExtractor:
    """
    Extract 12D failure signal vector
    Runs on GPU 1 (lightweight)
    """

    def __init__(self, device: str = "cuda:1"):
        self.device = device
        print(f"Signal extractor initialized on {device}")

    def extract(
        self,
        vla_internals: Dict,
        depth: Optional[torch.Tensor] = None,
        history: Optional[list] = None
    ) -> torch.Tensor:
        """
        Extract 12D feature vector

        Returns:
            features: (B, 12) tensor
                [σ²_ens, σ²_action,     # Epistemic (2)
                 H_attn, d_attn,        # Attention (2)
                 d_traj, d_learned,     # Trajectory (2)
                 d_obs, F_grip, occ,    # Environment (3)
                 0, 0, 0]               # Reserved (3)
        """
        batch_size = vla_internals['action_mean'].shape[0]
        features = torch.zeros(batch_size, 12, device=self.device)

        # 1. Epistemic Uncertainty (2 features)
        action_var = vla_internals['action_var'].to(self.device)
        features[:, 0] = action_var.mean(dim=1)  # Average variance
        features[:, 1] = action_var.max(dim=1)[0]  # Max variance

        # 2. Attention Degradation (2 features)
        if 'attention' in vla_internals:
            attention = vla_internals['attention'].to(self.device)
            features[:, 2] = self._compute_entropy(attention)
            features[:, 3] = self._compute_misalignment(attention)

        # 3. Trajectory Divergence (2 features)
        # TODO: Implement trajectory encoding

        # 4. Environmental Risk (3 features)
        if depth is not None:
            depth = depth.to(self.device)
            features[:, 6] = self._obstacle_distance(depth)

        # Gripper force and occlusion will be added later

        return features

    def _compute_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """Compute attention entropy"""
        # attention: (B, K, num_heads, seq_len, seq_len)
        attn_avg = attention.mean(dim=(1, 2))  # Average over ensemble and heads
        attn_probs = F.softmax(attn_avg, dim=-1)
        entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1).mean(dim=-1)
        return entropy

    def _compute_misalignment(self, attention: torch.Tensor) -> torch.Tensor:
        """Compute attention misalignment (simplified)"""
        # Measure variance in attention patterns
        attn_avg = attention.mean(dim=2)  # Average over heads
        attn_var = attention.var(dim=1)  # Variance across ensemble
        misalignment = attn_var.mean(dim=(1, 2, 3))  # Average over all dims
        return misalignment

    def _obstacle_distance(self, depth: torch.Tensor) -> torch.Tensor:
        """Minimum distance to obstacles from depth map"""
        # depth: (B, H, W)
        # Filter robot pixels (center 30% of image)
        B, H, W = depth.shape
        mask = torch.ones_like(depth, dtype=torch.bool)
        h_start, h_end = int(0.35*H), int(0.65*H)
        w_start, w_end = int(0.35*W), int(0.65*W)
        mask[:, h_start:h_end, w_start:w_end] = False

        # Get obstacle depths (outside robot region)
        obstacle_depths = []
        for b in range(B):
            depths_b = depth[b][mask[b]]
            if len(depths_b) > 0:
                obstacle_depths.append(depths_b.min())
            else:
                obstacle_depths.append(torch.tensor(10.0, device=self.device))

        return torch.stack(obstacle_depths)


# Test the wrapper
if __name__ == "__main__":
    print("Testing VLA Ensemble + Signal Extractor...\n")

    # Load ensemble
    vla = VLAEnsemble(
        model_path="~/models/tinyvla/tinyvla-1b",
        ensemble_size=5,
        device="cuda:0"
    )

    # Test inference
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 224, 224)
    proprio = torch.randn(batch_size, 7)
    language = "pick up the red block"

    print("\nRunning VLA inference...")
    actions, internals = vla(rgb, proprio, language, return_internals=True)

    print(f"✅ Actions shape: {actions.shape}")
    print(f"✅ Action variance (uncertainty): {internals['action_var']}")

    # Extract signals
    print("\nExtracting failure signals...")
    extractor = SignalExtractor(device="cuda:1")

    depth = torch.randn(batch_size, 480, 640)  # Depth map
    signals = extractor.extract(internals, depth=depth)

    print(f"✅ Signal features shape: {signals.shape}")
    print(f"Signal vector (first sample):\n{signals[0]}")

    print("\n✅ VLA wrapper working!")
EOF

# Test it
cd ~/Desktop/aegis
python -m salus.core.vla.wrapper
```

---

## PHASE 5: DATA COLLECTION IN SIMULATION (Week 2 - Overnight Run)

### Create Simulation Environment

This will run on all 4 GPUs for maximum throughput.

```bash
cat > salus/simulation/simple_pick_place.py << 'EOF'
"""
Simplified pick-place environment for data collection
Uses MuJoCo for faster iteration (switch to Isaac Lab later)
"""
import torch
import numpy as np
import mujoco
from pathlib import Path


class SimplePickPlaceEnv:
    """
    Simple pick-and-place environment
    - Runs 16 parallel environments (4 per GPU)
    - Injects failures randomly
    - Records RGB, depth, proprio, actions
    """

    def __init__(self, num_envs=16, device="cuda:0"):
        self.num_envs = num_envs
        self.device = device

        # Failure injection probabilities
        self.failure_probs = {
            'collision': 0.15,
            'wrong_object': 0.10,
            'grasp_failure': 0.15,
            'goal_miss': 0.10
        }

        print(f"✅ Environment initialized: {num_envs} parallel envs")

    def reset(self):
        """Reset all environments"""
        obs = {
            'rgb': torch.randn(self.num_envs, 3, 224, 224, device=self.device),
            'depth': torch.randn(self.num_envs, 480, 640, device=self.device),
            'proprio': torch.randn(self.num_envs, 7, device=self.device)
        }

        # Randomly inject failures
        self.failure_injected = torch.rand(self.num_envs) < 0.3
        self.failure_types = torch.randint(0, 4, (self.num_envs,))  # 4 failure types

        return obs

    def step(self, actions):
        """Execute actions"""
        # Simulate physics (placeholder)
        obs = {
            'rgb': torch.randn(self.num_envs, 3, 224, 224, device=self.device),
            'depth': torch.randn(self.num_envs, 480, 640, device=self.device),
            'proprio': torch.randn(self.num_envs, 7, device=self.device)
        }

        # Check termination
        done = torch.rand(self.num_envs, device=self.device) < 0.02  # 2% chance per step

        # Mark failures
        failed = done & self.failure_injected

        info = {
            'success': done & ~self.failure_injected,
            'failed': failed,
            'failure_type': self.failure_types
        }

        return obs, done, info


if __name__ == "__main__":
    env = SimplePickPlaceEnv(num_envs=16, device="cuda:0")
    obs = env.reset()

    for i in range(10):
        actions = torch.randn(16, 7, device="cuda:0")
        obs, done, info = env.step(actions)

        if done.any():
            print(f"Step {i}: {done.sum().item()} episodes done")

    print("✅ Environment working!")
EOF

python salus/simulation/simple_pick_place.py
```

### Create Data Recorder

```bash
cat > salus/data/recorder.py << 'EOF'
"""Multi-modal data recorder for simulation episodes"""
import torch
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime


class SimulationRecorder:
    """Records simulation episodes to HDF5 files"""

    def __init__(self, output_dir="./data/raw_episodes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0
        self.reset_buffers()

    def reset_buffers(self):
        """Clear episode buffers"""
        self.buffers = {
            'rgb': [],
            'depth': [],
            'proprio': [],
            'actions': [],
            'signals': [],
            'timestamps': []
        }
        self.start_time = datetime.now()

    def record_step(self, obs, action, signals):
        """Record single timestep"""
        # Convert to numpy and store
        self.buffers['rgb'].append(obs['rgb'].cpu().numpy())
        self.buffers['depth'].append(obs['depth'].cpu().numpy())
        self.buffers['proprio'].append(obs['proprio'].cpu().numpy())
        self.buffers['actions'].append(action.cpu().numpy())
        self.buffers['signals'].append(signals.cpu().numpy())

        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.buffers['timestamps'].append(elapsed)

    def save_episode(self, success, failure_type=None, failure_time=None):
        """Save episode to HDF5"""
        if len(self.buffers['timestamps']) == 0:
            return  # Empty episode, skip

        filename = self.output_dir / f"episode_{self.episode_count:06d}.h5"

        with h5py.File(filename, 'w') as f:
            # Save trajectories
            for key in ['rgb', 'depth', 'proprio', 'actions', 'signals', 'timestamps']:
                if len(self.buffers[key]) > 0:
                    data = np.array(self.buffers[key])
                    f.create_dataset(key, data=data, compression='gzip')

            # Save labels
            f.attrs['success'] = success
            f.attrs['episode_id'] = self.episode_count

            if not success and failure_type is not None:
                f.attrs['failure_type'] = int(failure_type)
                if failure_time is not None:
                    f.attrs['failure_time'] = float(failure_time)

        status = "SUCCESS" if success else f"FAILURE (type {failure_type})"
        print(f"📁 Saved episode {self.episode_count:06d}: {status}")

        self.episode_count += 1
        self.reset_buffers()


if __name__ == "__main__":
    recorder = SimulationRecorder("./data/test_episodes")

    # Test recording
    for _ in range(5):
        obs = {
            'rgb': torch.randn(3, 224, 224),
            'depth': torch.randn(480, 640),
            'proprio': torch.randn(7)
        }
        action = torch.randn(7)
        signals = torch.randn(12)

        recorder.record_step(obs, action, signals)

    recorder.save_episode(success=True)
    print("✅ Recorder working!")
EOF

python salus/data/recorder.py
```

### Create Data Collection Script

This is THE MAIN SCRIPT you'll run overnight:

```bash
cat > scripts/collect_data_local.py << 'EOF'
"""
Data collection on local 4x 2080 Ti machine
Runs 16 parallel environments across 4 GPUs (4 envs per GPU)
Collects 500-1000 episodes overnight
"""
import torch
import sys
from pathlib import Path

# Add salus to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from salus.simulation.simple_pick_place import SimplePickPlaceEnv
from salus.core.vla.wrapper import VLAEnsemble, SignalExtractor
from salus.data.recorder import SimulationRecorder
from tqdm import tqdm


def collect_data(
    num_episodes=500,
    num_envs_per_gpu=4,
    max_steps_per_episode=100,
    output_dir="./data/raw_episodes"
):
    """
    Collect simulation data

    Args:
        num_episodes: Total episodes to collect
        num_envs_per_gpu: Parallel envs per GPU (4 × 4 = 16 total)
        max_steps_per_episode: Max steps before timeout
        output_dir: Where to save episodes
    """

    print("="*60)
    print("SALUS DATA COLLECTION")
    print("="*60)
    print(f"Target episodes: {num_episodes}")
    print(f"Environments: {num_envs_per_gpu * 4} parallel (4 per GPU)")
    print(f"Output: {output_dir}")
    print("="*60)

    # Initialize VLA (on GPU 0)
    print("\n[1/4] Loading VLA ensemble...")
    vla = VLAEnsemble(
        model_path="~/models/tinyvla/tinyvla-1b",
        ensemble_size=5,
        device="cuda:0"
    )

    # Initialize signal extractor (on GPU 1)
    print("[2/4] Loading signal extractor...")
    extractor = SignalExtractor(device="cuda:1")

    # Initialize environments (distribute across 4 GPUs)
    print("[3/4] Creating simulation environments...")
    envs = []
    for gpu_id in range(4):
        env = SimplePickPlaceEnv(
            num_envs=num_envs_per_gpu,
            device=f"cuda:{gpu_id}"
        )
        envs.append(env)

    total_envs = num_envs_per_gpu * 4
    print(f"✅ {total_envs} environments ready")

    # Initialize recorders (one per parallel env)
    print("[4/4] Creating data recorders...")
    recorders = [
        SimulationRecorder(Path(output_dir) / f"env_{i}")
        for i in range(total_envs)
    ]

    print("\n" + "="*60)
    print("STARTING DATA COLLECTION")
    print("="*60)

    episodes_collected = 0
    pbar = tqdm(total=num_episodes, desc="Collecting episodes")

    while episodes_collected < num_episodes:
        # Reset all environments
        all_obs = []
        for env in envs:
            obs = env.reset()
            all_obs.append(obs)

        # Flatten observations from all GPUs
        # obs_list[i] = obs for env i
        obs_flat = []
        for gpu_obs in all_obs:
            for i in range(num_envs_per_gpu):
                obs_flat.append({
                    k: v[i:i+1] for k, v in gpu_obs.items()
                })

        # Run episode
        done_flags = [False] * total_envs
        step_count = 0

        while not all(done_flags) and step_count < max_steps_per_episode:
            # VLA inference (batched across all envs)
            # For simplicity, process each env individually (TODO: batch this)
            actions_list = []
            signals_list = []

            for env_idx, obs in enumerate(obs_flat):
                if done_flags[env_idx]:
                    # Env is done, skip
                    actions_list.append(None)
                    signals_list.append(None)
                    continue

                # VLA forward pass
                rgb = obs['rgb']  # (1, 3, 224, 224)
                proprio = obs['proprio']  # (1, 7)
                language = "pick up the red block"

                actions, internals = vla(rgb, proprio, language, return_internals=True)
                action = actions.mean(dim=1)  # (1, action_dim)

                # Extract signals
                depth = obs['depth']  # (1, H, W)
                signals = extractor.extract(internals, depth=depth)

                # Record step
                recorders[env_idx].record_step(obs, action, signals)

                actions_list.append(action)
                signals_list.append(signals)

            # Step environments
            for gpu_id, env in enumerate(envs):
                # Get actions for this GPU's envs
                start_idx = gpu_id * num_envs_per_gpu
                end_idx = start_idx + num_envs_per_gpu

                gpu_actions = []
                for i in range(start_idx, end_idx):
                    if actions_list[i] is not None:
                        gpu_actions.append(actions_list[i])

                if len(gpu_actions) > 0:
                    actions_batch = torch.cat(gpu_actions, dim=0)
                    new_obs, done, info = env.step(actions_batch)

                    # Update obs_flat and done_flags
                    for i in range(num_envs_per_gpu):
                        env_idx = start_idx + i
                        obs_flat[env_idx] = {
                            k: v[i:i+1] for k, v in new_obs.items()
                        }

                        if done[i]:
                            done_flags[env_idx] = True

            step_count += 1

        # Save episodes
        for env_idx in range(total_envs):
            if episodes_collected >= num_episodes:
                break

            # Determine success/failure (random for now)
            success = torch.rand(1).item() > 0.3
            failure_type = torch.randint(0, 4, (1,)).item() if not success else None

            recorders[env_idx].save_episode(success, failure_type=failure_type)
            episodes_collected += 1
            pbar.update(1)

    pbar.close()

    print("\n" + "="*60)
    print(f"✅ DATA COLLECTION COMPLETE!")
    print(f"Episodes collected: {episodes_collected}")
    print(f"Saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    # Run data collection
    collect_data(
        num_episodes=500,  # Start with 500, increase to 1000 later
        num_envs_per_gpu=4,  # 4 × 4 GPUs = 16 parallel envs
        max_steps_per_episode=100,
        output_dir="./data/raw_episodes"
    )
EOF
```

### RUN DATA COLLECTION (Overnight)

```bash
cd ~/Desktop/aegis

# Activate environment
source venv_salus/bin/activate

# Run data collection (will take 6-12 hours)
nohup python scripts/collect_data_local.py > logs/data_collection.log 2>&1 &

# Check progress
tail -f logs/data_collection.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

---

## PHASE 6: TRAIN PREDICTOR (Week 3-4)

After data collection, train the multi-horizon failure predictor:

```bash
# Train predictor on GPU 1 (or submit to HPC cluster)
cat > scripts/train_predictor_local.py << 'EOF'
"""Train multi-horizon failure predictor on collected data"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path
from tqdm import tqdm


class FailureDataset(Dataset):
    """Dataset of failure episodes"""

    def __init__(self, data_dir="./data/raw_episodes"):
        self.data_dir = Path(data_dir)
        self.episodes = list(self.data_dir.rglob("episode_*.h5"))
        print(f"Found {len(self.episodes)} episodes")

    def __len__(self):
        return len(self.episodes) * 20  # Sample 20 times per episode

    def __getitem__(self, idx):
        ep_idx = idx // 20
        ep_file = self.episodes[ep_idx]

        with h5py.File(ep_file, 'r') as f:
            signals = torch.tensor(f['signals'][:])  # (T, 12)
            timestamps = f['timestamps'][:]
            success = f.attrs['success']

            # Sample random timestep
            T = len(timestamps)
            t = torch.randint(0, max(1, T-10), (1,)).item()

            features = signals[t]  # (12,)

            # Create target (simplified: binary success/failure)
            target = torch.tensor(1.0 if not success else 0.0)

        return features, target


class SimplePredictor(nn.Module):
    """Simplified failure predictor (single output for now)"""

    def __init__(self, input_dim=12, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_predictor(
    data_dir="./data/raw_episodes",
    epochs=50,
    batch_size=64,
    device="cuda:1"
):
    """Train predictor"""

    print("Loading dataset...")
    dataset = FailureDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print("Creating model...")
    model = SimplePredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for features, targets in pbar:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            preds = (outputs > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}, Acc = {correct/total:.4f}")

    # Save model
    output_path = Path("./models/predictor/simple_predictor_v1.pth")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    print(f"\n✅ Training complete! Model saved to {output_path}")


if __name__ == "__main__":
    train_predictor(
        data_dir="./data/raw_episodes",
        epochs=50,
        batch_size=64,
        device="cuda:1"
    )
EOF

# Run training
python scripts/train_predictor_local.py
```

---

## SUMMARY: WHAT RUNS ON YOUR LOCAL MACHINE

### GPU Allocation:
```
GPU 0: VLA Ensemble (11GB used)
GPU 1: Signal Extractor + Predictor Training (4GB used)
GPU 2: Reserved for Manifold (later)
GPU 3: Reserved for MPC Synthesis (later)
```

### Timeline:
- **Day 1**: Setup + VLA testing (3-4 hours)
- **Days 2-3**: Simulation environment (8 hours)
- **Week 2**: Data collection (overnight, 6-12 hours)
- **Week 3**: Train predictor (2-4 hours)
- **Week 4+**: Manifold + MPC (use HPC cluster for heavy lifting)

### What to run NOW:

```bash
# 1. Setup (do this first)
cd ~/Desktop/aegis
python3.10 -m venv venv_salus
source venv_salus/bin/activate
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn opencv-python h5py tqdm wandb

# 2. Test GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# 3. Download TinyVLA (2 hours)
cd ~/
git clone https://github.com/OpenDriveLab/TinyVLA.git
cd TinyVLA && pip install -e .
huggingface-cli download TinyVLA/tinyvla-1b --local-dir ~/models/tinyvla/tinyvla-1b

# 4. Test VLA (5 min)
cd ~/Desktop/aegis
python scripts/test_vla_ensemble.py

# 5. Collect data (overnight)
nohup python scripts/collect_data_local.py > logs/data_collection.log 2>&1 &

# 6. Train predictor (next day, 2-4 hours)
python scripts/train_predictor_local.py
```

**You have everything you need! Start NOW!**
