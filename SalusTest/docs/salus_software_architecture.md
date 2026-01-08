# GUARDIAN Software Architecture

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GUARDIAN SYSTEM                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────▼────────┐         ┌────────▼────────┐        ┌────────▼────────┐
│  Data Layer    │         │  Runtime Layer  │        │  Training Layer │
│                │         │                 │        │                 │
│ • Recorder     │────────▶│ • VLA Wrapper   │◀───────│ • Predictor     │
│ • Labeler      │         │ • Predictor     │        │ • Manifold      │
│ • Storage      │         │ • Synthesizer   │        │ • Dynamics      │
│ • Validator    │         │ • Monitor       │        │ • Federated     │
└────────────────┘         └─────────────────┘        └─────────────────┘
        │                           │                           │
        │                           │                           │
        ▼                           ▼                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                             │
│  • ROS2 (Robot Control)                                            │
│  • Isaac Lab (Simulation)                                          │
│  • NVIDIA Warp (Physics)                                           │
│  • OpenFL (Federation)                                             │
│  • Ray Serve (Deployment)                                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```bash
guardian/
│
├── config/                      # Configuration files
│   ├── robot/
│   │   ├── g1_humanoid.yaml
│   │   └── sensors.yaml
│   ├── models/
│   │   ├── predictor.yaml
│   │   ├── manifold.yaml
│   │   └── dynamics.yaml
│   └── deployment/
│       ├── gpu_allocation.yaml
│       └── ray_config.yaml
│
├── core/                        # Core system components (IP-sensitive)
│   ├── __init__.py
│   ├── predictor/              # Multi-horizon failure predictor
│   │   ├── __init__.py
│   │   ├── model.py           # PyTorch model definition
│   │   ├── signals.py         # Signal extraction
│   │   ├── ensemble.py        # Ensemble wrapper
│   │   └── calibration.py     # Temperature scaling
│   │
│   ├── manifold/              # Learned safety manifold
│   │   ├── __init__.py
│   │   ├── encoder.py         # Latent space encoder
│   │   ├── decoder.py         # Action space decoder
│   │   ├── sampler.py         # Manifold-guided sampling
│   │   └── contrastive.py     # Training logic
│   │
│   ├── synthesis/             # Counterfactual action synthesis
│   │   ├── __init__.py
│   │   ├── mpc.py            # MPC implementation
│   │   ├── dynamics.py       # Learned dynamics model
│   │   ├── rollout.py        # Parallel GPU rollouts
│   │   └── validator.py      # Self-validation logic
│   │
│   └── vla/                   # VLA wrapper
│       ├── __init__.py
│       ├── wrapper.py        # Ensemble VLA with internals
│       ├── attention.py      # Attention extraction
│       └── uncertainty.py    # Model uncertainty
│
├── data/                      # Data handling
│   ├── __init__.py
│   ├── recorder.py           # Multi-modal data recording
│   ├── labeler.py            # Failure labeling GUI
│   ├── dataset.py            # PyTorch dataset classes
│   ├── augmentation.py       # Data augmentation
│   └── loader.py             # Data loading utilities
│
├── training/                  # Training scripts
│   ├── __init__.py
│   ├── train_predictor.py
│   ├── train_manifold.py
│   ├── train_dynamics.py
│   └── federated.py          # Federated learning
│
├── deployment/                # Deployment/runtime
│   ├── __init__.py
│   ├── runtime.py            # Main runtime loop
│   ├── monitor.py            # Performance monitoring
│   ├── intervention.py       # Intervention logging
│   └── ray_serve.py          # Ray Serve endpoints
│
├── simulation/                # Isaac Lab integration
│   ├── __init__.py
│   ├── tasks.py              # Task definitions
│   ├── envs.py               # Environment configs
│   ├── randomization.py      # Domain randomization
│   └── replay.py             # Counterfactual replay
│
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── gpu.py                # GPU management
│   ├── metrics.py            # Evaluation metrics
│   ├── visualization.py      # Plotting/visualization
│   └── ros_bridge.py         # ROS2 integration
│
├── scripts/                   # Standalone scripts
│   ├── collect_data.py
│   ├── label_failures.py
│   ├── evaluate.py
│   └── deploy.py
│
├── tests/                     # Unit tests
│   ├── test_predictor.py
│   ├── test_manifold.py
│   ├── test_synthesis.py
│   └── test_integration.py
│
├── models/                    # Saved models (gitignored, IP-sensitive)
│   ├── predictor/
│   │   ├── checkpoint_epoch_100.pth
│   │   └── best_model.pth
│   ├── manifold/
│   └── dynamics/
│
├── data/                      # Data storage (gitignored)
│   ├── raw/                  # Raw episodes
│   ├── processed/            # Processed datasets
│   └── labels/               # Failure annotations
│
├── notebooks/                 # Jupyter notebooks (analysis)
│   ├── data_exploration.ipynb
│   ├── model_analysis.ipynb
│   └── results_visualization.ipynb
│
├── requirements.txt
├── setup.py
├── README.md
├── LICENSE                    # Proprietary license
└── .gitignore
```

## Core Components (Detailed)

### 1. VLA Wrapper (core/vla/wrapper.py)

```python
"""
VLA Wrapper with Internal Signal Extraction
Loads ensemble of frozen VLA models and exposes internals
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class VLAEnsemble(nn.Module):
    """
    Manages ensemble of VLA models with internal signal extraction

    GPU Allocation: GPU 0 (11GB)
    - 5 VLA models @ 2.2GB each = 11GB
    """

    def __init__(
        self,
        model_name: str = "tinyvla-1b",
        ensemble_size: int = 5,
        device: str = "cuda:0"
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.device = device

        # Load ensemble (frozen)
        self.models = nn.ModuleList([
            self._load_vla(model_name) for _ in range(ensemble_size)
        ])

        # Freeze all parameters
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        # Move to GPU 0
        self.to(device)

    def _load_vla(self, model_name: str) -> nn.Module:
        """Load single VLA model"""
        # Import based on model type
        if "tinyvla" in model_name.lower():
            from tinyvla import TinyVLA
            return TinyVLA.from_pretrained(model_name)
        elif "openvla" in model_name.lower():
            from openvla import OpenVLA
            return OpenVLA.from_pretrained(model_name)
        else:
            raise ValueError(f"Unknown VLA model: {model_name}")

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        language: str,
        extract_internals: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through ensemble

        Args:
            obs: {'rgb': (B,3,H,W), 'proprio': (B,D)}
            language: Task description
            extract_internals: Whether to return attention/hidden states

        Returns:
            actions: (B, ensemble_size, action_dim)
            internals: {
                'attention': (B, K, num_heads, seq_len, seq_len),
                'hidden': (B, K, seq_len, hidden_dim),
                'action_mean': (B, action_dim),
                'action_var': (B, action_dim)
            }
        """
        actions = []
        attentions = []
        hiddens = []

        with torch.no_grad():  # VLA is frozen
            for model in self.models:
                output = model(
                    obs,
                    language,
                    output_attentions=extract_internals,
                    output_hidden_states=extract_internals
                )

                actions.append(output.action)
                if extract_internals:
                    attentions.append(output.attentions[-1])
                    hiddens.append(output.hidden_states[-2])

        actions = torch.stack(actions, dim=1)  # (B, K, action_dim)

        if extract_internals:
            internals = {
                'attention': torch.stack(attentions, dim=1),
                'hidden': torch.stack(hiddens, dim=1),
                'action_mean': actions.mean(dim=1),
                'action_var': actions.var(dim=1)  # Model uncertainty
            }
            return actions, internals
        else:
            return actions.mean(dim=1), None


class SignalExtractor:
    """
    Extract 12D failure precursor signal vector

    Runs on CPU/GPU 1 (lightweight)
    """

    def __init__(self, device: str = "cuda:1"):
        self.device = device
        # Load any auxiliary models (e.g., object detector)
        # self.object_detector = load_model(...)

    def extract(
        self,
        obs: Dict[str, torch.Tensor],
        vla_internals: Dict[str, torch.Tensor],
        history: list,
        depth: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract 12D feature vector

        Returns:
            features: (B, 12) tensor
                [σ²_ens, σ²_action,           # Epistemic (2)
                 H_attn, d_attn,              # Attention (2)
                 d_traj, d_learned,           # Trajectory (2)
                 d_obs, F_grip, occ, 0, 0, 0] # Environment (6, padded)
        """
        batch_size = obs['rgb'].shape[0]
        features = torch.zeros(batch_size, 12, device=self.device)

        # 1. Model uncertainty
        features[:, 0] = vla_internals['action_var'].mean(dim=1)
        features[:, 1] = vla_internals['action_var'].max(dim=1)[0]

        # 2. Attention degradation
        attention = vla_internals['attention'].mean(dim=1)  # Avg over ensemble
        features[:, 2] = self._compute_entropy(attention)
        features[:, 3] = self._compute_misalignment(attention)

        # 3. Trajectory divergence
        if len(history) > 0:
            traj_embedding = self._encode_trajectory(history)
            features[:, 4] = self._trajectory_distance(traj_embedding)

        # 4. Environmental risk
        features[:, 6] = self._obstacle_distance(depth)
        features[:, 7] = torch.norm(obs['proprio'][:, -6:], dim=1)  # Wrench

        return features

    def _compute_entropy(self, attention: torch.Tensor) -> torch.Tensor:
        """Compute attention entropy"""
        # attention: (B, num_heads, seq_len, seq_len)
        attn_probs = torch.softmax(attention.mean(dim=1), dim=-1)
        entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1).mean(dim=-1)
        return entropy

    def _obstacle_distance(self, depth: torch.Tensor) -> torch.Tensor:
        """Minimum distance to obstacles from depth map"""
        # depth: (B, H, W)
        # Filter robot pixels (center 30% of image)
        H, W = depth.shape[1:]
        mask = torch.ones_like(depth, dtype=torch.bool)
        mask[:, int(0.35*H):int(0.65*H), int(0.35*W):int(0.65*W)] = False

        obstacle_depth = depth.masked_select(mask).view(depth.shape[0], -1)
        min_dist = obstacle_depth.min(dim=1)[0]
        return min_dist
```

### 2. Multi-Horizon Predictor (core/predictor/model.py)

```python
"""
Multi-Horizon Failure Predictor
Predicts (failure_type, time_horizon) distribution

GPU Allocation: GPU 1 (4GB)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHorizonPredictor(nn.Module):
    """
    Predicts p(failure_type_i, horizon_j | signals)

    Input: (B, 12) signal features
    Output: (B, num_types, num_horizons) probabilities
    """

    FAILURE_TYPES = ['collision', 'wrong_object', 'grasp_failure', 'goal_miss']
    HORIZONS_MS = [200, 300, 400, 500]

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        num_types: int = 4,
        num_horizons: int = 4,
        device: str = "cuda:1"
    ):
        super().__init__()
        self.num_types = num_types
        self.num_horizons = num_horizons
        self.device = device

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Per-type prediction heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_horizons)
            for _ in range(num_types)
        ])

        # Temperature for calibration
        self.register_buffer('temperature', torch.ones(1))

        self.to(device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, 12) signal features

        Returns:
            probs: (B, num_types, num_horizons)
        """
        features = features.to(self.device)

        # Encode
        h = self.encoder(features)  # (B, hidden_dim)

        # Per-type predictions
        logits = []
        for head in self.heads:
            type_logits = head(h)  # (B, num_horizons)
            logits.append(type_logits)

        logits = torch.stack(logits, dim=1)  # (B, num_types, num_horizons)

        # Temperature scaling
        logits = logits / self.temperature

        # Softmax over horizons per type
        probs = torch.softmax(logits, dim=-1)

        return probs

    def should_intervene(
        self,
        probs: torch.Tensor,
        thresholds: dict = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Determine intervention based on adaptive thresholds

        Args:
            probs: (B, num_types, num_horizons)
            thresholds: Dict mapping type -> threshold (default: type-dependent)

        Returns:
            intervene: (B,) bool - whether to intervene
            failure_type: (B,) int - predicted failure type (0-3)
            horizon_idx: (B,) int - predicted horizon (0-3)
        """
        if thresholds is None:
            # Default adaptive thresholds
            thresholds = {
                0: 0.5,  # Collision - low threshold (critical)
                1: 0.7,  # Wrong object
                2: 0.7,  # Grasp failure
                3: 0.8   # Goal miss - high threshold (recoverable)
            }

        batch_size = probs.shape[0]

        # Max probability per type
        max_probs_per_type, _ = probs.max(dim=-1)  # (B, num_types)

        # Check threshold per type
        thresh_tensor = torch.tensor([
            thresholds[i] for i in range(self.num_types)
        ], device=probs.device).unsqueeze(0)

        # Does any type exceed threshold?
        exceeds_thresh = max_probs_per_type > thresh_tensor  # (B, num_types)
        intervene = exceeds_thresh.any(dim=-1)  # (B,)

        # Get most critical failure type (argmax over types)
        failure_type = max_probs_per_type.argmax(dim=-1)  # (B,)

        # Get horizon for that failure type
        horizon_idx = torch.gather(
            probs.argmax(dim=-1),  # (B, num_types)
            dim=1,
            index=failure_type.unsqueeze(1)
        ).squeeze(1)  # (B,)

        return intervene, failure_type, horizon_idx


class PredictorEnsemble:
    """
    Ensemble of K predictors for uncertainty estimation
    """

    def __init__(self, ensemble_size: int = 5, **model_kwargs):
        self.models = [
            MultiHorizonPredictor(**model_kwargs)
            for _ in range(ensemble_size)
        ]
        self.ensemble_size = ensemble_size

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            probs_mean: (B, num_types, num_horizons) - ensemble mean
            probs_std: (B, num_types, num_horizons) - ensemble std (uncertainty)
        """
        all_probs = []

        for model in self.models:
            probs = model(features)
            all_probs.append(probs)

        all_probs = torch.stack(all_probs, dim=0)  # (K, B, num_types, num_horizons)

        probs_mean = all_probs.mean(dim=0)
        probs_std = all_probs.std(dim=0)

        return probs_mean, probs_std
```

### 3. Safety Manifold (core/manifold/encoder.py)

```python
"""
Learned Safety Manifold
Encodes (state, action) pairs into low-dimensional safe subspace

GPU Allocation: GPU 2 (3GB)
"""

import torch
import torch.nn as nn

class SafetyManifoldEncoder(nn.Module):
    """
    Encoder: (s, a) → z  (13D + 23D → 8D)
    """

    def __init__(
        self,
        state_dim: int = 23,  # joint_pos(13) + joint_vel(13) + gripper(6) + obj_pose(6)
        action_dim: int = 13,
        latent_dim: int = 8,
        device: str = "cuda:2"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        input_dim = state_dim + action_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, latent_dim)
        )

        self.to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, 23)
            action: (B, 13)

        Returns:
            z: (B, 8) latent embedding
        """
        state = state.to(self.device)
        action = action.to(self.device)

        x = torch.cat([state, action], dim=-1)  # (B, 36)
        z = self.encoder(x)

        return z


class SafetyManifoldDecoder(nn.Module):
    """
    Decoder: z → a'  (8D → 13D)
    """

    def __init__(
        self,
        latent_dim: int = 8,
        action_dim: int = 13,
        device: str = "cuda:2"
    ):
        super().__init__()
        self.device = device

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Action bounds [-1, 1]
        )

        self.to(device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, 8) latent

        Returns:
            action: (B, 13)
        """
        z = z.to(self.device)
        action = self.decoder(z)
        return action


class ManifoldGuidedSampler:
    """
    Sample safe action candidates using learned manifold
    """

    def __init__(
        self,
        encoder: SafetyManifoldEncoder,
        decoder: SafetyManifoldDecoder,
        num_candidates: int = 15,
        latent_noise_std: float = 0.1
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.num_candidates = num_candidates
        self.latent_noise_std = latent_noise_std

    def sample(
        self,
        state: torch.Tensor,
        unsafe_action: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample M' action candidates around unsafe_action in manifold

        Args:
            state: (B, 23)
            unsafe_action: (B, 13)

        Returns:
            candidates: (B, M', 13)
        """
        batch_size = state.shape[0]

        # Encode unsafe action
        z_unsafe = self.encoder(state, unsafe_action)  # (B, 8)

        # Sample perturbations in latent space
        z_candidates = []
        for _ in range(self.num_candidates):
            noise = torch.randn_like(z_unsafe) * self.latent_noise_std
            z_perturbed = z_unsafe + noise
            z_candidates.append(z_perturbed)

        z_candidates = torch.stack(z_candidates, dim=1)  # (B, M', 8)

        # Decode to action space
        # Reshape for decoder: (B*M', 8)
        z_flat = z_candidates.reshape(batch_size * self.num_candidates, -1)
        actions_flat = self.decoder(z_flat)  # (B*M', 13)

        # Reshape back: (B, M', 13)
        candidates = actions_flat.reshape(batch_size, self.num_candidates, -1)

        # Add small exploration noise
        candidates = candidates + torch.randn_like(candidates) * 0.01
        candidates = torch.clamp(candidates, -1, 1)  # Respect action bounds

        return candidates
```

### 4. MPC Synthesis (core/synthesis/mpc.py)

```python
"""
Model Predictive Control with Learned Dynamics
Parallel rollouts on GPU

GPU Allocation: GPU 3 (8GB)
- 15 parallel environments @ 500MB each = 7.5GB
"""

import torch
import torch.nn as nn
import warp as wp  # NVIDIA Warp for GPU physics

class LearnedDynamics(nn.Module):
    """
    Forward dynamics: (s_t, a_t) → s_{t+1}
    """

    def __init__(
        self,
        state_dim: int = 23,
        action_dim: int = 13,
        hidden_dim: int = 256,
        device: str = "cuda:3"
    ):
        super().__init__()
        self.device = device

        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, state_dim)
        )

        # Residual connection (predict delta)
        self.use_residual = True

        # Self-validation buffer
        self.validation_buffer = []

        self.to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, 23)
            action: (B, 13)

        Returns:
            next_state: (B, 23)
        """
        state = state.to(self.device)
        action = action.to(self.device)

        x = torch.cat([state, action], dim=-1)
        delta = self.dynamics(x)

        if self.use_residual:
            next_state = state + delta
        else:
            next_state = delta

        return next_state

    def validate_and_update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        true_next_state: torch.Tensor,
        error_threshold: float = 0.05
    ):
        """
        Self-validation: check prediction error and store mismatches

        Args:
            state: (B, 23) - state before action
            action: (B, 13) - executed action
            true_next_state: (B, 23) - observed next state
            error_threshold: When to flag for retraining
        """
        # Predict
        pred_next_state = self.forward(state, action)

        # Compute error
        error = torch.norm(pred_next_state - true_next_state, dim=-1)  # (B,)

        # Store high-error transitions for retraining
        high_error_mask = error > error_threshold

        if high_error_mask.any():
            for i in range(len(state)):
                if high_error_mask[i]:
                    self.validation_buffer.append({
                        'state': state[i].cpu(),
                        'action': action[i].cpu(),
                        'next_state': true_next_state[i].cpu(),
                        'error': error[i].item()
                    })

        return error.mean().item()


class MPCSynthesizer:
    """
    MPC-based counterfactual action synthesis
    Uses manifold-guided candidates + learned dynamics rollouts
    """

    def __init__(
        self,
        dynamics: LearnedDynamics,
        predictor: MultiHorizonPredictor,
        manifold_sampler: ManifoldGuidedSampler,
        vla: VLAEnsemble,
        rollout_horizon: int = 5,
        device: str = "cuda:3"
    ):
        self.dynamics = dynamics
        self.predictor = predictor
        self.manifold_sampler = manifold_sampler
        self.vla = vla
        self.rollout_horizon = rollout_horizon
        self.device = device

    def synthesize(
        self,
        state: torch.Tensor,
        unsafe_action: torch.Tensor,
        obs: dict,
        language: str,
        safety_threshold: float = 0.3
    ) -> torch.Tensor:
        """
        Synthesize safe alternative action via MPC

        Args:
            state: (B, 23) current robot state
            unsafe_action: (B, 13) VLA's proposed action (flagged as unsafe)
            obs: Observation dict for VLA continuation
            language: Task description
            safety_threshold: Max allowed failure probability

        Returns:
            safe_action: (B, 13) or None if no safe action found
        """
        batch_size = state.shape[0]

        # 1. Sample candidates using manifold
        candidates = self.manifold_sampler.sample(state, unsafe_action)  # (B, M', 13)
        M = candidates.shape[1]

        # 2. Parallel rollouts for each candidate
        best_actions = []
        best_scores = []

        for b in range(batch_size):
            s_init = state[b:b+1].repeat(M, 1)  # (M, 23)
            cands_b = candidates[b]  # (M, 13)

            # Rollout each candidate
            safe_mask = torch.ones(M, dtype=torch.bool, device=self.device)
            task_scores = torch.zeros(M, device=self.device)

            s = s_init
            for t in range(self.rollout_horizon):
                # Apply candidate action
                s_next = self.dynamics(s, cands_b)  # (M, 23)

                # Get VLA continuation (for task score)
                # This is approximate - could use learned value function instead
                # For now, check if still safe

                # TODO: Extract signals and check safety
                # For now, simplified version

                s = s_next

            # Select best safe candidate
            if safe_mask.any():
                best_idx = task_scores[safe_mask].argmax()
                best_actions.append(cands_b[safe_mask][best_idx])
                best_scores.append(task_scores[safe_mask][best_idx])
            else:
                # No safe action found - return HALT (zeros)
                best_actions.append(torch.zeros(13, device=self.device))
                best_scores.append(torch.tensor(-1.0, device=self.device))

        safe_actions = torch.stack(best_actions)  # (B, 13)
        return safe_actions
```

Would you like me to continue with:
1. **Runtime Integration** (deployment/runtime.py) - main loop
2. **Data Collection** (data/recorder.py) - ROS2 integration
3. **Training Pipeline** (training/) - HPC scripts
4. **GPU Allocation Config** (config/deployment/)
5. **Isaac Lab Integration** (simulation/tasks.py)
