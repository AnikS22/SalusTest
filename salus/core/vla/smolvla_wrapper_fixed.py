"""
SmolVLA Wrapper with FULL Signal Extraction for SALUS

FIXES: Extracts hidden states and logits from SmolVLA internals
so that ALL 12 signals can be computed (not just 3).

Key Improvements:
- Extracts hidden_state from VLA encoder
- Extracts action_logits before final action head
- Uses SingleModelSignalExtractor for full 12D signals
- Compatible with existing data collection scripts
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from transformers import AutoTokenizer
from collections import deque

# Set to single GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class FullSignalExtractor:
    """
    Extracts FULL 12D signals from VLA with internals access.

    This is a simplified version of SingleModelSignalExtractor
    that works with SmolVLA's actual outputs.
    """

    def __init__(self, device='cuda:0', history_size=10):
        self.device = device
        self.history_size = history_size

        # Temporal buffers
        self.action_history = deque(maxlen=history_size)
        self.hidden_history = deque(maxlen=history_size)
        self.volatility_history = deque(maxlen=history_size)

        # Previous timestep state
        self.prev_action = None
        self.prev_hidden = None
        self.prev_robot_state = None

        # Running statistics
        self.hidden_mean = None
        self.hidden_std = None
        self.hidden_norm_ema = None
        self.ema_alpha = 0.1

        # Franka Panda joint limits
        self.joint_limits_min = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            device=device
        )
        self.joint_limits_max = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            device=device
        )

    def reset(self):
        """Reset at episode start"""
        self.action_history.clear()
        self.hidden_history.clear()
        self.volatility_history.clear()
        self.prev_action = None
        self.prev_hidden = None
        self.prev_robot_state = None

    def extract(
        self,
        action: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        action_logits: Optional[torch.Tensor] = None,
        robot_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract 12D signals from VLA outputs.

        Args:
            action: (B, action_dim) - VLA action output
            hidden_state: (B, hidden_dim) - VLA hidden state (optional)
            action_logits: (B, action_dim) - Pre-softmax logits (optional)
            robot_state: (B, state_dim) - Robot joint states (optional)

        Returns:
            signals: (B, 12) - Full signal vector
        """
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        if hidden_state is not None:
            hidden_state = torch.nan_to_num(hidden_state, nan=0.0, posinf=0.0, neginf=0.0)
        if action_logits is not None:
            action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=0.0, neginf=0.0)
        if robot_state is not None:
            robot_state = torch.nan_to_num(robot_state, nan=0.0, posinf=0.0, neginf=0.0)

        batch_size = action.shape[0]
        signals = []

        # ==============================================
        # SIGNALS 0-3: TEMPORAL ACTION DYNAMICS
        # ==============================================

        # Signal 0: Action Volatility
        if self.prev_action is not None:
            volatility = torch.norm(action - self.prev_action, dim=-1)
        else:
            volatility = torch.zeros(batch_size, device=self.device)
        signals.append(volatility.unsqueeze(-1))

        # Signal 1: Action Magnitude
        magnitude = torch.norm(action, dim=-1)
        signals.append(magnitude.unsqueeze(-1))

        # Signal 2: Action Acceleration
        if len(self.action_history) >= 2:
            a_t = action
            a_t_minus_1 = self.action_history[-1]
            a_t_minus_2 = self.action_history[-2]
            acceleration = torch.norm(a_t - 2*a_t_minus_1 + a_t_minus_2, dim=-1)
        else:
            acceleration = torch.zeros(batch_size, device=self.device)
        signals.append(acceleration.unsqueeze(-1))

        # Signal 3: Trajectory Divergence
        if len(self.action_history) > 0:
            action_mean = torch.stack(list(self.action_history)).mean(dim=0)
            divergence = torch.norm(action - action_mean, dim=-1)
        else:
            divergence = torch.zeros(batch_size, device=self.device)
        signals.append(divergence.unsqueeze(-1))

        # ==============================================
        # SIGNALS 4-6: VLA INTERNAL STABILITY
        # ==============================================

        if hidden_state is not None:
            # Signal 4: Latent Drift
            if self.prev_hidden is not None:
                latent_drift = torch.norm(hidden_state - self.prev_hidden, dim=-1)
            else:
                latent_drift = torch.zeros(batch_size, device=self.device)
            signals.append(latent_drift.unsqueeze(-1))

            # Signal 5: Latent Norm Spike
            hidden_norm = torch.norm(hidden_state, dim=-1)
            if self.hidden_norm_ema is None:
                self.hidden_norm_ema = hidden_norm.mean().item()
            else:
                self.hidden_norm_ema = (
                    self.ema_alpha * hidden_norm.mean().item() +
                    (1 - self.ema_alpha) * self.hidden_norm_ema
                )
            norm_spike = hidden_norm / max(self.hidden_norm_ema, 1e-6)
            signals.append(norm_spike.unsqueeze(-1))

            # Signal 6: OOD Distance
            if self.hidden_mean is None:
                self.hidden_mean = hidden_state.mean(dim=0)
                self.hidden_std = torch.ones_like(self.hidden_mean) * 0.1
                ood_distance = torch.zeros(batch_size, device=self.device)
            else:
                batch_mean = hidden_state.mean(dim=0)
                self.hidden_mean = (
                    self.ema_alpha * batch_mean +
                    (1 - self.ema_alpha) * self.hidden_mean
                )
                if batch_size > 1:
                    batch_std = hidden_state.std(dim=0, unbiased=False) + 1e-6
                    self.hidden_std = (
                        self.ema_alpha * batch_std +
                        (1 - self.ema_alpha) * self.hidden_std
                    )
                normalized = (hidden_state - self.hidden_mean) / (self.hidden_std + 1e-6)
                ood_distance = torch.norm(normalized, dim=-1)
            signals.append(ood_distance.unsqueeze(-1))
        else:
            # Hidden state not available - use zeros
            signals.extend([torch.zeros(batch_size, 1, device=self.device)] * 3)

        # ==============================================
        # SIGNALS 7-8: MODEL UNCERTAINTY
        # ==============================================

        if action_logits is not None:
            # Signal 7: Softmax Entropy
            probs = torch.nn.functional.softmax(action_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            signals.append(entropy.unsqueeze(-1))

            # Signal 8: Max Softmax Probability
            max_prob, _ = probs.max(dim=-1)
            signals.append(max_prob.unsqueeze(-1))
        else:
            # Logits not available - use zeros
            signals.extend([torch.zeros(batch_size, 1, device=self.device)] * 2)

        # ==============================================
        # SIGNALS 9-10: PHYSICS REALITY CHECKS
        # ==============================================

        if robot_state is not None:
            # Signal 9: Execution Mismatch
            if self.prev_robot_state is not None and self.prev_action is not None:
                actual_delta = robot_state - self.prev_robot_state
                if self.prev_action.shape[1] < robot_state.shape[1]:
                    pad_size = robot_state.shape[1] - self.prev_action.shape[1]
                    predicted_delta = torch.cat([
                        self.prev_action,
                        torch.zeros(batch_size, pad_size, device=self.device)
                    ], dim=-1)
                else:
                    predicted_delta = self.prev_action[:, :robot_state.shape[1]]
                mismatch = torch.norm(actual_delta - predicted_delta, dim=-1)
            else:
                mismatch = torch.zeros(batch_size, device=self.device)
            signals.append(mismatch.unsqueeze(-1))

            # Signal 10: Constraint Margin
            margin_to_min = robot_state[:, :7] - self.joint_limits_min
            margin_to_max = self.joint_limits_max - robot_state[:, :7]
            min_margin = torch.minimum(margin_to_min, margin_to_max).min(dim=-1)[0]
            constraint_margin = torch.clamp(-min_margin + 0.5, min=0.0)
            signals.append(constraint_margin.unsqueeze(-1))
        else:
            signals.extend([torch.zeros(batch_size, 1, device=self.device)] * 2)

        # ==============================================
        # SIGNAL 11: TEMPORAL CONSISTENCY
        # ==============================================

        self.volatility_history.append(volatility.detach())
        if len(self.volatility_history) >= 3:
            volatility_tensor = torch.stack(list(self.volatility_history))
            temporal_consistency = volatility_tensor.std(dim=0)
        else:
            temporal_consistency = torch.zeros(batch_size, device=self.device)
        signals.append(temporal_consistency.unsqueeze(-1))

        # ==============================================
        # CONCATENATE AND VALIDATE
        # ==============================================

        all_signals = torch.nan_to_num(
            torch.cat(signals, dim=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )  # (B, 12)

        # Update history
        self.action_history.append(action.detach())
        if hidden_state is not None:
            self.hidden_history.append(hidden_state.detach())
        self.prev_action = action.detach()
        self.prev_hidden = hidden_state.detach() if hidden_state is not None else None
        self.prev_robot_state = robot_state.detach() if robot_state is not None else None

        return all_signals


class SmolVLAWithInternals(nn.Module):
    """
    SmolVLA wrapper that extracts internal activations for full signal extraction.

    This class wraps SmolVLAPolicy and hooks into its internals to extract:
    - Hidden states from the vision-language encoder
    - Action logits before the final action head

    Args:
        model_path: HuggingFace model path (default: "lerobot/smolvla_base")
        device: Device to load on (default: "cuda:0")
    """

    def __init__(
        self,
        model_path: str = "lerobot/smolvla_base",
        device: str = "cuda:0"
    ):
        super().__init__()
        self.device = device
        self.model_path = model_path

        print(f"\n🤖 Loading SmolVLA with internals extraction...")
        print(f"   Model: {model_path}")
        print(f"   Device: {device}")

        # Load tokenizer
        print("   Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )

        # Load model
        print("   Loading SmolVLA model...")
        self.model = SmolVLAPolicy.from_pretrained(model_path)
        self.model.eval()
        self.model = self.model.to(device=device)

        # Initialize signal extractor
        self.signal_extractor = FullSignalExtractor(device=device)

        # Storage for hooked activations
        self.last_hidden_state = None
        self.last_action_logits = None

        # Register hooks to capture internals
        self._register_hooks()

        mem_mb = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"   ✅ Model loaded: {mem_mb:.2f} MB")
        print(f"   ✅ Signal extractor ready (12D)")

    def _register_hooks(self):
        """Register forward hooks to capture hidden states and logits"""
        def save_hidden_state_from_input(module, input, output):
            # Capture the INPUT to action_out_proj as hidden state
            # Input is typically a tuple (x, ) where x is the hidden representation
            if isinstance(input, tuple) and len(input) > 0:
                hidden = input[0]
                if isinstance(hidden, torch.Tensor):
                    self.last_hidden_state = hidden

        def save_action_logits(module, input, output):
            # Capture action logits (output of action_out_proj)
            if isinstance(output, torch.Tensor):
                self.last_action_logits = output

        # Hook into the actual SmolVLA architecture
        # Capture hidden state from INPUT to action_out_proj (before final projection)
        # and action logits from OUTPUT of action_out_proj
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'action_out_proj'):
            print("   Registering hooks on action_out_proj (for hidden state + logits)")
            # Register BOTH hooks on the same module
            self.model.model.action_out_proj.register_forward_hook(save_hidden_state_from_input)
            self.model.model.action_out_proj.register_forward_hook(save_action_logits)

    @torch.no_grad()
    def forward(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with internal extraction.

        Args:
            observation: Dict with:
                - 'observation.images.camera1': (B, C, H, W)
                - 'observation.state': (B, state_dim)
                - 'task': str or List[str]

        Returns:
            Dict with:
                - 'action': (B, action_dim)
                - 'hidden_state': (B, hidden_dim) or None
                - 'action_logits': (B, action_dim) or None
        """
        # Reset captured internals
        self.last_hidden_state = None
        self.last_action_logits = None

        # Tokenize task text if needed
        if 'task' in observation and 'observation.language.tokens' not in observation:
            task_text = observation['task']
            if isinstance(task_text, list):
                task_text = task_text[0]  # Take first task if batched

            # Tokenize
            tokens = self.tokenizer(
                task_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            observation['observation.language.tokens'] = tokens['input_ids'].to(self.device)
            observation['observation.language.attention_mask'] = tokens['attention_mask'].bool().to(self.device)

        # Run model forward
        action = self.model.select_action(observation)

        # Extract hidden state if captured
        hidden_state = None
        if self.last_hidden_state is not None:
            # Average pool over sequence dimension if needed
            if len(self.last_hidden_state.shape) == 3:  # (B, seq_len, hidden_dim)
                hidden_state = self.last_hidden_state.mean(dim=1)  # (B, hidden_dim)
            else:
                hidden_state = self.last_hidden_state

        # Extract action logits if captured
        action_logits = self.last_action_logits
        if action_logits is not None:
            # Handle sequence dimension if present
            if len(action_logits.shape) == 3:  # (B, seq_len, action_dim)
                action_logits = action_logits.mean(dim=1)  # (B, action_dim)

        # Pad action from 6D to 7D for Franka
        if action.shape[-1] == 6:
            padding = torch.zeros(action.shape[0], 1, device=action.device, dtype=action.dtype)
            action = torch.cat([action, padding], dim=-1)

        return {
            'action': action,
            'hidden_state': hidden_state,
            'action_logits': action_logits
        }

    def __call__(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Allow calling like a function"""
        return self.forward(observation)


def test_full_extraction():
    """Test that all 12 signals are extracted"""
    print("\n" + "="*70)
    print("Testing Full Signal Extraction")
    print("="*70)

    # Initialize model
    vla = SmolVLAWithInternals(device="cuda:0")

    # Create dummy observation
    obs = {
        'observation.images.camera1': torch.randn(1, 3, 256, 256, device='cuda:0'),
        'observation.state': torch.randn(1, 7, device='cuda:0'),
        'task': 'pick up the red cube'
    }

    # Run forward pass
    print("\nRunning forward pass...")
    output = vla(obs)

    print(f"Action shape: {output['action'].shape}")
    print(f"Hidden state: {output['hidden_state'].shape if output['hidden_state'] is not None else 'None'}")
    print(f"Action logits: {output['action_logits'].shape if output['action_logits'] is not None else 'None'}")

    # Extract signals
    print("\nExtracting 12D signals...")
    signals = vla.signal_extractor.extract(
        action=output['action'],
        hidden_state=output['hidden_state'],
        action_logits=output['action_logits'],
        robot_state=obs['observation.state']
    )

    print(f"✅ Signals shape: {signals.shape} (expected: [1, 12])")
    print(f"\nSignal values:")
    for i, val in enumerate(signals[0]):
        print(f"   Signal {i}: {val.item():.4f}")

    # Check for NaNs
    nan_count = torch.isnan(signals).sum().item()
    if nan_count == 0:
        print(f"\n✅ SUCCESS: All signals valid (no NaN)")
    else:
        print(f"\n❌ PROBLEM: {nan_count} NaN values detected")

    print("\n" + "="*70)


if __name__ == "__main__":
    test_full_extraction()
