"""
Single-Model Signal Extractor for SALUS

Extracts 12D uncertainty and dynamics signals from a SINGLE VLA model forward pass.
No ensemble, no perturbation testing - just temporal dynamics and internal model signals.

Key Features:
- 1 forward pass per timestep (8x faster than ensemble)
- Extracts softmax entropy as primary uncertainty signal
- Uses temporal action volatility instead of ensemble variance
- Leverages VLA internal hidden states for instability detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional, Dict
import numpy as np


class SingleModelSignalExtractor(nn.Module):
    """
    Extract 12D signals from single VLA model for SALUS failure prediction.

    Signal Breakdown:
    [1-4]  Temporal Action Dynamics (4D): volatility, magnitude, acceleration, divergence
    [5-7]  VLA Internal Stability (3D): latent drift, norm spike, OOD distance
    [8-9]  Model Uncertainty (2D): softmax entropy, max probability
    [10-11] Physics Reality Checks (2D): execution mismatch, constraint margin
    [12]    Temporal Consistency (1D): rolling action volatility std

    Args:
        device: torch device ('cuda' or 'cpu')
        history_size: Number of past timesteps to maintain (default: 10)
        ema_alpha: Exponential moving average smoothing factor (default: 0.1)
    """

    def __init__(
        self,
        device: torch.device,
        history_size: int = 10,
        ema_alpha: float = 0.1
    ):
        super().__init__()
        self.device = device
        self.history_size = history_size
        self.ema_alpha = ema_alpha

        # Temporal buffers (deque for efficient sliding window)
        self.action_history = deque(maxlen=history_size)
        self.hidden_history = deque(maxlen=history_size)
        self.volatility_history = deque(maxlen=history_size)  # For signal 12

        # Previous timestep state
        self.prev_action = None
        self.prev_hidden = None
        self.prev_robot_state = None

        # Running statistics for normalization (EMA)
        self.hidden_mean = None  # Running mean of hidden states
        self.hidden_std = None   # Running std of hidden states
        self.hidden_norm_ema = None  # Running mean of hidden norms

        # Franka Panda joint limits (rad) - for constraint margin
        self.joint_limits_min = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            device=device
        )
        self.joint_limits_max = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            device=device
        )

        self.reset()

    def reset(self):
        """Reset extractor state (call at episode start)."""
        self.action_history.clear()
        self.hidden_history.clear()
        self.volatility_history.clear()
        self.prev_action = None
        self.prev_hidden = None
        self.prev_robot_state = None
        # Don't reset running stats (maintained across episodes)

    def extract(
        self,
        vla_output: Dict[str, torch.Tensor],
        robot_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract 12D signals from single VLA forward pass.

        Args:
            vla_output: Dict containing:
                - 'action': (B, action_dim) - VLA action output
                - 'hidden_state': (B, hidden_dim) - Optional VLA internal representation
                - 'action_logits': (B, action_dim) - Optional pre-softmax logits
            robot_state: (B, state_dim) - Current robot joint states (optional)

        Returns:
            signals: (B, 12) - 12D signal vector
        """
        action = vla_output['action']  # (B, action_dim)
        hidden = vla_output.get('hidden_state')  # (B, hidden_dim) or None
        logits = vla_output.get('action_logits')  # (B, action_dim) or None

        batch_size = action.shape[0]
        signals = []

        # ============================================================
        # SIGNALS 1-4: TEMPORAL ACTION DYNAMICS
        # ============================================================

        # Signal 1: Temporal Action Volatility (replaces epistemic uncertainty)
        # Measures action instability over time
        if self.prev_action is not None:
            volatility = torch.norm(action - self.prev_action, dim=-1)  # (B,)
        else:
            volatility = torch.zeros(batch_size, device=self.device)
        signals.append(volatility.unsqueeze(-1))

        # Signal 2: Action Magnitude
        # Physical scale of commanded motion
        magnitude = torch.norm(action, dim=-1)  # (B,)
        signals.append(magnitude.unsqueeze(-1))

        # Signal 3: Action Acceleration
        # Detects sudden policy changes (2nd derivative of action)
        if len(self.action_history) >= 2:
            a_t = action  # Current
            a_t_minus_1 = self.action_history[-1]  # t-1
            a_t_minus_2 = self.action_history[-2]  # t-2
            acceleration = torch.norm(a_t - 2*a_t_minus_1 + a_t_minus_2, dim=-1)  # (B,)
        else:
            acceleration = torch.zeros(batch_size, device=self.device)
        signals.append(acceleration.unsqueeze(-1))

        # Signal 4: Trajectory Divergence
        # Deviation from recent average behavior
        if len(self.action_history) > 0:
            action_mean = torch.stack(list(self.action_history)).mean(dim=0)  # (B, action_dim)
            divergence = torch.norm(action - action_mean, dim=-1)  # (B,)
        else:
            divergence = torch.zeros(batch_size, device=self.device)
        signals.append(divergence.unsqueeze(-1))

        # ============================================================
        # SIGNALS 5-7: VLA INTERNAL STABILITY
        # ============================================================

        if hidden is not None:
            # Signal 5: Latent Drift
            # Changes in VLA internal representation
            if self.prev_hidden is not None:
                latent_drift = torch.norm(hidden - self.prev_hidden, dim=-1)  # (B,)
            else:
                latent_drift = torch.zeros(batch_size, device=self.device)
            signals.append(latent_drift.unsqueeze(-1))

            # Signal 6: Latent Norm Spike
            # Unusual activation magnitudes indicate uncertainty/OOD
            hidden_norm = torch.norm(hidden, dim=-1)  # (B,)

            # Update running norm EMA
            if self.hidden_norm_ema is None:
                self.hidden_norm_ema = hidden_norm.mean().item()
            else:
                self.hidden_norm_ema = (
                    self.ema_alpha * hidden_norm.mean().item() +
                    (1 - self.ema_alpha) * self.hidden_norm_ema
                )

            # Normalized spike (ratio to expected norm)
            norm_spike = hidden_norm / max(self.hidden_norm_ema, 1e-6)  # (B,)
            signals.append(norm_spike.unsqueeze(-1))

            # Signal 7: OOD Distance
            # Distance from normal latent space distribution
            if self.hidden_mean is None:
                # Initialize with first batch
                self.hidden_mean = hidden.mean(dim=0)  # (hidden_dim,)
                self.hidden_std = torch.ones_like(self.hidden_mean) * 0.1
                ood_distance = torch.zeros(batch_size, device=self.device)
            else:
                # Update running mean with EMA
                batch_mean = hidden.mean(dim=0)
                self.hidden_mean = (
                    self.ema_alpha * batch_mean +
                    (1 - self.ema_alpha) * self.hidden_mean
                )

                # Update running std with EMA (only if batch_size > 1)
                if batch_size > 1:
                    batch_std = hidden.std(dim=0, unbiased=False) + 1e-6
                    self.hidden_std = (
                        self.ema_alpha * batch_std +
                        (1 - self.ema_alpha) * self.hidden_std
                    )

                # Mahalanobis-like distance
                normalized = (hidden - self.hidden_mean) / (self.hidden_std + 1e-6)
                ood_distance = torch.norm(normalized, dim=-1)  # (B,)

            signals.append(ood_distance.unsqueeze(-1))

        else:
            # Hidden states not available - fill with zeros
            signals.append(torch.zeros(batch_size, 1, device=self.device))  # Signal 5
            signals.append(torch.zeros(batch_size, 1, device=self.device))  # Signal 6
            signals.append(torch.zeros(batch_size, 1, device=self.device))  # Signal 7

        # ============================================================
        # SIGNALS 8-9: MODEL UNCERTAINTY (from action distribution)
        # ============================================================

        if logits is not None:
            # Signal 8: Softmax Entropy (PRIMARY UNCERTAINTY SIGNAL)
            # High entropy = flat distribution = model uncertain
            probs = F.softmax(logits, dim=-1)  # (B, action_dim)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # (B,)
            signals.append(entropy.unsqueeze(-1))

            # Signal 9: Max Softmax Probability (SECONDARY UNCERTAINTY)
            # Low max prob = model uncertain
            max_prob, _ = probs.max(dim=-1)  # (B,)
            signals.append(max_prob.unsqueeze(-1))
        else:
            # Logits not available - fill with zeros
            signals.append(torch.zeros(batch_size, 1, device=self.device))  # Signal 8
            signals.append(torch.zeros(batch_size, 1, device=self.device))  # Signal 9

        # ============================================================
        # SIGNALS 10-11: PHYSICS REALITY CHECKS
        # ============================================================

        if robot_state is not None:
            # Signal 10: Execution Mismatch
            # Difference between predicted and actual state change
            if self.prev_robot_state is not None and self.prev_action is not None:
                actual_delta = robot_state - self.prev_robot_state  # (B, state_dim)

                # Use previous action as prediction proxy
                # (In reality, would use learned forward model)
                # Pad action to match robot_state dimensions
                if self.prev_action.shape[1] < robot_state.shape[1]:
                    pad_size = robot_state.shape[1] - self.prev_action.shape[1]
                    predicted_delta = torch.cat([
                        self.prev_action,
                        torch.zeros(batch_size, pad_size, device=self.device)
                    ], dim=-1)
                else:
                    predicted_delta = self.prev_action[:, :robot_state.shape[1]]

                mismatch = torch.norm(actual_delta - predicted_delta, dim=-1)  # (B,)
            else:
                mismatch = torch.zeros(batch_size, device=self.device)
            signals.append(mismatch.unsqueeze(-1))

            # Signal 11: Constraint Margin
            # Distance to joint limits (safety signal)
            margin_to_min = robot_state[:, :7] - self.joint_limits_min  # (B, 7)
            margin_to_max = self.joint_limits_max - robot_state[:, :7]  # (B, 7)
            min_margin = torch.minimum(margin_to_min, margin_to_max).min(dim=-1)[0]  # (B,)

            # Clamp negative (closer to limit = higher signal)
            constraint_margin = torch.clamp(-min_margin + 0.5, min=0.0)  # (B,)
            signals.append(constraint_margin.unsqueeze(-1))
        else:
            # Robot state not available
            signals.append(torch.zeros(batch_size, 1, device=self.device))  # Signal 10
            signals.append(torch.zeros(batch_size, 1, device=self.device))  # Signal 11

        # ============================================================
        # SIGNAL 12: TEMPORAL CONSISTENCY
        # ============================================================

        # Rolling std of action volatility over window
        # Low std = consistent, high std = erratic
        self.volatility_history.append(volatility.detach())

        if len(self.volatility_history) >= 3:
            volatility_tensor = torch.stack(list(self.volatility_history))  # (T, B)
            temporal_consistency = volatility_tensor.std(dim=0)  # (B,)
        else:
            temporal_consistency = torch.zeros(batch_size, device=self.device)

        signals.append(temporal_consistency.unsqueeze(-1))

        # ============================================================
        # CONCATENATE ALL SIGNALS
        # ============================================================

        all_signals = torch.cat(signals, dim=-1)  # (B, 12)

        # Sanity checks
        assert all_signals.shape == (batch_size, 12), \
            f"Expected (B, 12), got {all_signals.shape}"
        assert not torch.isnan(all_signals).any(), "Signals contain NaN"
        assert not torch.isinf(all_signals).any(), "Signals contain Inf"

        # ============================================================
        # UPDATE HISTORY
        # ============================================================

        self.action_history.append(action.detach())
        if hidden is not None:
            self.hidden_history.append(hidden.detach())
        self.prev_action = action.detach()
        self.prev_hidden = hidden.detach() if hidden is not None else None
        self.prev_robot_state = robot_state.detach() if robot_state is not None else None

        return all_signals


    def get_signal_names(self) -> list:
        """Return human-readable names for all 12 signals."""
        return [
            "1. Temporal Action Volatility",
            "2. Action Magnitude",
            "3. Action Acceleration",
            "4. Trajectory Divergence",
            "5. Latent Drift",
            "6. Latent Norm Spike",
            "7. OOD Distance",
            "8. Softmax Entropy (Uncertainty)",
            "9. Max Softmax Probability",
            "10. Execution Mismatch",
            "11. Constraint Margin",
            "12. Temporal Consistency"
        ]
