"""
SALUS Failure Predictor
Neural network that predicts multi-horizon failures from VLA uncertainty signals

Architecture:
  Input: 12D uncertainty signals
  Encoder: 3-layer MLP [12 → 128 → 256 → 128]
  Decoder: Multi-horizon prediction heads
  Output: 16D failure logits (4 horizons × 4 failure types)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class SALUSPredictor(nn.Module):
    """
    SALUS Failure Predictor

    Predicts failures at multiple time horizons from VLA uncertainty signals
    """

    def __init__(
        self,
        signal_dim: int = 12,
        hidden_dims: list = [128, 256, 128],
        num_horizons: int = 4,
        num_failure_types: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            signal_dim: Input signal dimensionality (12D from SignalExtractor)
            hidden_dims: Hidden layer dimensions for encoder
            num_horizons: Number of prediction horizons (4: short/medium/long/very-long)
            num_failure_types: Number of failure classes (4: collision/drop/miss/timeout)
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.signal_dim = signal_dim
        self.num_horizons = num_horizons
        self.num_failure_types = num_failure_types
        self.output_dim = num_horizons * num_failure_types  # 16D

        # Encoder: 3-layer MLP
        encoder_layers = []
        in_dim = signal_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Multi-horizon decoder heads
        # Separate head for each horizon allows specialization
        self.horizon_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], num_failure_types)
            for _ in range(num_horizons)
        ])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, signals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            signals: (B, 12) uncertainty signals from VLA

        Returns:
            Dict with:
                - 'logits': (B, 16) raw logits for all horizon-failure combinations
                - 'probs': (B, 4, 4) probabilities per horizon and failure type
                - 'horizon_probs': (B, 4) probability of ANY failure per horizon
                - 'max_prob': (B,) maximum failure probability across all predictions
                - 'predicted_horizon': (B,) which horizon has highest failure risk
                - 'predicted_type': (B,) which failure type is most likely
        """
        # Encode signals
        features = self.encoder(signals)  # (B, hidden_dim)

        # Decode into multi-horizon predictions
        horizon_logits = []
        for head in self.horizon_heads:
            logits = head(features)  # (B, num_failure_types)
            horizon_logits.append(logits)

        # Stack horizon predictions
        horizon_logits = torch.stack(horizon_logits, dim=1)  # (B, num_horizons, num_failure_types)

        # Flatten for output
        logits_flat = horizon_logits.reshape(-1, self.output_dim)  # (B, 16)

        # Convert to probabilities
        probs = torch.sigmoid(horizon_logits)  # (B, num_horizons, num_failure_types)

        # Probability of ANY failure per horizon (max across failure types)
        horizon_probs = probs.max(dim=-1)[0]  # (B, num_horizons)

        # Overall maximum failure probability
        max_prob, max_idx = horizon_probs.max(dim=-1)  # (B,)

        # Predicted horizon and failure type
        batch_indices = torch.arange(signals.shape[0], device=signals.device)
        predicted_horizon = max_idx
        predicted_type = probs[batch_indices, predicted_horizon].argmax(dim=-1)

        return {
            'logits': logits_flat,
            'probs': probs,
            'horizon_probs': horizon_probs,
            'max_prob': max_prob,
            'predicted_horizon': predicted_horizon,
            'predicted_type': predicted_type,
            'features': features  # For visualization/analysis
        }

    def predict_failure(
        self,
        signals: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Predict whether a failure will occur

        Args:
            signals: (B, 12) uncertainty signals
            threshold: Probability threshold for positive prediction

        Returns:
            Dict with:
                - 'failure_predicted': (B,) bool, whether failure is predicted
                - 'failure_horizon': (B,) int, predicted failure horizon
                - 'failure_type': (B,) int, predicted failure type
                - 'confidence': (B,) float, prediction confidence
        """
        with torch.no_grad():
            output = self.forward(signals)

            # Predict failure if max_prob exceeds threshold
            failure_predicted = output['max_prob'] > threshold

            return {
                'failure_predicted': failure_predicted,
                'failure_horizon': output['predicted_horizon'],
                'failure_type': output['predicted_type'],
                'confidence': output['max_prob']
            }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    Focuses training on hard examples and rare classes (failures)
    """

    def __init__(self, alpha: float = 2.0, gamma: float = 2.0):
        """
        Args:
            alpha: Weight for positive class (failures are rare, so weight them more)
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss

        Args:
            logits: (B, C) raw predictions
            targets: (B, C) binary labels

        Returns:
            loss: scalar loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Focal loss formula: -α * (1 - p_t)^γ * log(p_t)
        # where p_t = p if y=1, else 1-p

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight for positive class
        alpha_weight = targets * self.alpha + (1 - targets) * 1.0

        # Focal loss
        focal_loss = alpha_weight * focal_weight * bce

        return focal_loss.mean()


class MultiHorizonFocalLoss(nn.Module):
    """
    Focal Loss for multi-horizon prediction

    Applies focal loss independently to each horizon
    """

    def __init__(
        self,
        alpha: float = 2.0,
        gamma: float = 2.0,
        horizon_weights: Optional[list] = None
    ):
        """
        Args:
            alpha: Weight for positive class
            gamma: Focusing parameter
            horizon_weights: Optional weights per horizon [w1, w2, w3, w4]
                            Default: [1.0, 1.0, 1.0, 1.0] (equal weight)
        """
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

        if horizon_weights is None:
            horizon_weights = [1.0, 1.0, 1.0, 1.0]

        self.register_buffer('horizon_weights', torch.tensor(horizon_weights))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-horizon focal loss

        Args:
            logits: (B, 16) predictions (4 horizons × 4 failure types)
            targets: (B, 4, 4) labels per horizon and failure type

        Returns:
            loss: scalar loss value
            loss_dict: dict with per-horizon losses for logging
        """
        batch_size = logits.shape[0]

        # Reshape logits to (B, 4, 4)
        logits = logits.reshape(batch_size, 4, 4)

        # Compute loss per horizon
        horizon_losses = []
        loss_dict = {}

        for h in range(4):
            h_logits = logits[:, h, :]  # (B, 4)
            h_targets = targets[:, h, :]  # (B, 4)

            h_loss = self.focal_loss(h_logits, h_targets)
            weighted_loss = h_loss * self.horizon_weights[h]

            horizon_losses.append(weighted_loss)
            loss_dict[f'loss_h{h+1}'] = h_loss.item()

        # Total loss: weighted sum across horizons
        total_loss = torch.stack(horizon_losses).sum()

        loss_dict['loss_total'] = total_loss.item()

        return total_loss, loss_dict


# Test the predictor
if __name__ == "__main__":
    print("🧪 Testing SALUS Predictor...\n")

    # Check CUDA
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Create predictor
    predictor = SALUSPredictor(
        signal_dim=12,
        hidden_dims=[128, 256, 128],
        num_horizons=4,
        num_failure_types=4
    ).to(device)

    print("📊 Model Architecture:")
    print(f"   Parameters: {sum(p.numel() for p in predictor.parameters()):,}")
    print(f"   Input: 12D signals")
    print(f"   Output: 16D logits (4 horizons × 4 failure types)")
    print()

    # Test forward pass
    print("🔄 Testing forward pass...")
    batch_size = 8
    signals = torch.randn(batch_size, 12, device=device)

    output = predictor(signals)

    print(f"   Input shape: {signals.shape}")
    print(f"   Output logits shape: {output['logits'].shape}")
    print(f"   Output probs shape: {output['probs'].shape}")
    print(f"   Horizon probs shape: {output['horizon_probs'].shape}")
    print(f"   Max prob range: [{output['max_prob'].min():.3f}, {output['max_prob'].max():.3f}]")
    print()

    # Test prediction
    print("🔮 Testing failure prediction...")
    predictions = predictor.predict_failure(signals, threshold=0.5)

    num_failures = predictions['failure_predicted'].sum().item()
    print(f"   Failures predicted: {num_failures}/{batch_size}")
    print(f"   Confidence range: [{predictions['confidence'].min():.3f}, {predictions['confidence'].max():.3f}]")
    print()

    # Test loss
    print("📉 Testing focal loss...")
    focal_loss = MultiHorizonFocalLoss(alpha=2.0, gamma=2.0)

    # Create dummy targets (some failures)
    targets = torch.zeros(batch_size, 4, 4, device=device)
    targets[0, 0, 0] = 1.0  # Collision at horizon 1
    targets[1, 1, 1] = 1.0  # Drop at horizon 2
    targets[2, 2, 2] = 1.0  # Miss at horizon 3

    loss, loss_dict = focal_loss(output['logits'], targets)

    print(f"   Total loss: {loss.item():.4f}")
    for key, value in loss_dict.items():
        print(f"   {key}: {value:.4f}")
    print()

    print("✅ SALUS Predictor test passed!")
    print("\n📋 Next steps:")
    print("   1. Build training infrastructure")
    print("   2. Train on collected data")
    print("   3. Evaluate on test set")
    print("   4. Deploy with adaptation module")
