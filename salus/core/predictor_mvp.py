"""
SALUS MVP Predictor
Simplified failure predictor for V1

Input: 6D signals (not 12D)
Output: 4D failure probabilities (one per failure type)
No multi-horizon - just predict if failure will happen this episode

This is MUCH simpler for MVP!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SALUSPredictorMVP(nn.Module):
    """
    Simple MLP predictor for SALUS MVP

    Much simpler than full version:
    - 6D input (not 12D)
    - 4D output (failure types, no horizons)
    - Smaller network (faster training)
    """

    def __init__(
        self,
        signal_dim: int = 6,
        hidden_dim: int = 64,  # Smaller for MVP
        num_failure_types: int = 4
    ):
        super().__init__()

        self.signal_dim = signal_dim
        self.num_failure_types = num_failure_types

        # Simple 2-layer MLP
        self.net = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_failure_types)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, signals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            signals: (B, 6) uncertainty signals

        Returns:
            Dict with:
                - 'logits': (B, 4) raw logits per failure type
                - 'probs': (B, 4) probabilities per failure type
                - 'predicted_type': (B,) most likely failure type
                - 'max_prob': (B,) maximum failure probability
        """
        logits = self.net(signals)  # (B, 4)
        probs = torch.sigmoid(logits)  # (B, 4)

        max_prob, predicted_type = probs.max(dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'predicted_type': predicted_type,
            'max_prob': max_prob
        }

    def predict(self, signals: torch.Tensor, threshold: float = 0.5) -> Dict:
        """
        Predict if failure will occur

        Args:
            signals: (B, 6)
            threshold: Probability threshold

        Returns:
            Dict with predictions
        """
        with torch.no_grad():
            output = self.forward(signals)

            failure_predicted = output['max_prob'] > threshold

            return {
                'failure_predicted': failure_predicted,
                'failure_type': output['predicted_type'],
                'confidence': output['max_prob']
            }


class SimpleBCELoss(nn.Module):
    """
    Simple Binary Cross Entropy Loss with class weights

    Simpler than focal loss - good for MVP
    """

    def __init__(self, pos_weight: float = 2.0):
        """
        Args:
            pos_weight: Weight for positive class (failures are rare)
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss

        Args:
            logits: (B, 4) predictions
            targets: (B, 4) labels

        Returns:
            loss: scalar
        """
        # Weighted BCE
        pos_weight = torch.ones(4, device=logits.device) * self.pos_weight
        loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pos_weight
        )

        return loss


# Test
if __name__ == "__main__":
    print("🧪 Testing SALUS MVP Predictor...\n")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Create predictor
    predictor = SALUSPredictorMVP(
        signal_dim=6,
        hidden_dim=64,
        num_failure_types=4
    ).to(device)

    print("📊 Model:")
    print(f"   Parameters: {sum(p.numel() for p in predictor.parameters()):,}")
    print(f"   Input: 6D signals")
    print(f"   Output: 4D failure probabilities")
    print()

    # Test forward
    batch_size = 8
    signals = torch.randn(batch_size, 6, device=device)

    output = predictor(signals)

    print("🔄 Forward pass:")
    print(f"   Input: {signals.shape}")
    print(f"   Logits: {output['logits'].shape}")
    print(f"   Probs: {output['probs'].shape}")
    print(f"   Max prob range: [{output['max_prob'].min():.3f}, {output['max_prob'].max():.3f}]")
    print()

    # Test prediction
    predictions = predictor.predict(signals, threshold=0.5)
    print("🔮 Predictions:")
    print(f"   Failures predicted: {predictions['failure_predicted'].sum().item()}/{batch_size}")
    print()

    # Test loss
    criterion = SimpleBCELoss(pos_weight=2.0)

    # Dummy targets
    targets = torch.zeros(batch_size, 4, device=device)
    targets[0, 0] = 1.0  # One failure

    loss = criterion(output['logits'], targets)
    print(f"📉 Loss: {loss.item():.4f}")
    print()

    print("✅ MVP Predictor test passed!")
    print("\n📋 This is MUCH simpler than full version:")
    print("   - 6D input (not 12D)")
    print("   - 4D output (not 16D)")
    print("   - ~4K params (not 70K)")
    print("   - Faster to train")
    print("   - Good for MVP/V1")
