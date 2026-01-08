# SALUS Deployment Usage

## Quick Start

```python
import torch
from salus.models.temporal_predictor import HybridTemporalPredictor

# Load deployment-ready model
checkpoint = torch.load('salus_deployment_ready.pt')

model = HybridTemporalPredictor(
    signal_dim=12,
    conv_dim=64,
    gru_dim=128,
    num_horizons=4,
    num_failure_types=4
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get calibration parameters
temperature = checkpoint['temperature']  # 1.500
threshold = checkpoint['optimal_threshold']  # 0.630
window_size = checkpoint['window_size']  # 20 timesteps

# Inference
with torch.no_grad():
    logits = model(signal_window)  # shape: (1, 16)

    # Apply temperature scaling for calibrated probabilities
    calibrated_probs = torch.sigmoid(logits / temperature)

    # Reshape to (4 horizons, 4 failure types)
    probs_shaped = calibrated_probs.reshape(4, 4)

    # Check 500ms horizon, any failure type
    risk_score = probs_shaped[3].max().item()  # 500ms horizon

    if risk_score > threshold:
        print(f"⚠️  ALERT: Failure risk = {risk_score:.2%}")
        # Trigger safety stop
```

## Deployment Metrics

Tested on 747 episodes:
- AUROC: 0.992
- AUPRC: 0.965
- ECE (calibration): 0.467 ✅
- Mean lead time: 133.7ms
- False alarms: 1.81/min
- Miss rate: 14.0%

## Requirements

- Window size: 20 timesteps (667ms @ 30Hz)
- Must apply temperature scaling: T=1.500
- Alert threshold: 0.630
- Signals: 12D (temporal + internal + uncertainty + physics + consistency)

## Safety Notes

- **Calibration is CRITICAL**: Always use temperature scaling
- **Lead time**: ~134ms average warning before failure
- **False alarms**: Expect ~1.8 per minute at threshold=0.630
- **Missed failures**: ~14% of failures may not be predicted
- **Real robot validation**: Collect data and fine-tune for your robot

## Adjusting Threshold

Current threshold: 0.630 (balanced F1)

Higher threshold (more conservative):
- Fewer false alarms
- More missed failures
- Try: 0.6-0.7

Lower threshold (more cautious):
- More false alarms
- Fewer missed failures
- Try: 0.3-0.4

Always re-calibrate on your robot's data!
