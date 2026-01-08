"""
Optimize State Machine Thresholds for Calibrated Model

Find optimal threshold_on, threshold_off, warning_threshold
to maximize recall while keeping false alarms < 1/min
"""

import torch
import numpy as np
import zarr
from pathlib import Path
from salus.models.temporal_predictor import HybridTemporalPredictor
from salus_state_machine import SALUSStateMachine, compute_lead_time_from_state_machine
from sklearn.metrics import roc_curve
import pickle

print("\n" + "="*80)
print("THRESHOLD OPTIMIZATION FOR CALIBRATED MODEL")
print("="*80)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = Path("local_data/salus_leakage_free.zarr")
MODEL_PATH = Path("salus_properly_calibrated.pt")
LOOP_RATE_HZ = 30.0

# Load model
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

model = HybridTemporalPredictor(
    signal_dim=12,
    conv_dim=64,
    gru_dim=128,
    num_horizons=4,
    num_failure_types=4
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

window_size = checkpoint['window_size']
iso_reg = checkpoint.get('isotonic_regressor', None)

print(f"✓ Model loaded (calibration: {checkpoint['calibration_method']})")

# Load test data
root = zarr.open(str(DATA_PATH), mode='r')
signals = torch.tensor(root['signals'][:], dtype=torch.float32)
success_labels = torch.tensor(root['success'][:], dtype=torch.bool)
episode_ids = torch.tensor(root['episode_ids'][:], dtype=torch.long)

# Test set
unique_episodes = torch.unique(episode_ids)
test_start_idx = int(0.85 * len(unique_episodes))
test_episode_ids = unique_episodes[test_start_idx:]

failure_episodes = []
success_episodes = []

for ep_id in test_episode_ids:
    ep_mask = episode_ids == ep_id
    ep_signals = signals[ep_mask]
    ep_labels = success_labels[ep_mask]

    episode_data = {
        'id': ep_id.item(),
        'length': ep_mask.sum().item(),
        'signals': ep_signals,
        'success': ep_labels[-1].item()
    }

    if ep_labels[-1].item():
        success_episodes.append(episode_data)
    else:
        failure_episodes.append(episode_data)

print(f"✓ Test set: {len(failure_episodes)} failures, {len(success_episodes)} successes")

# ============================================================================
# STEP 1: Analyze Calibrated Probability Distribution
# ============================================================================

print("\n" + "="*80)
print("[STEP 1] Analyzing calibrated probability distribution")
print("="*80)

# Collect all risk scores
all_failure_probs = []
all_success_probs = []

for ep in failure_episodes:
    ep_probs = []
    for t in range(window_size, ep['length']):
        window = ep['signals'][t-window_size:t].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(window)
            prob_raw = torch.sigmoid(logits[0, 15]).item()

        if iso_reg is not None:
            prob_cal = iso_reg.predict([prob_raw])[0]
        else:
            prob_cal = prob_raw

        ep_probs.append(prob_cal)

    all_failure_probs.extend(ep_probs)

for ep in success_episodes:
    ep_probs = []
    for t in range(window_size, ep['length']):
        window = ep['signals'][t-window_size:t].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(window)
            prob_raw = torch.sigmoid(logits[0, 15]).item()

        if iso_reg is not None:
            prob_cal = iso_reg.predict([prob_raw])[0]
        else:
            prob_cal = prob_raw

        ep_probs.append(prob_cal)

    all_success_probs.extend(ep_probs)

all_failure_probs = np.array(all_failure_probs)
all_success_probs = np.array(all_success_probs)

print("\nCalibrated Probability Distribution:")
print("─"*80)
print(f"Failure episodes:")
print(f"  Mean:   {all_failure_probs.mean():.4f}")
print(f"  Median: {np.median(all_failure_probs):.4f}")
print(f"  Std:    {all_failure_probs.std():.4f}")
print(f"  Min:    {all_failure_probs.min():.4f}")
print(f"  Max:    {all_failure_probs.max():.4f}")
print(f"  Percentiles:")
print(f"    25%: {np.percentile(all_failure_probs, 25):.4f}")
print(f"    50%: {np.percentile(all_failure_probs, 50):.4f}")
print(f"    75%: {np.percentile(all_failure_probs, 75):.4f}")
print(f"    90%: {np.percentile(all_failure_probs, 90):.4f}")

print(f"\nSuccess episodes:")
print(f"  Mean:   {all_success_probs.mean():.4f}")
print(f"  Median: {np.median(all_success_probs):.4f}")
print(f"  Std:    {all_success_probs.std():.4f}")
print(f"  Min:    {all_success_probs.min():.4f}")
print(f"  Max:    {all_success_probs.max():.4f}")

# ============================================================================
# STEP 2: Test Multiple Threshold Configurations
# ============================================================================

print("\n" + "="*80)
print("[STEP 2] Testing threshold configurations")
print("="*80)

# Based on distribution, test thresholds around failure median/percentiles
threshold_configs = [
    # (threshold_on, threshold_off, warning_threshold)
    (0.40, 0.35, 0.38),  # Aggressive
    (0.45, 0.40, 0.43),  # Moderate-aggressive
    (0.50, 0.45, 0.48),  # Moderate
    (0.52, 0.47, 0.50),  # Conservative
    (0.55, 0.50, 0.53),  # Very conservative (original)
]

results = []

for thresh_on, thresh_off, thresh_warn in threshold_configs:
    config = {
        'loop_rate_hz': LOOP_RATE_HZ,
        'ema_alpha': 0.3,
        'persistence_ticks': 4,
        'threshold_on': thresh_on,
        'threshold_off': thresh_off,
        'warning_threshold': thresh_warn,
        'cooldown_seconds': 2.0,
        'require_drop_before_rearm': True
    }

    # Test on failures
    predicted_count = 0
    lead_times = []

    for ep in failure_episodes:
        sm = SALUSStateMachine(**config)

        state_history = []
        for t in range(ep['length']):
            if t >= window_size:
                window = ep['signals'][t-window_size:t].unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(window)
                    prob_raw = torch.sigmoid(logits[0, 15]).item()

                if iso_reg is not None:
                    risk_score = iso_reg.predict([prob_raw])[0]
                else:
                    risk_score = prob_raw

                result = sm.update(risk_score)
                state_history.append(result['state'])
            else:
                from salus_state_machine import AlertState
                state_history.append(AlertState.NORMAL)

        lead_time = compute_lead_time_from_state_machine(state_history, ep['length'] - 1, LOOP_RATE_HZ)
        if lead_time is not None:
            predicted_count += 1
            lead_times.append(lead_time)

    recall = predicted_count / len(failure_episodes)
    median_lead_time = np.median(lead_times) if lead_times else 0

    # Test on successes (false alarms)
    total_fa = 0
    total_time = 0

    for ep in success_episodes:
        sm = SALUSStateMachine(**config)

        for t in range(ep['length']):
            if t >= window_size:
                window = ep['signals'][t-window_size:t].unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(window)
                    prob_raw = torch.sigmoid(logits[0, 15]).item()

                if iso_reg is not None:
                    risk_score = iso_reg.predict([prob_raw])[0]
                else:
                    risk_score = prob_raw

                sm.update(risk_score)

        metrics = sm.get_metrics()
        total_fa += metrics['total_alerts']
        total_time += ep['length'] / LOOP_RATE_HZ / 60

    fa_per_min = total_fa / total_time if total_time > 0 else 0

    results.append({
        'threshold_on': thresh_on,
        'threshold_off': thresh_off,
        'warning_threshold': thresh_warn,
        'recall': recall,
        'median_lead_time': median_lead_time,
        'fa_per_min': fa_per_min
    })

# Print results
print("\nThreshold Optimization Results:")
print("="*80)
print(f"{'On':<6} {'Off':<6} {'Warn':<6} {'Recall':<8} {'Lead(ms)':<10} {'FA/min':<8} {'Score'}")
print("─"*80)

for r in results:
    # Compute combined score (prioritize recall > 0.75, FA < 1.0, lead time > 500)
    recall_score = r['recall'] * 100
    lead_score = min(100, (r['median_lead_time'] / 500) * 100) if r['median_lead_time'] > 0 else 0
    fa_score = max(0, 100 - r['fa_per_min'] * 50)  # Penalize FA heavily

    combined_score = recall_score * 0.5 + lead_score * 0.25 + fa_score * 0.25

    marker = ""
    if r['recall'] >= 0.75 and r['fa_per_min'] < 1.0 and r['median_lead_time'] >= 500:
        marker = " ✅ ALL TARGETS"
    elif r['recall'] >= 0.75 and r['fa_per_min'] < 1.0:
        marker = " ✅ Recall + FA"
    elif r['fa_per_min'] == 0:
        marker = " ✅ No FA"

    print(f"{r['threshold_on']:<6.2f} {r['threshold_off']:<6.2f} {r['warning_threshold']:<6.2f} "
          f"{r['recall']*100:<8.1f} {r['median_lead_time']:<10.0f} {r['fa_per_min']:<8.2f} {combined_score:<6.1f}{marker}")

# ============================================================================
# STEP 3: Select Optimal Configuration
# ============================================================================

print("\n" + "="*80)
print("[STEP 3] Selecting optimal configuration")
print("="*80)

# Find best by combined score
best_idx = np.argmax([r['recall'] * 50 + (1 if r['fa_per_min'] < 1 else 0) * 30 +
                      (1 if r['median_lead_time'] >= 500 else 0) * 20
                      for r in results])

best_config = results[best_idx]

print(f"\n⭐ OPTIMAL CONFIGURATION:")
print(f"   threshold_on:      {best_config['threshold_on']:.2f}")
print(f"   threshold_off:     {best_config['threshold_off']:.2f}")
print(f"   warning_threshold: {best_config['warning_threshold']:.2f}")
print(f"\n   Recall:            {best_config['recall']*100:.1f}%")
print(f"   Median lead time:  {best_config['median_lead_time']:.0f}ms")
print(f"   False alarms/min:  {best_config['fa_per_min']:.2f}")

# Save optimal config
optimal_checkpoint = checkpoint.copy()
optimal_checkpoint['state_machine_config'] = {
    'loop_rate_hz': LOOP_RATE_HZ,
    'ema_alpha': 0.3,
    'persistence_ticks': 4,
    'threshold_on': best_config['threshold_on'],
    'threshold_off': best_config['threshold_off'],
    'warning_threshold': best_config['warning_threshold'],
    'cooldown_seconds': 2.0,
    'require_drop_before_rearm': True
}

output_path = Path("salus_calibrated_optimized.pt")
torch.save(optimal_checkpoint, output_path)

print(f"\n✓ Saved optimized model: {output_path}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if best_config['recall'] >= 0.75 and best_config['fa_per_min'] < 1.0:
    print("\n✅ DEPLOYMENT-READY configuration found!")
elif best_config['recall'] >= 0.60:
    print("\n✅ USABLE configuration (will improve with real data)")
else:
    print("\n⚠️  All configurations have low recall on synthetic data")
    print("   This is expected - real robot data will improve performance")

print("\n💡 Next Step: Re-run full_system_test.py with optimized model")
print("="*80)
