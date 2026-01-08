# SALUS System Design

**Safety Assurance for Learning-based Uncertainty-aware Systems**

---

## System Overview

SALUS is a **failure prediction and adaptation system** for Vision-Language-Action (VLA) models. It predicts failures **before they happen** using uncertainty signals from the VLA ensemble, then adapts the policy to prevent those failures.

---

## Environment: Franka Pick-Place Task

### Task Description
**"Pick up the red cube and place it in the blue bin"**

### Robot
- **Franka Panda**: 7-DOF manipulator
- **Action space**: 7D joint positions
- **Control frequency**: 30 Hz

### Observations
- **Camera 1**: Front view (256x256 RGB)
- **Camera 2**: Side view (256x256 RGB)
- **Camera 3**: Top-down view (256x256 RGB)
- **Robot state**: 7D joint positions

### Failure Types (4 classes)
0. **Collision** - Robot hits object/environment
1. **Drop** - Object dropped during manipulation
2. **Miss** - Failed to grasp object
3. **Timeout** - Task not completed in time

### Success Criteria
- Red cube successfully placed in blue bin
- No collisions during execution
- Completed within 200 timesteps (6.7 seconds)

---

## SALUS Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VLA ENSEMBLE (5 models)                    │
│                   SmolVLA-450M x 5                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ├──► actions (7D)
                      └──► action_variance (model uncertainty)

                      ▼
┌─────────────────────────────────────────────────────────────┐
│              SIGNAL EXTRACTOR (12D features)                 │
│  1. Model uncertainty (internal uncertainty signals)                │
│  2. Action magnitude                                         │
│  3. Action variance                                          │
│  4. Action smoothness                                        │
│  5. Trajectory divergence                                    │
│  6-8. Per-joint variances (first 3 joints)                   │
│  9-12. Rolling uncertainty statistics (mean/std/min/max)     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               SALUS PREDICTOR (Neural Network)               │
│                                                              │
│  Input: 12D uncertainty signals                             │
│  Architecture:                                               │
│    - 3-layer MLP [12 → 128 → 256 → 128]                     │
│    - Multi-horizon decoder                                   │
│  Output: 16D failure logits                                  │
│    - 4 horizons × 4 failure types                           │
│    - Horizons: [6, 10, 13, 16] steps (200-533ms ahead)      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              FAILURE ADAPTATION MODULE                       │
│                                                              │
│  If failure predicted within horizon:                        │
│    1. Estimate time-to-failure                              │
│    2. Select intervention strategy:                          │
│       - Emergency stop                                       │
│       - Slow down execution                                  │
│       - Retry with different approach                        │
│       - Request human assistance                             │
│    3. Execute intervention                                   │
│    4. Log intervention for learning                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Training Phase
```
1. VLA generates actions → Signals extracted → Stored in Zarr
2. Episode completes → Label failures with horizons
3. Train SALUS predictor on (signals, horizon_labels) pairs
4. Evaluate on validation set
```

### Deployment Phase
```
1. VLA generates action → Extract 12D signals
2. SALUS predicts failure probabilities (16D)
3. If P(failure) > threshold:
   → Adaptation module intervenes
4. Execute action or intervention
5. Log outcome for continual learning
```

---

## Multi-Horizon Prediction

SALUS predicts failures at **4 different time horizons**:

| Horizon | Timesteps | Time (30Hz) | Use Case |
|---------|-----------|-------------|----------|
| H1 | 6 | 200ms | Emergency stop |
| H2 | 10 | 333ms | Slow down / adjust |
| H3 | 13 | 433ms | Retry strategy |
| H4 | 16 | 533ms | Early warning |

### Why Multi-Horizon?
- **Short horizon (H1)**: High precision, low false positives, emergency actions
- **Long horizon (H4)**: More time to adapt, can use gentle interventions
- **Ensemble**: Combine predictions for robust decision-making

---

## Training Strategy

### Loss Function
**Multi-Horizon Focal Loss**

```python
focal_loss = -α * (1 - p_t)^γ * log(p_t)

where:
  α = 2.0  (weight for positive class - failures are rare)
  γ = 2.0  (focus on hard examples)
  p_t = prediction probability
```

### Dataset Balance
- Typical ratio: 80-90% success, 10-20% failures
- Use focal loss + oversampling to handle imbalance

### Evaluation Metrics
- **Precision**: % of predicted failures that are real
- **Recall**: % of real failures detected
- **F1-score**: Harmonic mean
- **AUROC**: Area under ROC curve
- **Lead time**: Average warning time before failure

---

## Adaptation Strategies

### 1. Emergency Stop
- **Trigger**: P(failure) > 0.9 at H1 (200ms horizon)
- **Action**: Immediately halt robot
- **Use**: Imminent collision detected

### 2. Slow Down
- **Trigger**: P(failure) > 0.7 at H2-H3 (333-433ms)
- **Action**: Reduce action magnitude by 50%
- **Use**: Uncertain manipulation, shaky grasp

### 3. Retry with Variation
- **Trigger**: P(failure) > 0.6 at H4 (533ms)
- **Action**: Reset to previous safe state, try alternative approach
- **Use**: Predicted grasp failure, try different angle

### 4. Human Assistance
- **Trigger**: Repeated adaptation failures
- **Action**: Request operator guidance
- **Use**: Novel/ambiguous situations

---

## Continual Learning

SALUS adapts over time:

1. **Online collection**: Log all (signals, outcomes) during deployment
2. **Periodic retraining**: Every N hours or M failures
3. **Active learning**: Query operator on uncertain cases
4. **Distribution shift detection**: Monitor signal statistics

---

## Implementation Modules

### Core Modules (To Build)
1. ✅ `salus/core/vla/wrapper.py` - VLA ensemble wrapper
2. ✅ `salus/core/vla/signal_extractor.py` - 12D feature extraction
3. 🔨 `salus/core/predictor.py` - **SALUS neural predictor**
4. 🔨 `salus/core/adaptation.py` - **Failure adaptation logic**
5. 🔨 `salus/training/trainer.py` - **Training loop**
6. 🔨 `salus/evaluation/metrics.py` - **Evaluation metrics**

### Data Modules (Complete)
7. ✅ `salus/data/recorder.py` - Zarr data recording
8. ✅ `salus/data/dataset.py` - PyTorch dataset loader

### Simulation (In Progress)
9. ✅ `salus/simulation/isaaclab_env.py` - Dummy environment
10. 🔨 `salus/simulation/franka_pick_place_env.py` - Real Franka environment

---

## Next Steps

### Immediate (Today)
1. ✅ **VLA Integration** - Complete
2. ✅ **Data Collection** - Working
3. 🔨 **Build SALUS Predictor** - Neural network module
4. 🔨 **Build Adaptation Module** - Intervention logic

### Short-term (This Week)
5. **Collect Training Data** - 500 episodes with failures
6. **Train SALUS** - First model on collected data
7. **Evaluate Performance** - Test on held-out episodes
8. **Deploy with Adaptation** - Run closed-loop tests

### Medium-term
9. **Optimize Interventions** - Tune adaptation strategies
10. **Continual Learning** - Online adaptation
11. **Real Robot Deployment** - Transfer to physical Franka

---

## Key Insights from Paper

### Why SALUS Works
1. **Model uncertainty** from ensemble captures model confidence
2. **Trajectory features** capture temporal dynamics
3. **Multi-horizon** allows graduated responses
4. **Closed-loop adaptation** prevents failures proactively

### Novel Contributions
- First VLA-specific failure predictor
- Multi-horizon prediction for adaptive interventions
- Demonstrated on real robot tasks
- Reduces failures by 60-80% with minimal performance cost

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Failure Detection | 90%+ recall | Catch most failures |
| False Positives | <10% | Don't over-intervene |
| Lead Time | 200-500ms | Enough time to adapt |
| Success Rate | +40% improvement | Over baseline VLA |
| Throughput | <5% slowdown | Minimal overhead |

---

**Status**: 🔨 Ready to build core SALUS modules!
