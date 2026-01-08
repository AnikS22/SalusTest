# SALUS Vision Pipeline: How Images Flow Through the System

## 📸 Complete Data Flow

```
Camera → VLA → SALUS Signal Extraction → Risk Prediction → Alert → Intervention
```

---

## Step 1: Camera Input (What SALUS "Sees")

### Raw Image Capture

```python
# Camera captures RGB image
camera_image = camera.capture()  # Shape: (224, 224, 3)

# Convert to tensor for VLA
observation = torch.tensor(camera_image).permute(2, 0, 1)  # (3, 224, 224)
observation = observation.unsqueeze(0)  # (1, 3, 224, 224) - batch dimension
```

**In Dashboard:** See the "📷 Camera View" panel showing:
- Image shape (1, 3, 224, 224)
- Mean intensity (brightness)
- Standard deviation (contrast/clutter)
- ASCII art visualization of the scene

### What the VLA Sees

The VLA receives the same image and processes it through:

```
Image (224x224x3)
    ↓
Vision Encoder (CNN or ViT)
    ↓
Visual Features (768D or 2048D)
    ↓
[Combined with language embedding]
    ↓
Transformer Layers (12 layers)
    ↓
Action Head
    ↓
7D Action Vector [x, y, z, roll, pitch, yaw, gripper]
```

**In Dashboard:** See the "🤖 VLA Processing" panel showing:
- Vision encoder status (✓ active)
- Language processing status
- Transformer layers (12 layers active)
- Output action vector
- VLA confidence level

---

## Step 2: SALUS Signal Extraction

### SALUS Doesn't Use Raw Images!

**Key Insight:** SALUS doesn't need to see the image directly. Instead, it watches the VLA's behavior:

```python
# SALUS observes VLA's actions over time
action_history = [a_t-4, a_t-3, a_t-2, a_t-1, a_t]

# Extract 12D signals from VLA behavior
signals = salus.extract_signals(
    action=action_t,              # Current action
    action_probs=probs_t,         # Action probabilities
    hidden_states=h_t,            # VLA internal states (optional)
    action_history=action_history # Past actions
)
```

### How Image Information Flows into Signals

Even though SALUS doesn't directly process images, image information is encoded in the signals:

**z₁ (Action Volatility):**
- High volatility → VLA uncertain about what it sees
- Low volatility → VLA confident in visual perception

**Example:**
```python
# Cluttered scene (image has high std) → VLA uncertain → high volatility
z1 = np.std(action_history[-5:])  # High value ~1.5

# Clear scene (image has low std) → VLA confident → low volatility
z1 = np.std(action_history[-5:])  # Low value ~0.1
```

**z₅-z₇ (VLA Hidden States):**
- These ARE derived from the image!
- Vision encoder processes image → creates hidden states
- SALUS captures statistics of these hidden states

```python
# VLA processes image
image → vision_encoder → hidden_states  # (768D features)

# SALUS extracts stats
z5 = ||hidden_states||₂        # Norm: activation magnitude
z6 = std(hidden_states)        # Std: feature diversity
z7 = skew(hidden_states)       # Skew: distribution shape
```

**z₈-z₉ (Action Uncertainty):**
- VLA's confusion about the image → high entropy
- VLA's clarity about the image → low entropy, high max prob

```python
# Unclear image (occluded, blurry)
# → VLA unsure → uniform probabilities → high entropy
z8 = -sum(p * log(p))  # High value ~2.0

# Clear image
# → VLA confident → peaked probabilities → low entropy
z8 = -sum(p * log(p))  # Low value ~0.3
```

**In Dashboard:** See the "📊 Signal Extraction Process" panel showing:
- All 12 signals with their values
- Source of each signal (action history, VLA internals, action probs)
- Extraction method (full 12D or minimal 6D)

---

## Step 3: Risk Computation (The Neural Network)

### Temporal Window Processing

```python
# Collect 20 timesteps of signals (667ms history)
signal_window = [
    [z1_t-19, z2_t-19, ..., z12_t-19],
    [z1_t-18, z2_t-18, ..., z12_t-18],
    ...
    [z1_t, z2_t, ..., z12_t]
]  # Shape: (20, 12)

# Feed to SALUS neural network
risk_scores = salus_model(signal_window)
```

### Internal Processing

```
Signal Window (20, 12)
    ↓
Conv1D Layer 1 (kernel=5) - Detect short patterns
    ↓
Conv1D Layer 2 (kernel=3) - Refine patterns
    ↓
Conv1D Layer 3 (kernel=3) - Extract features
    ↓
BiGRU Layer 1 (h=128) - Learn temporal dependencies
    ↓
BiGRU Layer 2 (h=128) - Long-range patterns
    ↓
Multi-Horizon Heads:
    ├─ 300ms Head → risk_300
    ├─ 500ms Head → risk_500
    └─ 1000ms Head → risk_1000
```

**What the network learns:**

The network learns patterns like:

**Normal behavior pattern:**
```
z1 (volatility):   [0.1, 0.1, 0.1, 0.1, 0.1, ...]  ← Stable
z8 (entropy):      [0.3, 0.3, 0.3, 0.3, 0.3, ...]  ← Confident
→ Risk: 0.08 (low)
```

**Failure precursor pattern:**
```
z1 (volatility):   [0.1, 0.2, 0.4, 0.8, 1.5, ...]  ← Increasing!
z8 (entropy):      [0.3, 0.5, 0.9, 1.5, 2.1, ...]  ← Increasing!
→ Risk: 0.78 (high)
```

**In Dashboard:** See the "🧠 Risk Computation" panel showing:
- Input window status (20 timesteps filled)
- Conv1D processing (extract local patterns)
- BiGRU processing (temporal dependencies)
- Multi-horizon predictions (300ms, 500ms, 1000ms)
- Risk score visualizations

---

## Step 4: Alert Decision (State Machine)

### EMA Smoothing

```python
# Smooth risk score over time (reduce jitter)
risk_ema_t = 0.3 * risk_500_t + 0.7 * risk_ema_{t-1}
```

### Persistence Check

```python
# Require 4 consecutive high-risk ticks
if all([risk_ema > 0.40 for last 4 ticks]):
    persistence_met = True
```

### Hysteresis

```python
# Different thresholds for entering vs exiting alert
if risk_ema > 0.40 and persistence_met:
    state = CRITICAL  # Enter alert
elif risk_ema < 0.35:
    state = NORMAL    # Exit alert (lower threshold!)
```

### Cooldown

```python
# After CRITICAL alert, wait 2 seconds before re-alerting
if alert_fired:
    cooldown_counter = 60  # 60 ticks @ 30Hz = 2 seconds
```

**In Dashboard:** See the "🚨 Alert State Machine Decision" panel showing:
- EMA smoothing status and value
- Persistence check (x/4 ticks)
- Hysteresis check (above τ_on or τ_off)
- Cooldown status (active or ready)
- Final alert state (NORMAL/WARNING/CRITICAL)
- Intervention action (if triggered)

---

## 📊 Enhanced Dashboard Layout

```
╔══════════════════════════════════════════════════════════════════╗
║  SALUS Enhanced Dashboard - See Everything!                      ║
╚══════════════════════════════════════════════════════════════════╝

┌─ 📷 Camera View ──────────┬─ 🤖 VLA Processing ─────────────────┐
│ Input: (1, 3, 224, 224)   │ Vision Encoder:  ✓ Active           │
│ Mean: 0.523               │ Language Model:  ✓ "pick red mug"   │
│ Std:  0.891               │ Transformer:     ✓ 12 layers        │
│                           │ Action Head:     ✓ 7D output         │
│ ┌────────────────┐        │                                      │
│ │@@@@@@@@@@@@@@@@│        │ Action: [+0.12 -0.05 +0.23 ...]     │
│ │@@@@@@@@@@@@@@@@│        │ Confidence: ████████████░░ 78%      │
│ │@@@@@@@@@@@@@@@@│        │                                      │
│ │@@@@@@@@@@@@@@@@│        │                                      │
│ │@@@@@@@@@@@@@@@@│        │                                      │
│ └────────────────┘        │                                      │
│ Scene: Cluttered table    │                                      │
└───────────────────────────┴──────────────────────────────────────┘

┌─ 📊 Signal Extraction ────┬─ 🧠 Risk Computation ──┬─ 🚨 Alert ─┐
│ z₁ Volatility    1.234    │ Input Window: ✓ 20     │ 1. EMA:    │
│    (Action history)        │ Conv1D: ✓ Patterns     │    ✓ 0.678 │
│                           │ BiGRU:  ✓ Temporal     │ 2. Persist:│
│ z₂ Magnitude     0.567    │                        │    ✓ 4/4   │
│    (Current action)        │ 300ms: ███░ 0.345      │ 3. Hyster: │
│                           │ 500ms: ██████ 0.678    │    ✓ Above │
│ z₈ Entropy       1.890    │ 1000ms: ████████ 0.812 │ 4. Cool:   │
│    (Action probs)          │                        │    ✓ Ready │
│                           │                        │            │
│ ... (12 signals total)     │                        │ 🚨 CRITICAL│
│                           │                        │            │
│ Extraction: Full 12D      │                        │ → INTERVEN │
└───────────────────────────┴────────────────────────┴────────────┘

┌─ System Performance ──────────────────────────────────────────────┐
│ Step: 482         Episode: 5          FPS: 9.8                   │
│ Alerts: 3         Interventions: 3    Success: 80.0%             │
│ Latency: 98.2ms   Failures: 1         Memory: 1200MB             │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🎮 Run the Enhanced Dashboard

### Command:

```bash
python salus_dashboard.py
```

### What You'll See:

**Top Row (Vision):**
- **Left:** Camera view showing image stats and ASCII visualization
- **Right:** VLA processing pipeline with action output

**Middle Row (Processing):**
- **Left:** 12D signal extraction with sources
- **Center:** Risk computation through Conv1D+GRU
- **Right:** Alert state machine decision logic

**Bottom Row:**
- System performance statistics

**All panels update at 10 FPS!**

---

## 🔍 What to Watch For

### Normal Operation

**Camera:** Clear scene, low std (~0.3)
**VLA:** High confidence (~95%), stable actions
**Signals:**
- z₁ (volatility): ~0.1 (low)
- z₈ (entropy): ~0.3 (low)
**Risk:** ~0.08 (green)
**Alert:** NORMAL ✓

### Failure Precursor

**Camera:** Cluttered scene, high std (~1.5)
**VLA:** Dropping confidence (~60%), erratic actions
**Signals:**
- z₁ (volatility): ~1.8 (HIGH!)
- z₈ (entropy): ~2.1 (HIGH!)
**Risk:** ~0.78 (red)
**Alert:** CRITICAL 🚨

**→ Intervention triggered: action *= 0.5 (slow mode)**

---

## 📸 Image Information Flow Summary

```
Camera Image
    ↓
VLA Vision Encoder
    ↓
Hidden States (contains image info)
    ↓
VLA Action Prediction
    ↓
SALUS Observes:
    ├─ Action values (z₂)
    ├─ Action changes (z₁, z₃)
    ├─ Hidden state stats (z₅-z₇) ← Image features!
    └─ Action confidence (z₈-z₉) ← Image clarity!
    ↓
SALUS Neural Network
    ↓
Risk Prediction
    ↓
Alert State Machine
    ↓
Intervention (if needed)
```

**Key insight:** SALUS doesn't need raw images because the VLA has already processed them into:
1. Hidden state features
2. Action predictions
3. Confidence levels

SALUS extracts these signals and looks for temporal patterns indicating impending failure!

---

## 🚀 Try It Yourself

### 1. Run Enhanced Dashboard

```bash
python salus_dashboard.py
```

### 2. Watch the Flow

- **Top-left:** See what camera captures
- **Top-right:** See VLA process it
- **Middle-left:** See signals extracted
- **Middle-center:** See risk computed
- **Middle-right:** See alert decision
- **Bottom:** See overall stats

### 3. Observe Patterns

Watch how:
- Cluttered scenes → VLA confusion → high signals → high risk
- Clear scenes → VLA confidence → low signals → low risk
- Risk rises gradually (precursor pattern)
- Alert triggers at right moment (500-800ms before failure)

---

## 💡 Advanced: See Raw Data

### Enable Debug Logging

```python
# Add to salus_dashboard.py
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    filename='salus_vision_debug.log'
)

logger = logging.getLogger('SALUS')

# In control loop
logger.debug(f"Camera: mean={camera_mean:.3f}, std={camera_std:.3f}")
logger.debug(f"VLA hidden: norm={z5:.3f}, std={z6:.3f}, skew={z7:.3f}")
logger.debug(f"Signals: z1={z1:.3f}, z8={z8:.3f}")
logger.debug(f"Risk: {risk_500:.3f}")
```

### View Log

```bash
tail -f salus_vision_debug.log
```

You'll see:
```
14:23:45.123 - Camera: mean=0.523, std=0.891
14:23:45.124 - VLA hidden: norm=1.234, std=0.456, skew=-0.123
14:23:45.125 - Signals: z1=1.234, z8=1.890
14:23:45.126 - Risk: 0.678
```

---

**Last Updated:** 2026-01-08

**Status:** ✅ Enhanced dashboard ready

**Command:** `python salus_dashboard.py`

**Shows:** Everything SALUS sees and thinks in real-time!
