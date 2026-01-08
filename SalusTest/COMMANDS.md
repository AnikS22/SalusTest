# SALUS Dashboard Commands

## 🎯 Quick Start

### See the Enhanced Dashboard (RECOMMENDED)

```bash
python salus_dashboard.py
```

**Shows:**
- 📷 **Camera View** - What the VLA sees (image stats + ASCII art)
- 🤖 **VLA Processing** - Internal processing pipeline
- 📊 **Signal Extraction** - All 12D signals being extracted
- 🧠 **Risk Computation** - Neural network processing
- 🚨 **Alert Decision** - State machine logic
- 📈 **System Stats** - Performance metrics

**Updates:** 10 FPS (real-time)

**Press Ctrl+C to exit**

---

### See the Simple Terminal GUI

```bash
python salus_live_demo.py
```

**Shows:**
- Left panel: 12D signals with bars
- Right panel: Multi-horizon predictions
- Footer: Episode statistics

---

### See with Isaac Sim (3D Visualization)

```bash
# 1. Source Isaac Sim environment
source ~/.local/share/ov/pkg/isaac_sim-*/setup_python_env.sh

# 2. Run
python salus_isaac_sim.py
```

**Shows:**
- Isaac Sim window: Robot moving in 3D
- Terminal: All SALUS metrics

---

## 📊 What Each Command Shows

### Enhanced Dashboard (salus_dashboard.py)

```
╔══════════════════════════════════════════════╗
║  SALUS Enhanced Dashboard                    ║
╚══════════════════════════════════════════════╝

┌─ 📷 Camera View ─────┬─ 🤖 VLA Processing ──┐
│ (1, 3, 224, 224)     │ Vision: ✓ Active     │
│ Mean: 0.523          │ Language: ✓ Active   │
│ Std:  0.891          │ Transformer: ✓ 12    │
│                      │ Action: [+0.12 ...]  │
│ ┌────────────┐       │ Confidence: 78%      │
│ │@@@@@@@@@@@@│       │                      │
│ └────────────┘       │                      │
└──────────────────────┴──────────────────────┘

┌─ 📊 Signals ─────┬─ 🧠 Risk ───┬─ 🚨 Alert ─┐
│ z₁ 1.234         │ Conv1D: ✓   │ EMA: 0.678 │
│ z₂ 0.567         │ BiGRU: ✓    │ Persist: ✓ │
│ z₈ 1.890         │             │ 🚨 CRITICAL│
│ ... (12 total)   │ 500ms: 0.68 │ → INTERVEN │
└──────────────────┴─────────────┴────────────┘

┌─ Performance ─────────────────────────────────┐
│ Step: 482  Episode: 5  FPS: 9.8              │
└───────────────────────────────────────────────┘
```

**Best for:** Understanding the full pipeline

---

### Simple GUI (salus_live_demo.py)

```
┌─ 12D Signal Vector ──────┬─ Multi-Horizon ──────┐
│ z₁  Action Volatility    │ 300ms   ████░░░░░░░  │
│     ████████░░░░░░░░░░░  │ 500ms   ████████░░░  │
│ z₂  Action Magnitude     │ 1000ms  ████████████ │
│     ████████████░░░░░░░  │                       │
│ ... (all 12 signals)     │ EMA     ██████░░░░░  │
│                          │                       │
│ ✓ NORMAL                 │                       │
└──────────────────────────┴───────────────────────┘
```

**Best for:** Quick monitoring, less detail

---

### Isaac Sim (salus_isaac_sim.py)

**Two windows:**

**Window 1 (Isaac Sim):**
```
[3D Robot View]
  ┌─────────────────┐
  │                 │
  │   [Robot Arm]   │
  │      │          │
  │     [Cube]      │
  │                 │
  └─────────────────┘
```

**Window 2 (Terminal):**
```
Same as Simple GUI
```

**Best for:** Seeing robot movement + metrics

---

## 🔍 Debug Commands

### See What SALUS is Thinking (Real-Time Log)

```bash
# Run dashboard with logging
python salus_dashboard.py 2>&1 | tee salus_live.log

# In another terminal, watch:
tail -f salus_live.log
```

### See Raw Signal Values

```bash
# Add --verbose flag (if implemented)
python salus_dashboard.py --verbose
```

### Monitor System Resources

```bash
# CPU/Memory usage
watch -n 1 'ps aux | grep salus_dashboard'

# GPU usage (if using CUDA)
watch -n 1 nvidia-smi
```

---

## 🎮 Keyboard Controls

While running any dashboard:

- **Ctrl+C** - Exit gracefully
- **Ctrl+Z** - Suspend (resume with `fg`)
- **q** - Quit (if implemented)

---

## 📈 Performance Tips

### Reduce Refresh Rate (Lower CPU)

```python
# Edit salus_dashboard.py line ~1050
with Live(self.layout, console=self.console, refresh_per_second=5):  # Was 10
```

### Run Headless (No GUI, Logging Only)

```bash
python salus_dashboard.py --headless --log salus_output.log
```

### Reduce Window Size

```python
# Edit salus_dashboard.py
self.window_size = 10  # Was 20 (333ms instead of 667ms)
```

---

## 🚀 Integration Commands

### Use Your Own VLA

```python
# Edit salus_dashboard.py line ~1080
# Replace:
vla = MockVLA()

# With:
from your_vla import YourVLA
vla = YourVLA.load_pretrained("model-name")
```

### Custom Intervention

```python
# Edit salus_dashboard.py line ~1120
if salus.should_intervene(alert_result):
    # Custom logic here
    if signals[0] > 1.5:  # High volatility
        action = action * 0.0  # Freeze
    else:
        action = action * 0.5  # Slow mode
```

---

## 📊 Data Export Commands

### Export Episode Data

```python
# Run with export flag
python salus_dashboard.py --export episodes.json
```

### Export Signal History

```python
# In dashboard, signals are in salus.signal_history
import pickle
with open('signal_history.pkl', 'wb') as f:
    pickle.dump(list(salus.signal_history), f)
```

### Export for Analysis

```bash
# Run and redirect output
python salus_dashboard.py 2>&1 | grep "CRITICAL" > alerts.txt
```

---

## 🎓 Advanced Commands

### Profile Performance

```bash
python -m cProfile -o salus.prof salus_dashboard.py
python -m pstats salus.prof
```

### Memory Profiling

```bash
python -m memory_profiler salus_dashboard.py
```

### Distributed Monitoring (Multiple Robots)

```bash
# Robot 1
python salus_dashboard.py --robot-id robot1 --port 5001

# Robot 2
python salus_dashboard.py --robot-id robot2 --port 5002

# Central monitor
python salus_monitor_aggregate.py --robots robot1:5001,robot2:5002
```

---

## 🐛 Troubleshooting Commands

### Check if Demo Works

```bash
python -c "from salus_dashboard import *; print('Import OK')"
```

### Verify Model Loaded

```bash
python -c "import torch; c=torch.load('salus_fixed_pipeline.pt', weights_only=False); print('Model OK')"
```

### Test GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Check Dependencies

```bash
pip list | grep -E "rich|torch|numpy|scipy"
```

---

## 📚 Quick Reference

| Command | Purpose | Updates/s | Best For |
|---------|---------|-----------|----------|
| `python salus_dashboard.py` | Full dashboard | 10 | Understanding pipeline |
| `python salus_live_demo.py` | Simple GUI | 10 | Quick monitoring |
| `python salus_isaac_sim.py` | 3D + GUI | 10 | Seeing robot move |
| `./setup_demo.sh` | Setup | - | First time setup |

---

## 🎯 Recommended Workflow

### 1. First Time

```bash
./setup_demo.sh
python salus_dashboard.py
```

Watch for 1-2 minutes to understand the flow.

### 2. Integration

```bash
# Edit salus_dashboard.py to use your VLA
# Then run
python salus_dashboard.py
```

Validate signals are extracted correctly.

### 3. Production

```bash
# Use simple GUI for lower overhead
python salus_live_demo.py 2>&1 | tee production.log
```

Monitor continuously.

---

## 💡 Pro Tips

### Split Screen

```bash
# Terminal 1: Dashboard
python salus_dashboard.py

# Terminal 2: Log watching
tail -f salus_live.log

# Terminal 3: System monitor
htop
```

### Background Running

```bash
# Run in background
nohup python salus_dashboard.py > dashboard.log 2>&1 &

# Check status
ps aux | grep salus_dashboard

# Kill if needed
pkill -f salus_dashboard
```

### Quick Test

```bash
# Run for 30 seconds and exit
timeout 30 python salus_dashboard.py
```

---

**Last Updated:** 2026-01-08

**Quick Start:** `python salus_dashboard.py`

**Help:** See `VISION_PIPELINE_GUIDE.md` for details on what's shown

**Integration:** See `INTEGRATION_GUIDE.md` for using your VLA
