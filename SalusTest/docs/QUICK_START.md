# SALUS Quick Start - Your 4x RTX 2080 Ti Machine

## Run These Commands RIGHT NOW (Copy-Paste):

```bash
# Navigate to aegis directory
cd ~/Desktop/aegis

# Run automated setup
./setup_local.sh

# This will:
# ✅ Check your 4 GPUs
# ✅ Create virtual environment
# ✅ Install PyTorch with CUDA 11.8
# ✅ Install all dependencies
# ✅ Create project structure
# ✅ Test GPU access
```

**Time required**: 15-20 minutes

---

## After Setup Completes:

### Download VLA Model (~30 minutes):

```bash
# Install Hugging Face CLI
source venv_salus/bin/activate
pip install huggingface-hub

# Clone TinyVLA
cd ~/
git clone https://github.com/OpenDriveLab/TinyVLA.git
cd TinyVLA
pip install -e .

# Download pretrained weights (~2.5GB)
mkdir -p ~/models/tinyvla
huggingface-cli download TinyVLA/tinyvla-1b --local-dir ~/models/tinyvla/tinyvla-1b

# Verify
ls -lh ~/models/tinyvla/tinyvla-1b/
# Should see pytorch_model.bin (~2.2GB)
```

---

## GPU Allocation on Your Machine:

```
┌─────────────────────────────────────────┐
│  YOUR 4x RTX 2080 Ti (11GB each)       │
└─────────────────────────────────────────┘

GPU 0 (11GB) → VLA Ensemble
  └─ 5× TinyVLA-1B models = 11GB

GPU 1 (4GB)  → Signal Extractor + Predictor
  └─ Lightweight inference + training

GPU 2 (3GB)  → Safety Manifold (Week 7+)
  └─ Reserved for later phases

GPU 3 (8GB)  → MPC Synthesis (Week 9+)
  └─ Reserved for later phases
```

**For data collection**: All 4 GPUs run parallel environments (16 total environments)

---

## Your Timeline:

| Week | Task | Run On | Time |
|------|------|--------|------|
| **1** | Environment setup + VLA test | Local | 4 hours |
| **2** | Simulation environment | Local | 8 hours |
| **2-3** | Data collection (500 episodes) | Local | Overnight (6-12h) |
| **3** | Train failure predictor | Local or HPC | 2-4 hours |
| **4-6** | Train manifold + dynamics | **HPC Cluster** | Heavy compute |
| **7-10** | Integration + testing | Local | Development |

---

## Next Steps (After VLA Download):

### 1. Test VLA on GPU 0:

```bash
cd ~/Desktop/aegis
source venv_salus/bin/activate

# Copy VLA wrapper from LOCAL_MACHINE_SETUP.md
# File: salus/core/vla/wrapper.py

# Test it
python -m salus.core.vla.wrapper
```

Expected output:
```
Loading VLA ensemble (5 models on cuda:0)...
  Model 1/5 loaded
  Model 2/5 loaded
  ...
✅ Ensemble ready on cuda:0
✅ Actions shape: torch.Size([2, 5, 7])
✅ VLA wrapper working!
```

### 2. Create Simulation Environment:

```bash
# Copy simulation code from LOCAL_MACHINE_SETUP.md
# File: salus/simulation/simple_pick_place.py

# Test it
python salus/simulation/simple_pick_place.py
```

### 3. Start Data Collection (Overnight Run):

```bash
cd ~/Desktop/aegis
source venv_salus/bin/activate

# Copy data collection script from LOCAL_MACHINE_SETUP.md
# File: scripts/collect_data_local.py

# Run in background
nohup python scripts/collect_data_local.py > logs/data_collection.log 2>&1 &

# Check progress
tail -f logs/data_collection.log

# Monitor GPUs
watch -n 1 nvidia-smi
```

**Expected runtime**: 6-12 hours (overnight)
**Output**: 500 episodes in `data/raw_episodes/`

### 4. Train Predictor (Next Day):

```bash
# Copy training script from LOCAL_MACHINE_SETUP.md
# File: scripts/train_predictor_local.py

# Train on GPU 1
python scripts/train_predictor_local.py
```

**Expected runtime**: 2-4 hours
**Output**: `models/predictor/simple_predictor_v1.pth`

---

## What You're NOT Waiting For:

- ❌ Physical robot (use simulation until Week 12+)
- ❌ ROS2 setup (not needed until robot deployment)
- ❌ Isaac Lab (can use MuJoCo instead for faster iteration)
- ❌ More GPUs (4x 2080 Ti is perfect for development)

---

## What to Do on HPC Cluster (Later):

**Send to HPC when ready** (Week 4+):
- Train safety manifold (needs 8+ hours, benefits from more GPUs)
- Train dynamics model (heavy compute)
- Hyperparameter sweeps
- Large-scale experiments

**Keep on local machine**:
- Development and debugging
- Data collection (you have 4 GPUs!)
- Predictor training (fast enough)
- Integration testing

---

## Monitoring Your Progress:

```bash
# Check GPU usage
nvidia-smi

# Watch GPUs continuously
watch -n 1 nvidia-smi

# Check data collection progress
ls data/raw_episodes/ | wc -l   # Count episodes

# Check disk space (data collection generates ~50GB)
df -h ~/Desktop/aegis/data

# View logs
tail -f logs/data_collection.log
tail -f logs/training.log
```

---

## Troubleshooting:

### GPUs not detected:
```bash
nvidia-smi
# Update driver if needed:
sudo apt install nvidia-driver-535
sudo reboot
```

### PyTorch CUDA error:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should be True
# If False, reinstall PyTorch with correct CUDA version
```

### Out of memory on GPU 0:
```bash
# Reduce ensemble size from 5 to 3 models
# Edit salus/core/vla/wrapper.py:
# ensemble_size=3  # instead of 5
```

### Data collection too slow:
```bash
# Reduce parallel environments
# Edit scripts/collect_data_local.py:
# num_envs_per_gpu=2  # instead of 4
```

---

## File Locations:

```
~/Desktop/aegis/
├── setup_local.sh              ← Run this first
├── LOCAL_MACHINE_SETUP.md      ← Detailed instructions
├── GETTING_STARTED.md          ← Complete guide
├── QUICK_START.md              ← This file
│
├── salus/                   ← Core SALUS code
│   ├── core/vla/wrapper.py    ← VLA ensemble + signals
│   ├── simulation/            ← Simulation environments
│   ├── data/recorder.py       ← Data collection
│   └── training/              ← Training scripts
│
├── scripts/                    ← Runnable scripts
│   ├── collect_data_local.py  ← Run overnight
│   ├── train_predictor_local.py
│   └── test_*.py              ← Testing scripts
│
├── data/                       ← Generated data
│   ├── raw_episodes/          ← Collected episodes
│   └── processed/             ← After signal extraction
│
├── models/                     ← Saved models
│   └── predictor/             ← Trained predictors
│
└── logs/                       ← Execution logs
    ├── data_collection.log
    └── training.log
```

---

## Summary: What You Do TODAY:

1. **Run setup script** (20 min)
   ```bash
   cd ~/Desktop/aegis
   ./setup_local.sh
   ```

2. **Download TinyVLA** (30 min)
   ```bash
   cd ~/
   git clone https://github.com/OpenDriveLab/TinyVLA.git
   cd TinyVLA && pip install -e .
   huggingface-cli download TinyVLA/tinyvla-1b --local-dir ~/models/tinyvla/tinyvla-1b
   ```

3. **Implement VLA wrapper** (2 hours)
   - Copy code from `LOCAL_MACHINE_SETUP.md` → PHASE 4
   - Create `salus/core/vla/wrapper.py`
   - Test with `python -m salus.core.vla.wrapper`

4. **Start data collection overnight** (5 min setup + overnight run)
   - Copy code from `LOCAL_MACHINE_SETUP.md` → PHASE 5
   - Run `nohup python scripts/collect_data_local.py > logs/data_collection.log 2>&1 &`

**Tomorrow morning**: Check logs, verify 500 episodes collected, start predictor training!

---

## Questions?

See detailed documentation:
- `LOCAL_MACHINE_SETUP.md` - Complete setup guide for your machine
- `GETTING_STARTED.md` - Overall SALUS development guide
- `salus_implementation_guide.md` - Full implementation reference
- `salus_software_architecture.md` - System architecture

**You have everything you need. Start NOW!**
