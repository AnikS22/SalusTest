# VLA Models Download Status

## Summary

**Total VLA Models Downloaded: 2 unique models**

---

## Downloaded Models

### 1. SmolVLA-450M ✅

**Location**: `~/models/smolvla/smolvla_base/`

**Model Details:**
- **Parameters**: 450 Million (450M)
- **Model Size**: 865 MB (model.safetensors)
- **Architecture**: SmolVLA (based on Qwen2-VL)
- **Status**: ✅ **Fully Integrated**
- **Used by**: `salus/core/vla/wrapper.py` (SmolVLAEnsemble)

**Files:**
```
model.safetensors             865 MB   - Model weights
config.json                   2.3 KB   - Model configuration
policy_preprocessor.json      1.9 KB   - Preprocessing config
policy_postprocessor.json     660 B    - Postprocessing config
collage_small.gif             7.7 MB   - Demo GIF
README.md                     1.8 KB   - Documentation
```

**Integration Status:**
- ✅ Code: `salus/core/vla/wrapper.py` (SmolVLAEnsemble class)
- ✅ Tested: Integration complete (see SMOLVLA_INTEGRATION_COMPLETE.md)
- ✅ Ready for: Data collection with real VLA

---

### 2. TinyVLA-400M (Llava-Pythia-400M) ✅

**Location**: `~/models/tinyvla/Llava-Pythia-400M/`

**Model Details:**
- **Parameters**: 400 Million (400M)
- **Model Size**: 715 MB (model.safetensors)
- **Architecture**: Llava-Pythia (Vision-Language-Action)
- **Base LLM**: Pythia-400M
- **Status**: ✅ **Downloaded, Not Yet Integrated**
- **Used by**: `salus/core/vla/tinyvla_wrapper.py` (TinyVLAEnsemble - placeholder)

**Files:**
```
model.safetensors             715 MB   - Model weights
config.json                   5.4 KB   - Model configuration
tokenizer.json                2.1 MB   - Tokenizer (if present)
preprocessor_config.json      316 B    - Image preprocessing config
generation_config.json        111 B    - Generation settings
special_tokens_map.json       471 B    - Special tokens
tokenizer_config.json         4.7 KB   - Tokenizer configuration
```

**Integration Status:**
- ⚠️ Code: `salus/core/vla/tinyvla_wrapper.py` exists but needs TinyVLA package installation
- ❌ Not tested: Requires TinyVLA package dependencies
- ⏳ Ready for: Integration after installing TinyVLA package

---

### 3. TinyVLA-droid-metaworld ⚠️

**Location**: `~/models/tinyvla/TinyVLA-droid-metaworld/`

**Status**: ⚠️ **Downloaded but incomplete/unknown**

**Note**: This appears to be a variant or alternative download. Need to verify if this is a complete model or partial download.

---

## Model Comparison

| Model | Size | Parameters | Status | Integration | Usage |
|-------|------|------------|--------|-------------|-------|
| **SmolVLA-450M** | 865 MB | 450M | ✅ Ready | ✅ Complete | Primary (active) |
| **TinyVLA-400M** | 715 MB | 400M | ✅ Downloaded | ⏳ Pending | Alternative |
| **TinyVLA-droid** | Unknown | Unknown | ⚠️ Unknown | ❌ None | Unknown |

---

## Storage Summary

**Total Model Storage:**
- SmolVLA-450M: ~873 MB (including all files)
- TinyVLA-400M: ~718 MB (including all files)
- TinyVLA-droid: Unknown size
- **Total**: ~1.6 GB+ (minimum)

---

## Current Usage in SALUS

### Active Model: SmolVLA-450M

**Code Location**: `salus/core/vla/wrapper.py`

```python
class SmolVLAEnsemble(nn.Module):
    def __init__(
        self,
        model_path: str = "~/models/smolvla/smolvla_base",
        ensemble_size: int = 5,
        device: str = "cuda:0"
    ):
        # Loads 5× SmolVLA-450M models for ensemble
```

**Ensemble Configuration:**
- 5 models × 865 MB ≈ 4.3 GB (compressed)
- Fits on: GPU 0 (11GB RTX 2080 Ti) ✅
- Memory usage: ~4.5 GB VRAM (approximate)

---

## Recommended Actions

### For Production Use:

✅ **Use SmolVLA-450M** (currently integrated and tested)

### For Development/Testing:

⏳ **Integrate TinyVLA-400M** (smaller, faster inference)
- Requires: TinyVLA package installation
- Benefits: Faster inference, smaller memory footprint
- Trade-off: Slightly less capable than SmolVLA-450M

---

## Next Steps

1. ✅ **SmolVLA-450M**: Already integrated, ready for data collection
2. ⏳ **TinyVLA-400M**: Install TinyVLA package to enable integration
3. ❓ **TinyVLA-droid**: Verify if this is a complete model or can be removed

---

**Last Updated**: January 3, 2026
**Status**: 2 VLA models downloaded (1 fully integrated, 1 pending integration)




