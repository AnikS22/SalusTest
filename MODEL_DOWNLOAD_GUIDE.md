# Model Download Guide for SALUS

## Current Situation

The TinyVLA-1B model referenced in the documentation (`TinyVLA/tinyvla-1b` on HuggingFace) is not publicly available or requires special access. The repository `OpenDriveLab/TinyVLA` also doesn't exist.

## Recommended Solution: Use OpenVLA-7B

OpenVLA is a well-established, publicly available VLA model that will work with SALUS.

### Download OpenVLA:

```bash
# Activate environment
cd "/home/mpcr/Desktop/Salus Test/SalusTest"
source venv_salus/bin/activate

# Clone OpenVLA repository
cd ~
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# Download model weights (Warning: ~14GB download)
mkdir -p ~/models/openvla
huggingface-cli download openvla/openvla-7b --local-dir ~/models/openvla/openvla-7b

# Verify download
ls -lh ~/models/openvla/openvla-7b/
# Should see model files
```

### Update Wrapper to Use OpenVLA:

The wrapper we created can be adapted to use OpenVLA instead. You'll need to:

1. Update the model path in the wrapper
2. Adjust for OpenVLA's API (which may differ from TinyVLA)
3. Note: OpenVLA-7B is larger (requires more VRAM), so you may need to reduce ensemble size from 5 to 2-3 models

## Alternative: Use Placeholder/Mock for Development

For initial development and testing of the SALUS system architecture, you can use a mock VLA implementation that matches the expected interface:

```python
# Mock implementation for development
class MockVLA:
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, rgb, proprio, language):
        # Return dummy actions and internals
        return {
            'action': torch.randn(rgb.shape[0], 7),
            'internals': {
                'attention': torch.randn(rgb.shape[0], 8, 224, 224),
                'hidden': torch.randn(rgb.shape[0], 512)
            }
        }
```

This allows you to:
- Test the SALUS wrapper interface
- Develop data collection scripts
- Test simulation environments
- Work on predictor training pipeline

Then swap in the real model once available.

## Finding the Real TinyVLA Model

If you need TinyVLA specifically:

1. **Check with the SALUS paper authors** - They may have access or can point you to the right source
2. **Check HuggingFace with authentication** - Try logging in:
   ```bash
   huggingface-cli login
   # Then try downloading again
   ```
3. **Check alternative repositories** - The model might be hosted elsewhere
4. **Train your own TinyVLA** - Use the `vovw/tinyvla` repository to train a model from scratch (requires dataset and compute)

## Recommendation

**For MVP/Development**: Use OpenVLA-7B (it's available and works)
**For Paper/Research**: Contact the SALUS authors to get access to TinyVLA-1B
**For Testing Architecture**: Use mock implementation to unblock development

