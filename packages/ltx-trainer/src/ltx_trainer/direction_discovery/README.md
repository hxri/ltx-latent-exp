# Stable Direction Discovery in Latent Space

A modular Python codebase for discovering and evaluating latent "knobs" (directions)
that produce meaningful and reversible changes in generated videos from frozen LTX-2 diffusion inference.

## 🎯 Overview

This system enables:

1. **Latent Extraction**: Encode videos to diffusion latents using the VAE encoder
2. **Direction Generation**: Create candidate latent directions using:
   - Random sampling (N(0, I))
   - Difference-based methods (z_B - z_A)
   - Transformation-based methods (grayscale, blur, etc.)
3. **Stability Evaluation**: Measure reversibility and strength of directions
4. **Visualization**: Generate grids and metrics plots
5. **Ranking**: Identify highest-quality directions

## 🏗️ Architecture

```
direction_discovery/
├── __init__.py                    # Public API
├── __main__.py                    # CLI entry
├── main.py                        # CLI app
├── config.py                      # Pydantic configs (100 lines)
│
├── models/
│   ├── __init__.py
│   └── vae.py                     # VAE interface (150 lines)
│
├── latent/
│   ├── __init__.py
│   ├── directions.py              # Direction generators (350 lines)
│   └── controller.py              # Latent manipulation (100 lines)
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # Distance metrics (350 lines)
│   └── stability.py               # Stability evaluator (200 lines)
│
├── utils/
│   ├── __init__.py
│   ├── video_io.py                # Video I/O (200 lines)
│   └── visualization.py           # Plotting (300 lines)
│
├── experiments/
│   ├── __init__.py
│   ├── run_discovery.py           # Main experiment (350 lines)
│   └── analyze_results.py         # Result analysis (150 lines)
│
└── tests/
    ├── __init__.py
    └── test_directions.py         # Unit tests (100 lines)

Total: ~2000 lines of clean, documented code
```

## ⚡ Quick Start

### Installation

```bash
# Install dependencies
pip install torch torchvision opencv-python matplotlib numpy

# Optional: Install LPIPS for better perceptual metrics
pip install lpips
```

### Basic Usage

```bash
# Run diffusion-backed direction discovery on a video
python -m ltx_trainer.direction_discovery run-direction-discovery \
    --video-path /path/to/video.mp4 \
    --checkpoint-path /path/to/ltx2.safetensors \
  --text-encoder-path /path/to/gemma \
  --prompt "cinematic shot of the same scene" \
    --method random \
    --num-directions 10 \
    --metric l2 \
    --output-dir ./outputs/direction_discovery

# Analyze results
python -m ltx_trainer.direction_discovery analyze-results \
    --metrics-path ./outputs/direction_discovery/metrics.json \
    --top-k 10
```

## 📖 Usage Examples

### Python API

```python
from ltx_trainer.direction_discovery.models.vae import VAEInterface
from ltx_trainer.direction_discovery.latent.directions import create_direction_generator
from ltx_trainer.direction_discovery.evaluation.stability import StabilityEvaluator
from ltx_trainer.direction_discovery.evaluation.metrics import create_metric
from ltx_trainer.direction_discovery.utils.video_io import load_video
from ltx_trainer.model_loader import load_video_vae_encoder, load_video_vae_decoder

import torch

# Load video
video, fps = load_video("input.mp4", max_frames=64)

# Load VAE
device = torch.device("cuda")
encoder = load_video_vae_encoder("checkpoint.safetensors", device=device)
decoder = load_video_vae_decoder("checkpoint.safetensors", device=device)
vae = VAEInterface(encoder, decoder, device=device)

# Encode to latents
latents = vae.encode(video.unsqueeze(0))

# Generate random directions
generator = create_direction_generator("random")
directions = generator.generate(latents, num_directions=10)

# Evaluate stability
metric = create_metric("l2", device=device)
evaluator = StabilityEvaluator(vae, metric, device=device)

alphas = [-1.0, 0.0, 1.0]
results = evaluator.evaluate_directions(latents, directions, alphas)

# Print best direction
best = max(results, key=lambda r: r.quality_score)
print(f"Best direction: {best.direction_id}")
print(f"Quality score: {best.quality_score:.4f}")
print(f"Strength: {best.strength_score:.4f}")
print(f"Reversibility error: {best.reversibility_score:.6f}")
```

### Advanced: Custom Direction Generator

```python
from ltx_trainer.direction_discovery.latent.directions import DirectionGenerator
import torch

class MyCustomGenerator(DirectionGenerator):
    def generate(self, latents, num_directions, seed=42):
        """Generate custom directions."""
        torch.manual_seed(seed)
        directions = []
        
        for i in range(num_directions):
            # Your custom logic here
            v = torch.randn_like(latents[0])
            v = v / (torch.norm(v) + 1e-8)
            directions.append(v)
        
        return directions

generator = MyCustomGenerator()
directions = generator.generate(latents, num_directions=10)
```

## 🧮 Evaluation Metrics

### Quality Score

$$\text{Quality} = \frac{\text{Strength}}{\text{Reversibility} + \epsilon}$$

Where:
- **Strength**: Distance between original and modified video (higher = more effect)
- **Reversibility**: Distance between original and reversed video (lower = better reversibility)
- $\epsilon = 10^{-6}$ for numerical stability

### Supported Metrics

- **L2**: Simple Euclidean distance (fast, deterministic)
- **SSIM**: Structural Similarity Index (perception-based)
- **LPIPS**: Learned Perceptual Image Patch Similarity (requires installation)

## 🎨 Output Structure

```
outputs/direction_discovery/
├── metrics.json                    # Raw metrics (all directions)
├── analysis.json                   # Ranked and aggregated analysis
├── stability_plot.png              # Strength vs. reversibility scatter
├── direction_0_original.mp4        # Original video (top direction)
├── direction_0_grid.png            # Grid: alphas [-2, -1, 0, 1, 2]
├── direction_0_metrics.png         # Plot: distance curves
├── direction_1_original.mp4        # (next best)
├── direction_1_grid.png
└── direction_1_metrics.png
```

## 🔧 Configuration

Create a custom config file:

```yaml
# config.yaml
output_dir: "./my_outputs"
device: "cuda"
vae_tiling: true

direction_config:
  method: "random"
  num_directions: 20
  seed: 42

evaluation_config:
  metric: "lpips"
  alphas: [-2.0, -1.0, 0.0, 1.0, 2.0]

visualization_config:
  save_videos: true
  dpi: 150
```

## 🧠 Direction Generation Methods

### Random Directions

Sample from standard normal distribution:
$$v \sim \mathcal{N}(0, I)$$

**Use case**: Explore arbitrary latent space regions

```python
generator = create_direction_generator("random")
```

### Difference-Based Directions

Compute latent differences between videos:
$$v = z_B - z_A$$

**Use case**: Discover transformations between two videos

### Transformation-Based Directions

Apply image transforms and compute latent differences:
$$v = z(\text{original}) - z(\text{transformed})$$

**Transforms**: Grayscale, Blur, Brightness, Contrast

**Use case**: Find interpretable, semantic directions

## 📊 Metrics Output

Example `metrics.json`:

```json
{
  "config": {
    "method": "random",
    "num_directions": 10,
    "metric": "l2",
    "alphas": [-2.0, -1.0, 0.0, 1.0, 2.0]
  },
  "results": [
    {
      "direction_id": 3,
      "quality_score": 0.8542,
      "strength_score": 0.1234,
      "reversibility_score": 0.0001234,
      "alphas": [-2.0, -1.0, 0.0, 1.0, 2.0],
      "distances_forward": {
        "-2.0": 0.2468,
        "-1.0": 0.1234,
        "0.0": 0.0,
        "1.0": 0.1234,
        "2.0": 0.2468
      },
      "distances_reversed": {
        "-2.0": 0.0002468,
        "1.0": 0.0001234,
        ...
      }
    }
    ...
  ]
}
```

## ⚡ Performance Tuning

### For Large Videos

```python
# Enable VAE tiling for large videos
vae = VAEInterface(encoder, decoder, device=device, vae_tiling=True)

# Or via config
config.vae_tiling = True
```

### For Speed

```python
# Use L2 metric (fastest)
metric = create_metric("l2")

# Reduce number of directions
num_directions = 5

# Reduce alpha values tested
alphas = [-1.0, 0.0, 1.0]
```

### For Quality

```python
# Use LPIPS or SSIM
metric = create_metric("lpips")

# More alphas
alphas = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

# More directions
num_directions = 20
```

## 🐛 Troubleshooting

**LPIPS not available**
```python
# Falls back to L2 distance
metric = create_metric("lpips")  # Uses L2 if LPIPS not installed
```

**GPU out of memory**
```python
# Enable tiling
config.vae_tiling = True

# Reduce max frames during loading
video, fps = load_video("video.mp4", max_frames=32)

# Use reduced precision
dtype = torch.float32  # instead of bfloat16
```

**Slow evaluation**
```python
# Reduce directions
num_directions = 5

# Use L2 metric
metric = create_metric("l2")

# Disable visualization
config.visualization_config.save_videos = False
```

## 🔗 Integration with LTX-2

This codebase is designed to integrate seamlessly with LTX-2:

```python
from ltx_core.model.video_vae import VideoEncoder, VideoDecoder
from ltx_trainer.direction_discovery.models.vae import VAEInterface

# Load LTX-2 components
encoder: VideoEncoder = ...
decoder: VideoDecoder = ...

# Wrap in VAEInterface
vae = VAEInterface(encoder, decoder, device=device)

# Use with direction discovery
directions = generator.generate(latents, num_directions=10)
```

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{direction_discovery_2024,
  title={Stable Direction Discovery in Latent Space},
  author={Your Name},
  year={2024}
}
```

## 📄 License

Same as LTX-2 repository

## 🤝 Contributing

Contributions welcome! Please:

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Create pull requests with clear descriptions

---

**Questions?** Open an issue or refer to the main LTX-2 documentation.