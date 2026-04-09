# Direction Discovery - Complete Usage Guide

## Getting Started

### 1. Basic Example

```bash
# Run direction discovery on a video
python -m ltx_trainer.direction_discovery run-direction-discovery \
    --video-path input.mp4 \
    --checkpoint-path ltx2.safetensors \
    --method random \
    --num-directions 10 \
    --output-dir ./outputs/my_experiment
```

### 2. What Happens

1. **Video Loading**: Loads the video and converts to tensor
2. **VAE Encoding**: Encodes video to latent space using LTX-2's VAE
3. **Direction Generation**: Creates 10 random directions in latent space
4. **Evaluation**: For each direction, tests 5 alpha values [-2, -1, 0, 1, 2]
5. **Metrics**: Computes stability and reversibility scores
6. **Visualization**: Generates grids and plots showing results
7. **Ranking**: Lists directions by quality

### 3. Output

Check `./outputs/my_experiment/`:
- `metrics.json`: Raw metrics for all directions
- `stability_plot.png`: Scatter plot of strength vs. reversibility
- `direction_0_grid.png`: Visual grid of alpha progression
- `direction_0_metrics.png`: Metrics curve plot
- `direction_0_original.mp4`: Video file (if enabled)

## Direction Generation Methods

### Random Directions (Fastest)

```bash
python -m ltx_trainer.direction_discovery run-direction-discovery \
    --video-path video.mp4 \
    --checkpoint-path ltx2.safetensors \
    --method random \
    --num-directions 20
```

- Explores arbitrary latent space regions
- Fast computation
- Good for understanding general latent structure

### Difference-Based Directions (Medium)

```bash
# Compute: z_video2 - z_video1
# Encodes the semantic change between two videos
```

- Discovers semantic transformations
- Requires two videos
- More interpretable directions

### Transformation-Based Directions (Slower)

```bash
python -m ltx_trainer.direction_discovery run-direction-discovery \
    --video-path video.mp4 \
    --checkpoint-path ltx2.safetensors \
    --method transformation \
    --num-directions 5
```

- Applies grayscale, blur, brightness, contrast
- Most interpretable
- Good for understanding VAE latent structure

## Metrics Explained

### Strength Score

"How much does applying this direction change the video?"

- **Low** (< 0.05): Weak effect, subtle changes
- **Medium** (0.05-0.2): Noticeable but not overwhelming
- **High** (> 0.2): Strong, obvious changes

### Reversibility Score

"How well can we undo the direction?" (lower is better)

- **Excellent** (< 0.001): Perfect reversibility
- **Good** (0.001-0.01): Nearly perfect
- **Poor** (> 0.01): Direction application loses information

### Quality Score

$$\text{Quality} = \frac{\text{Strength}}{\text{Reversibility}}$$

- **High quality**: Strong effect + good reversibility
- **Ideal**: Strong enough to see effect, reversible enough to undo

## Interpreting Results

### Example Results Table

| Rank | Quality | Strength | Reversibility | Interpretation |
|------|---------|----------|---------------|---|
| 1 | 0.854 | 0.123 | 0.000144 | ✓ Excellent: strong, reversible direction |
| 2 | 0.425 | 0.089 | 0.000209 | ✓ Good: solid direction |
| 3 | 0.012 | 0.001 | 0.085 | ✗ Poor: weak effect, hard to reverse |

### What Makes a Good Direction?

1. **Visible Effect** (Strength > 0.05)
   - Changes should be noticeable
   - At least ~10% change in latent values

2. **Good Reversibility** (Reversibility < 0.01)
   - Original should recoverable
   - Low information loss

3. **Consistent Across Alphas**
   - Effect scales smoothly with alpha
   - No sudden jumps or artifacts

## Advanced Usage

### Custom Alpha Values

```python
from ltx_trainer.direction_discovery.experiments.run_discovery import run_direction_discovery
from ltx_trainer.direction_discovery.config import DirectionDiscoveryConfig

config = DirectionDiscoveryConfig()
config.evaluation_config.alphas = [-3.0, -1.0, 0.0, 1.0, 3.0]  # Custom alphas
```

### Batch Processing

```bash
# Process multiple videos
for video in videos/*.mp4; do
    python -m ltx_trainer.direction_discovery run-direction-discovery \
        --video-path "$video" \
        --checkpoint-path ltx2.safetensors \
        --output-dir "./outputs/$(basename "$video" .mp4)"
done

# Compare results
python analyze_results.py ./outputs/method_random/metrics.json
python analyze_results.py ./outputs/method_difference/metrics.json
python analyze_results.py ./outputs/method_transformation/metrics.json
```

### Compare Methods

```bash
# Run each method on same video
for method in random difference transformation; do
    python -m ltx_trainer.direction_discovery run-direction-discovery \
        --video-path video.mp4 \
        --checkpoint-path ltx2.safetensors \
        --method "$method" \
        --output-dir "./outputs/method_$method"
done

# Compare results
python analyze_results.py ./outputs/method_random/metrics.json
python analyze_results.py ./outputs/method_difference/metrics.json
python analyze_results.py ./outputs/method_transformation/metrics.json
```

## Performance Optimization

### Memory Usage

```bash
# Enable VAE tiling for large videos
python -m ltx_trainer.direction_discovery run-direction-discovery \
    --video-path large_video.mp4 \
    --checkpoint-path ltx2.safetensors \
    --vae-tiling  # Enable tiling
```

### Speed

```bash
# Fast mode: fewer directions, lower metric quality, no videos
python -m ltx_trainer.direction_discovery run-direction-discovery \
    --video-path video.mp4 \
    --checkpoint-path ltx2.safetensors \
    --method random \
    --num-directions 5 \
    --metric l2  # Fastest metric
```

### Quality

```bash
# High quality: more directions, LPIPS metric, save all videos
python -m ltx_trainer.direction_discovery run-direction-discovery \
    --video-path video.mp4 \
    --checkpoint-path ltx2.safetensors \
    --method transformation \
    --num-directions 20 \
    --metric lpips  # Most accurate perceptual metric
```

## Troubleshooting

### "CUDA out of memory"

1. Enable VAE tiling:
   ```bash
   --vae-tiling
   ```

2. Reduce frames:
   ```python
   video, fps = load_video("video.mp4", max_frames=32)
   ```

3. Reduce directions:
   ```bash
   --num-directions 3
   ```

### "LPIPS not installed"

Install it:
```bash
pip install lpips
```

Or use L2 metric (automatic fallback):
```bash
--metric l2
```

### "Direction visualization is poor quality"

Increase DPI in config:
```yaml
visualization_config:
  dpi: 200  # Higher DPI = better quality
```

## Next Steps

### Use Discovered Directions

```python
from ltx_trainer.direction_discovery.models.vae import VAEInterface
from ltx_trainer.direction_discovery.latent.controller import LatentController

# Load best direction from metrics.json
best_dir_id = 3  # From analysis

# Apply direction with custom alpha
z_modified = LatentController.apply_direction(latents, directions[best_dir_id], alpha=1.5)
modified_video = vae.decode(z_modified)
```

### Training with Directions

Use discovered directions as conditioning for fine-tuning:
- Apply random alphas during training
- Use high-quality directions as regularization

### Semantic Analysis

Analyze what each direction represents:
- Top directions often encode semantic changes
- Use SAE (Sparse Autoencoders) for interpretation
- Group directions by similarity

## FAQ

**Q: How many directions should I generate?**
A: Start with 10-20. More gives better coverage (slower), fewer is faster.

**Q: Which metric should I use?**
A: L2 (speed) > SSIM (balance) > LPIPS (accuracy). Use LPIPS for final analysis.

**Q: How do I choose alpha values?**
A: Default [-2, -1, 0, 1, 2] is good. Adjust based on effect strength.

**Q: Can I use directions from one video on another?**
A: Yes! Directions are learned from latent structure, likely transfer across similar content.

**Q: What does negative alpha do?**
A: Applies direction in opposite direction (z - alpha * v instead of z + alpha * v).

## Further Reading

- [LTX-2 Documentation](../README.md)
- [VAE Latent Space Visualization](https://distill.pub/2020/circuit-search/)
- [Learned Perceptual Metrics](https://github.com/richzhang/PerceptualSimilarity)

---

Happy exploring! 🚀