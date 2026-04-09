"""
Stable Direction Discovery in Latent Space for Video Diffusion Pipelines.

This package provides tools for discovering and evaluating latent "knobs" (directions)
that produce meaningful and reversible changes in generated videos.

Quick Start:
    >>> from ltx_trainer.direction_discovery.experiments.run_discovery import run_direction_discovery
    >>> run_direction_discovery(
    ...     video_path="input.mp4",
    ...     checkpoint_path="model.safetensors",
    ...     method="random",
    ...     num_directions=10
    ... )

Modules:
    - config: Configuration schemas
    - models: VAE interface
    - latent: Direction generation and control
    - evaluation: Stability metrics
    - utils: Video I/O and visualization
    - experiments: Main experiment scripts
"""

__version__ = "0.1.0"

from ltx_trainer.direction_discovery.config import DirectionDiscoveryConfig
from ltx_trainer.direction_discovery.evaluation import (
    DistanceMetric,
    L2Metric,
    LPIPSMetric,
    SSIMMetric,
    StabilityEvaluator,
    StabilityResults,
    create_metric,
)
from ltx_trainer.direction_discovery.latent import (
    DirectionGenerator,
    LatentController,
    LatentManipulation,
    RandomDirectionGenerator,
    create_direction_generator,
)
from ltx_trainer.direction_discovery.models import DiffusionGenerationConfig, DiffusionInterface, VAEInterface
from ltx_trainer.direction_discovery.utils import (
    create_direction_grid,
    create_metric_curve,
    create_reversibility_plot,
    extract_frames,
    load_video,
    save_video,
)

__all__ = [
    # Config
    "DirectionDiscoveryConfig",
    # Models
    "VAEInterface",
    "DiffusionGenerationConfig",
    "DiffusionInterface",
    # Latent
    "DirectionGenerator",
    "RandomDirectionGenerator",
    "LatentController",
    "LatentManipulation",
    "create_direction_generator",
    # Evaluation
    "DistanceMetric",
    "L2Metric",
    "SSIMMetric",
    "LPIPSMetric",
    "StabilityEvaluator",
    "StabilityResults",
    "create_metric",
    # Utils
    "load_video",
    "save_video",
    "extract_frames",
    "create_direction_grid",
    "create_reversibility_plot",
    "create_metric_curve",
]