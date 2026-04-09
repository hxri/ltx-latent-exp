import logging
import os
import sys
from logging import getLogger
from pathlib import Path

from rich.logging import RichHandler

# Get the process rank
IS_MULTI_GPU = os.environ.get("LOCAL_RANK") is not None
RANK = int(os.environ.get("LOCAL_RANK", "0"))

# Configure with Rich
logging.basicConfig(
    level="INFO",
    format=f"\\[rank {RANK}] %(message)s" if IS_MULTI_GPU else "%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=False,
            markup=True,
        )
    ],
)

# Get the logger and configure it
logger = getLogger("ltxv_trainer")
logger.setLevel(logging.DEBUG)
logger.propagate = True

# Set level based on process
if RANK != 0:
    logger.setLevel(logging.WARNING)

# Expose common logging functions directly
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical


# Add the root directory to the Python path so we can import from scripts.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

__all__ = [
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "logger",
]

# Direction Discovery subpackage
try:
    from ltx_trainer.direction_discovery import (
        DirectionDiscoveryConfig,
        DiffusionGenerationConfig,
        DiffusionInterface,
        VAEInterface,
        LatentController,
        LatentManipulation,
        create_direction_generator,
        StabilityEvaluator,
        StabilityResults,
        create_metric,
        load_video,
        save_video,
        create_direction_grid,
    )
    
    __all__.extend([
        "DirectionDiscoveryConfig",
        "DiffusionGenerationConfig",
        "DiffusionInterface",
        "VAEInterface",
        "LatentController",
        "LatentManipulation",
        "create_direction_generator",
        "StabilityEvaluator",
        "StabilityResults",
        "create_metric",
        "load_video",
        "save_video",
        "create_direction_grid",
    ])
except Exception:
    pass  # Direction Discovery not available
