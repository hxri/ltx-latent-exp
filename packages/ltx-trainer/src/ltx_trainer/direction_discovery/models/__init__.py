"""Model interfaces for direction discovery."""

from ltx_trainer.direction_discovery.models.diffusion import DiffusionGenerationConfig, DiffusionInterface
from ltx_trainer.direction_discovery.models.vae import VAEInterface

__all__ = ["VAEInterface", "DiffusionGenerationConfig", "DiffusionInterface"]