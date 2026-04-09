"""Methods for generating latent directions (knobs)."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class DirectionGenerator(ABC):
    """Base class for latent direction generation."""

    @abstractmethod
    def generate(self, latents: torch.Tensor, num_directions: int, seed: int = 42) -> list[torch.Tensor]:
        """
        Generate latent directions.

        Args:
            latents: Reference latent tensor(s)
            num_directions: Number of directions to generate
            seed: Random seed

        Returns:
            List of direction vectors
        """
        pass


class RandomDirectionGenerator(DirectionGenerator):
    """Generate random directions in latent space."""

    def generate(self, latents: torch.Tensor, num_directions: int, seed: int = 42) -> list[torch.Tensor]:
        """
        Generate random directions.

        Args:
            latents: Reference latent tensor [B, C, F, H, W]
            num_directions: Number of directions
            seed: Random seed

        Returns:
            List of normalized random directions
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        directions = []
        latent_shape = latents.shape[1:]  # [C, F, H, W]

        for _ in range(num_directions):
            # Sample from standard normal distribution
            v = torch.randn(*latent_shape, device=latents.device, dtype=latents.dtype)

            # Normalize to unit length
            v = v / (torch.norm(v) + 1e-8)
            directions.append(v)

        return directions


class DifferenceDirectionGenerator(DirectionGenerator):
    """Generate directions from differences between encoded videos."""

    def generate(
        self,
        latents: torch.Tensor,
        num_directions: int,
        seed: int = 42,
        reference_latents: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """
        Generate directions from latent differences.

        Args:
            latents: Primary latent tensor [B, C, F, H, W]
            num_directions: Number of directions (uses batch dimension)
            seed: Random seed
            reference_latents: Optional reference for computing differences

        Returns:
            List of difference directions
        """
        if reference_latents is None:
            reference_latents = latents.mean(dim=0, keepdim=True)

        directions = []

        # Use batch samples or repeat reference
        for i in range(min(num_directions, latents.shape[0])):
            v = latents[i] - reference_latents.squeeze(0)
            v = v / (torch.norm(v) + 1e-8)
            directions.append(v)

        return directions


class TransformationDirectionGenerator(DirectionGenerator):
    """Generate directions from simple image transformations."""

    def generate(
        self,
        latents: torch.Tensor,
        vae: "VAEInterface",  # type: ignore
        num_directions: int,
        seed: int = 42,
    ) -> list[torch.Tensor]:
        """
        Generate directions from transformations applied to decoded videos.

        Args:
            latents: Reference latent tensor [B, C, F, H, W]
            vae: VAE interface for encode/decode
            num_directions: Number of directions to generate
            seed: Random seed

        Returns:
            List of transformation-based directions
        """
        import torch.nn.functional as F

        torch.manual_seed(seed)
        np.random.seed(seed)

        directions = []

        # Decode original
        video = vae.decode(latents)  # [B, C, F, H, W]

        # Define transformations
        transform_fns = [
            self._grayscale,
            self._blur,
            self._brightness,
            self._contrast,
        ]

        for i, transform_fn in enumerate(transform_fns):
            if i >= num_directions:
                break

            try:
                # Apply transformation
                transformed_video = transform_fn(video)

                # Encode transformed video
                transformed_latents = vae.encode(transformed_video)

                # Compute direction
                v = transformed_latents - latents
                v = v / (torch.norm(v) + 1e-8)
                directions.append(v)
            except Exception as e:
                print(f"Warning: transformation {i} failed: {e}")

        return directions

    @staticmethod
    def _grayscale(video: torch.Tensor) -> torch.Tensor:
        """Convert video to grayscale."""
        # video: [B, C, F, H, W]
        gray = video.mean(dim=1, keepdim=True)
        # Blend with original
        return 0.3 * video + 0.7 * gray

    @staticmethod
    def _blur(video: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Apply Gaussian blur to video."""
        import torch.nn.functional as F

        B, C, F, H, W = video.shape
        # Reshape to [B*F, C, H, W] for conv
        video_flat = video.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)

        # Simple box blur
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=video.device) / (kernel_size**2)
        kernel = kernel.repeat(C, 1, 1, 1)

        blurred = F.conv2d(video_flat, kernel, padding=kernel_size // 2, groups=C)
        # Reshape back
        blurred = blurred.reshape(B, F, C, H, W).permute(0, 2, 1, 3, 4)
        return blurred

    @staticmethod
    def _brightness(video: torch.Tensor, factor: float = 0.2) -> torch.Tensor:
        """Adjust brightness."""
        return video * (1 + factor)

    @staticmethod
    def _contrast(video: torch.Tensor, factor: float = 0.1) -> torch.Tensor:
        """Adjust contrast around mean."""
        mean = video.mean()
        return mean + (video - mean) * (1 + factor)


def create_direction_generator(method: str) -> DirectionGenerator:
    """Factory function to create direction generator."""
    generators = {
        "random": RandomDirectionGenerator,
        "difference": DifferenceDirectionGenerator,
        "transformation": TransformationDirectionGenerator,
    }

    if method not in generators:
        raise ValueError(f"Unknown direction method: {method}")

    return generators[method]()