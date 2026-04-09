"""Latent manipulation and control."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LatentManipulation:
    """Result of applying a direction to a latent."""

    original: torch.Tensor  # Original latent
    direction: torch.Tensor  # Direction vector
    alpha: float  # Strength multiplier
    modified: torch.Tensor  # z + alpha * v
    reversed: torch.Tensor  # (z + alpha * v) - alpha * v


class LatentController:
    """Controller for applying and reversing latent directions."""

    # Keep latent updates large enough to survive bfloat16 quantization while staying moderate.
    _DEFAULT_DIRECTION_RMS = 0.05

    @staticmethod
    def _normalized_direction(direction: torch.Tensor) -> torch.Tensor:
        """Return direction normalized to unit L2 norm in float32 for stable arithmetic."""
        direction_fp32 = direction.float()
        return direction_fp32 / (torch.norm(direction_fp32) + 1e-8)

    @staticmethod
    def _delta_scale(direction: torch.Tensor, target_rms: float) -> float:
        """Compute scalar so normalized direction has approximately target RMS per element."""
        return target_rms * (direction.numel() ** 0.5)

    @staticmethod
    def apply_direction(
        latents: torch.Tensor,
        direction: torch.Tensor,
        alpha: float = 1.0,
        target_rms: float = _DEFAULT_DIRECTION_RMS,
    ) -> torch.Tensor:
        """
        Apply a direction to latents.

        Args:
            latents: Original latent [B, C, F, H, W]
            direction: Direction vector [C, F, H, W]
            alpha: Strength multiplier
            target_rms: Desired RMS magnitude of the unit-step perturbation

        Returns:
            Modified latent z + alpha * v
        """
        if direction.dim() == latents.dim() - 1:
            direction = direction.unsqueeze(0)

        direction_unit = LatentController._normalized_direction(direction)
        delta_scale = LatentController._delta_scale(direction, target_rms)
        delta = (alpha * delta_scale) * direction_unit

        return latents + delta.to(latents.dtype)

    @staticmethod
    def reverse_direction(
        modified_latents: torch.Tensor,
        direction: torch.Tensor,
        alpha: float = 1.0,
        target_rms: float = _DEFAULT_DIRECTION_RMS,
    ) -> torch.Tensor:
        """
        Reverse the application of a direction.

        Args:
            modified_latents: Modified latent (after applying direction)
            direction: Same direction vector used
            alpha: Same strength multiplier
            target_rms: Desired RMS magnitude of the unit-step perturbation

        Returns:
            Original latent (approximately)
        """
        if direction.dim() == modified_latents.dim() - 1:
            direction = direction.unsqueeze(0)

        direction_unit = LatentController._normalized_direction(direction)
        delta_scale = LatentController._delta_scale(direction, target_rms)
        delta = (alpha * delta_scale) * direction_unit

        return modified_latents - delta.to(modified_latents.dtype)

    @staticmethod
    def interpolate_directions(
        latents: torch.Tensor,
        directions: list[torch.Tensor],
        alphas: list[float],
        vae: "VAEInterface",  # type: ignore
    ) -> dict[str, torch.Tensor]:
        """
        Generate interpolations along multiple directions.

        Args:
            latents: Original latent
            directions: List of direction vectors
            alphas: Strength multipliers to test
            vae: VAE interface for decoding

        Returns:
            Dictionary with decoded videos for each (direction, alpha) pair
        """
        results = {}

        for dir_idx, direction in enumerate(directions):
            for alpha in alphas:
                key = f"direction_{dir_idx}_alpha_{alpha:.2f}"
                modified = LatentController.apply_direction(latents, direction, alpha)
                video = vae.decode(modified)
                results[key] = video

        return results