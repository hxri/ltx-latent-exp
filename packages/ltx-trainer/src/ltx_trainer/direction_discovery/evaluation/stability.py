"""Stability evaluation for latent directions."""

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from ltx_trainer.direction_discovery.evaluation.metrics import DistanceMetric
from ltx_trainer.direction_discovery.latent.controller import LatentController


@dataclass
class StabilityResults:
    """Results of stability evaluation for a direction."""

    direction_id: int
    strength_score: float  # LPIPS(x0, x1)
    reversibility_score: float  # LPIPS(x0, x2)
    quality_score: float  # Overall quality (strength / reversibility)
    alphas: list[float] = field(default_factory=list)
    distances_forward: dict[float, float] = field(default_factory=dict)  # alpha -> distance
    distances_reversed: dict[float, float] = field(default_factory=dict)  # alpha -> distance

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "direction_id": self.direction_id,
            "strength_score": self.strength_score,
            "reversibility_score": self.reversibility_score,
            "quality_score": self.quality_score,
            "alphas": self.alphas,
            "distances_forward": self.distances_forward,
            "distances_reversed": self.distances_reversed,
        }


class StabilityEvaluator:
    """Evaluates the stability and reversibility of latent directions."""

    def __init__(
        self,
        vae: object,
        metric: DistanceMetric,
        device: torch.device = torch.device("cpu"),
        edit_after_diffusion: bool = False,
    ):
        """
        Initialize evaluator.

        Args:
            vae: VAE interface for encoding/decoding
            metric: Distance metric to use
            device: Device for computation
            edit_after_diffusion: Apply direction after diffusion denoising (DiffusionInterface only)
        """
        self.vae = vae
        self.metric = metric
        self.device = device
        self.edit_after_diffusion = edit_after_diffusion

    @torch.inference_mode()
    def evaluate_direction(
        self,
        latents: torch.Tensor,
        direction: torch.Tensor,
        alphas: list[float],
        direction_id: int = 0,
        target_rms: float = 0.05,
        step_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> StabilityResults:
        """
        Evaluate stability and reversibility of a single direction.

        Args:
            latents: Original latent [B, C, F, H, W]
            direction: Direction vector [C, F, H, W]
            alphas: Strength multipliers to test
            direction_id: Index for this direction

        Returns:
            StabilityResults object
        """
        post_diffusion_mode = (
            self.edit_after_diffusion
            and hasattr(self.vae, "denoise_latents")
            and hasattr(self.vae, "decode_from_denoised_latents")
        )

        if post_diffusion_mode:
            latent_base = self.vae.denoise_latents(latents)
            x0 = self.vae.decode_from_denoised_latents(latent_base)
        else:
            latent_base = latents
            x0 = self.vae.decode(latent_base)

        distances_forward = {}
        distances_reversed = {}

        for alpha in alphas:
            # Apply direction
            if post_diffusion_mode:
                modified_latents = LatentController.apply_direction(
                    latent_base,
                    direction,
                    alpha,
                    target_rms=target_rms,
                )
                x1 = self.vae.decode_from_denoised_latents(modified_latents)
            else:
                modified_latents = LatentController.apply_direction(
                    latent_base,
                    direction,
                    alpha,
                    target_rms=target_rms,
                )
                x1 = self.vae.decode(modified_latents)

            # Reverse direction
            reversed_latents = LatentController.reverse_direction(
                modified_latents,
                direction,
                alpha,
                target_rms=target_rms,
            )
            if post_diffusion_mode:
                x2 = self.vae.decode_from_denoised_latents(reversed_latents)
            else:
                x2 = self.vae.decode(reversed_latents)

            # Compute distances
            distances_forward[alpha] = self.metric.compute(x0, x1)
            distances_reversed[alpha] = self.metric.compute(x0, x2)

            if step_callback is not None:
                step_callback(len(distances_forward), len(alphas), alpha)

        # Compute summary scores at alpha=1.0
        if 1.0 in distances_forward:
            strength_score = distances_forward[1.0]
            reversibility_score = distances_reversed[1.0]
        else:
            # Use closest alpha
            closest_alpha = min(alphas, key=lambda a: abs(a - 1.0))
            strength_score = distances_forward[closest_alpha]
            reversibility_score = distances_reversed[closest_alpha]

        # Quality score: higher strength with lower reversibility error
        quality_score = strength_score / (reversibility_score + 1e-6)

        return StabilityResults(
            direction_id=direction_id,
            strength_score=strength_score,
            reversibility_score=reversibility_score,
            quality_score=quality_score,
            alphas=alphas,
            distances_forward=distances_forward,
            distances_reversed=distances_reversed,
        )

    @torch.inference_mode()
    def evaluate_directions(
        self,
        latents: torch.Tensor,
        directions: list[torch.Tensor],
        alphas: list[float],
        target_rms: float = 0.05,
    ) -> list[StabilityResults]:
        """
        Evaluate multiple directions.

        Args:
            latents: Original latent
            directions: List of direction vectors
            alphas: Strength multipliers to test

        Returns:
            List of StabilityResults
        """
        results = []
        for dir_idx, direction in enumerate(directions):
            result = self.evaluate_direction(
                latents,
                direction,
                alphas,
                direction_id=dir_idx,
                target_rms=target_rms,
            )
            results.append(result)

        return results