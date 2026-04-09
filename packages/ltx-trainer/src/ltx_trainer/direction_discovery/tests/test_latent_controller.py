"""Tests for latent direction application and reversal."""

import torch

from ltx_trainer.direction_discovery.latent.controller import LatentController


def test_apply_direction_changes_bfloat16_latents():
    """Applying a unit direction should not become a no-op in bfloat16."""
    latents = torch.randn(1, 32, 4, 8, 8, dtype=torch.bfloat16)
    direction = torch.randn(32, 4, 8, 8, dtype=torch.bfloat16)
    direction = direction / (torch.norm(direction.float()) + 1e-8)

    modified = LatentController.apply_direction(latents, direction, alpha=1.0)

    assert not torch.equal(modified, latents)
    assert torch.norm((modified - latents).float()) > 0.0


def test_reverse_direction_recovers_original_with_tolerance():
    """Reversing the same edit should approximately recover original latents."""
    latents = torch.randn(1, 16, 4, 4, 4, dtype=torch.float32)
    direction = torch.randn(16, 4, 4, 4, dtype=torch.float32)
    direction = direction / (torch.norm(direction) + 1e-8)

    modified = LatentController.apply_direction(latents, direction, alpha=1.5)
    recovered = LatentController.reverse_direction(modified, direction, alpha=1.5)

    assert torch.allclose(recovered, latents, atol=1e-5, rtol=1e-5)
