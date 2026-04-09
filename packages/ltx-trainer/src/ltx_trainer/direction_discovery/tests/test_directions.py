"""Tests for direction generation."""

import torch
from ltx_trainer.direction_discovery.latent.directions import (
    RandomDirectionGenerator,
    create_direction_generator,
)


def test_random_direction_generator():
    """Test random direction generation."""
    generator = RandomDirectionGenerator()
    latents = torch.randn(1, 128, 8, 24, 24)  # Realistic latent shape
    
    directions = generator.generate(latents, num_directions=5, seed=42)
    
    assert len(directions) == 5
    assert all(d.shape == (128, 8, 24, 24) for d in directions)
    
    # Check normalization
    for d in directions:
        norm = torch.norm(d)
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)


def test_direction_factory():
    """Test direction generator factory."""
    gen_random = create_direction_generator("random")
    gen_diff = create_direction_generator("difference")
    
    assert isinstance(gen_random, RandomDirectionGenerator)
    assert gen_diff is not None


if __name__ == "__main__":
    test_random_direction_generator()
    test_direction_factory()
    print("✓ All tests passed")