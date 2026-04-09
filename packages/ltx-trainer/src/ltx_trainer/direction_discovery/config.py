"""Configuration management for direction discovery experiments."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class DirectionGenerationConfig(BaseModel):
    """Configuration for direction generation methods."""

    method: Literal["random", "difference", "transformation"] = Field(
        default="random",
        description="Direction generation method",
    )
    num_directions: int = Field(
        default=10,
        description="Number of directions to generate",
        ge=1,
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    normalize: bool = Field(
        default=True,
        description="Normalize directions to unit length",
    )


class EvaluationConfig(BaseModel):
    """Configuration for stability evaluation."""

    metric: Literal["lpips", "ssim", "l2"] = Field(
        default="l2",
        description="Distance metric to use",
    )
    alphas: list[float] = Field(
        default=[-2.0, -1.0, 0.0, 1.0, 2.0],
        description="Strength multipliers to test",
    )
    device: str = Field(
        default="cuda",
        description="Device to use for computation",
    )
    dtype: str = Field(
        default="bfloat16",
        description="Data type for computation",
    )


class VisualizationConfig(BaseModel):
    """Configuration for visualization and output."""

    save_videos: bool = Field(
        default=True,
        description="Save video files for each direction",
    )
    save_frames: bool = Field(
        default=True,
        description="Save frame grids for visualization",
    )
    frame_indices: list[int] = Field(
        default=[0, -1],
        description="Frame indices to visualize (first and last)",
    )
    dpi: int = Field(
        default=100,
        description="DPI for saved figures",
    )


class DirectionDiscoveryConfig(BaseModel):
    """Main configuration for direction discovery experiments."""

    # Paths
    output_dir: Path = Field(
        default=Path("outputs/direction_discovery"),
        description="Output directory for results",
    )
    device: str = Field(
        default="cuda",
        description="Device to use (cuda/cpu)",
    )
    prompt: str = Field(
        default="",
        description="Prompt used when diffusion-backed decoding is enabled",
    )
    negative_prompt: str = Field(
        default="",
        description="Negative prompt used for diffusion-backed decoding",
    )
    checkpoint_path: Path | None = Field(
        default=None,
        description="Path to the LTX checkpoint",
    )
    text_encoder_path: Path | None = Field(
        default=None,
        description="Path to the Gemma text encoder directory for diffusion-backed decoding",
    )
    use_diffusion: bool = Field(
        default=True,
        description="Use the full frozen diffusion model for denoising edited latents",
    )
    num_inference_steps: int = Field(
        default=30,
        description="Number of diffusion denoising steps",
        ge=1,
    )
    guidance_scale: float = Field(
        default=4.0,
        description="Classifier-free guidance scale for diffusion-backed decoding",
    )
    stg_scale: float = Field(
        default=0.0,
        description="Optional STG scale during diffusion-backed decoding",
    )
    stg_blocks: list[int] | None = Field(
        default=None,
        description="Optional transformer blocks to perturb for STG",
    )
    stg_mode: Literal["stg_av", "stg_v"] = Field(
        default="stg_v",
        description="STG mode for diffusion-backed decoding",
    )

    # Components
    direction_config: DirectionGenerationConfig = Field(
        default_factory=DirectionGenerationConfig,
    )
    evaluation_config: EvaluationConfig = Field(
        default_factory=EvaluationConfig,
    )
    visualization_config: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
    )

    # VAE settings
    vae_tiling: bool = Field(
        default=False,
        description="Enable VAE tiling for large videos",
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True