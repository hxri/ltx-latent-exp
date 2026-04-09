"""VAE wrapper for encoding and decoding videos."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ltx_core.model.video_vae import VideoDecoder, VideoEncoder


class VAEInterface:
    """Clean interface for VAE encoding and decoding."""

    def __init__(
        self,
        encoder: VideoEncoder,
        decoder: VideoDecoder,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        vae_tiling: bool = False,
    ):
        """
        Initialize VAE interface.

        Args:
            encoder: Video VAE encoder
            decoder: Video VAE decoder
            device: Target device
            dtype: Data type for computation
            vae_tiling: Whether to use tiled encoding/decoding
        """
        self.encoder = encoder.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.device = device
        self.dtype = dtype
        self.vae_tiling = vae_tiling

    @torch.inference_mode()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent space.

        Args:
            video: Video tensor [B, C, F, H, W] or [C, F, H, W]

        Returns:
            Latent tensor [B, C_latent, F_latent, H_latent, W_latent]
        """
        # Add batch dim if needed
        if video.dim() == 4:
            video = video.unsqueeze(0)

        video = video.to(self.device, dtype=self.dtype)

        if self.vae_tiling:
            # Use tiled encoding for large videos
            latents = self.encoder.tiled_encode(video)
            latents = list(latents)  # Collect generator output
            latents = torch.cat(latents, dim=2)  # Concatenate along temporal dim
        else:
            latents = self.encoder(video)

        return latents.detach()

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Decode latents to video space.

        Args:
            latents: Latent tensor [B, C_latent, F_latent, H_latent, W_latent]
            generator: Random generator for stochastic decoding

        Returns:
            Video tensor [B, C, F, H, W]
        """
        latents = latents.to(self.device, dtype=self.dtype)

        if self.vae_tiling:
            # Use tiled decoding
            from ltx_core.model.video_vae import TilingConfig

            tiling_config = TilingConfig.default()
            chunks = list(self.decoder.tiled_decode(latents, tiling_config=tiling_config, generator=generator))
            video = torch.cat(chunks, dim=2)  # [B, C, F, H, W]
        else:
            video = self.decoder(latents, generator=generator)

        return video.detach()

    def get_latent_shape(self, video_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get expected latent shape for given video shape."""
        # VAE compression factors: spatial 32x, temporal 8x (with +1 offset)
        B, C, F, H, W = video_shape
        latent_shape = (B, 128, 1 + (F - 1) // 8, H // 32, W // 32)
        return latent_shape