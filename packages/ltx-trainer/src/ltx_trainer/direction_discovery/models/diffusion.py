"""Inference-only diffusion wrapper for latent direction discovery."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import CFGGuider, STGGuider
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier, VideoLatentShape
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import X0Model
from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core.tools import VideoLatentTools
from ltx_core.types import LatentState, SpatioTemporalScaleFactors, VideoPixelShape
from ltx_trainer.validation_sampler import TiledDecodingConfig

if TYPE_CHECKING:
    from ltx_core.model.transformer import LTXModel
    from ltx_core.model.video_vae import VideoDecoder, VideoEncoder
    from ltx_core.text_encoders.gemma import GemmaTextEncoder
    from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


@dataclass(frozen=True)
class DiffusionGenerationConfig:
    """Inference-only diffusion configuration for direction discovery."""

    prompt: str = ""
    negative_prompt: str = ""
    num_inference_steps: int = 30
    guidance_scale: float = 4.0
    seed: int = 42
    noise_scale: float = 1.0
    stg_scale: float = 0.0
    stg_blocks: list[int] | None = None
    stg_mode: Literal["stg_av", "stg_v"] = "stg_v"
    tiled_decoding: TiledDecodingConfig | Literal[False] | None = None

    def resolved_tiled_decoding(self) -> TiledDecodingConfig:
        if self.tiled_decoding is False:
            return TiledDecodingConfig(enabled=False)
        if self.tiled_decoding is None:
            return TiledDecodingConfig()
        return self.tiled_decoding


class DiffusionInterface:
    """Inference-only interface that denoises edited video latents with the LTX transformer."""

    def __init__(
        self,
        transformer: "LTXModel",
        encoder: "VideoEncoder",
        decoder: "VideoDecoder",
        text_encoder: "GemmaTextEncoder",
        embeddings_processor: "EmbeddingsProcessor",
        *,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        generation_config: DiffusionGenerationConfig,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.transformer = transformer.eval()
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()
        self.text_encoder = text_encoder.eval()
        self.embeddings_processor = embeddings_processor.eval()
        self.device = device
        self.dtype = dtype
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        self.generation_config = generation_config
        self.patchifier = VideoLatentPatchifier(patch_size=1)
        self.video_tools = VideoLatentTools(
            patchifier=self.patchifier,
            target_shape=VideoLatentShape.from_pixel_shape(
                VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=frame_rate)
            ),
            fps=frame_rate,
            scale_factors=VIDEO_SCALE_FACTORS,
            causal_fix=True,
        )
        self._cached_prompt_embeddings: tuple[Tensor, Tensor | None] | None = None

    @torch.no_grad()
    def encode(self, video: Tensor) -> Tensor:
        """Encode video to patchified diffusion latents."""
        if video.dim() == 4:
            video = video.unsqueeze(0)

        self._validate_video_shape(video)
        video = (video.to(self.device, dtype=torch.float32) * 2.0) - 1.0

        self.encoder.to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            latent = self.encoder(video)
        self.encoder.to("cpu")

        state = self.video_tools.create_initial_state(
            device=self.device,
            dtype=self.dtype,
            initial_latent=latent.to(dtype=self.dtype),
        )
        return state.latent.detach()

    @torch.no_grad()
    def decode(self, latents: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Denoise patchified diffusion latents with the frozen transformer and decode to video."""
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)

        clean_state = self._build_clean_state(latents)
        noiser = GaussianNoiser(generator=generator or self._make_generator())
        noisy_state = noiser(latent_state=clean_state, noise_scale=self.generation_config.noise_scale)
        denoised_state = self._run_denoising(noisy_state, clean_state)
        return self._decode_video_latent(denoised_state.latent)

    @torch.no_grad()
    def denoise_latents(self, latents: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Run diffusion denoising only and return denoised patchified latents."""
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)

        clean_state = self._build_clean_state(latents)
        noiser = GaussianNoiser(generator=generator or self._make_generator())
        noisy_state = noiser(latent_state=clean_state, noise_scale=self.generation_config.noise_scale)
        denoised_state = self._run_denoising(noisy_state, clean_state)
        return denoised_state.latent.detach()

    @torch.no_grad()
    def decode_from_denoised_latents(self, denoised_latents: Tensor) -> Tensor:
        """Decode already-denoised patchified latents directly to video."""
        if denoised_latents.dim() == 2:
            denoised_latents = denoised_latents.unsqueeze(0)
        return self._decode_video_latent(denoised_latents)

    def _build_clean_state(self, latents: Tensor) -> LatentState:
        state = self.video_tools.create_initial_state(device=self.device, dtype=self.dtype)
        latent = latents.to(self.device, dtype=self.dtype)
        return replace(state, latent=latent, clean_latent=latent.clone())

    def _make_generator(self) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(self.generation_config.seed)

    def _get_prompt_embeddings(self) -> tuple[Tensor, Tensor | None]:
        if self._cached_prompt_embeddings is not None:
            pos, neg = self._cached_prompt_embeddings
            return pos.to(self.device), neg.to(self.device) if neg is not None else None

        self.text_encoder.to(self.device)
        self.embeddings_processor.to(self.device)

        pos_hs, pos_mask = self.text_encoder.encode(self.generation_config.prompt)
        pos_out = self.embeddings_processor.process_hidden_states(pos_hs, pos_mask)
        pos = pos_out.video_encoding.detach().cpu()

        neg = None
        if self.generation_config.guidance_scale != 1.0:
            neg_hs, neg_mask = self.text_encoder.encode(self.generation_config.negative_prompt)
            neg_out = self.embeddings_processor.process_hidden_states(neg_hs, neg_mask)
            neg = neg_out.video_encoding.detach().cpu()

        self.text_encoder.model.to("cpu")
        self.embeddings_processor.to("cpu")

        self._cached_prompt_embeddings = (pos, neg)
        return pos.to(self.device), neg.to(self.device) if neg is not None else None

    def _run_denoising(self, video_state: LatentState, clean_state: LatentState) -> LatentState:
        sigmas = LTX2Scheduler().execute(steps=self.generation_config.num_inference_steps).to(self.device).float()
        stepper = EulerDiffusionStep()
        cfg_guider = CFGGuider(self.generation_config.guidance_scale)
        stg_guider = STGGuider(self.generation_config.stg_scale)
        stg_perturbations = self._build_stg_perturbation_config() if stg_guider.enabled() else None
        context_pos, context_neg = self._get_prompt_embeddings()

        video = Modality(
            enabled=True,
            latent=video_state.latent,
            sigma=sigmas[0].repeat(video_state.latent.shape[0]),
            timesteps=video_state.denoise_mask,
            positions=video_state.positions,
            context=context_pos,
            context_mask=None,
        )

        self.transformer.to(self.device)
        x0_model = X0Model(self.transformer)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            for step_idx, sigma in enumerate(sigmas[:-1]):
                video = replace(
                    video,
                    latent=video_state.latent,
                    sigma=sigma.repeat(video_state.latent.shape[0]),
                    timesteps=sigma * video_state.denoise_mask,
                    positions=video_state.positions,
                )
                pos_video, _ = x0_model(video=video, audio=None, perturbations=None)
                denoised_video = pos_video

                if cfg_guider.enabled() and context_neg is not None:
                    neg_video, _ = x0_model(video=replace(video, context=context_neg), audio=None, perturbations=None)
                    denoised_video = denoised_video + cfg_guider.delta(pos_video, neg_video)

                if stg_guider.enabled() and stg_perturbations is not None:
                    perturbed_video, _ = x0_model(video=video, audio=None, perturbations=stg_perturbations)
                    denoised_video = denoised_video + stg_guider.delta(pos_video, perturbed_video)

                denoised_video = denoised_video * video_state.denoise_mask + clean_state.latent.float() * (
                    1 - video_state.denoise_mask
                )
                video_state = replace(
                    video_state,
                    latent=stepper.step(
                        sample=video.latent,
                        denoised_sample=denoised_video,
                        sigmas=sigmas,
                        step_index=step_idx,
                    ),
                )

        self.transformer.to("cpu")
        return video_state

    def _decode_video_latent(self, latent: Tensor) -> Tensor:
        latent_frames = self.num_frames // VIDEO_SCALE_FACTORS.time + 1
        latent_height = self.height // VIDEO_SCALE_FACTORS.height
        latent_width = self.width // VIDEO_SCALE_FACTORS.width
        unpatchified = self.patchifier.unpatchify(
            latent,
            output_shape=VideoLatentShape(
                batch=1,
                channels=128,
                frames=latent_frames,
                height=latent_height,
                width=latent_width,
            ),
        ).to(dtype=self.dtype, device=self.device)

        self.decoder.to(self.device)
        tiled_config = self.generation_config.resolved_tiled_decoding()
        if tiled_config.enabled:
            chunks = []
            for video_chunk in self.decoder.tiled_decode(
                unpatchified,
                tiling_config=TilingConfig(
                    spatial_config=SpatialTilingConfig(
                        tile_size_in_pixels=tiled_config.tile_size_pixels,
                        tile_overlap_in_pixels=tiled_config.tile_overlap_pixels,
                    ),
                    temporal_config=TemporalTilingConfig(
                        tile_size_in_frames=tiled_config.tile_size_frames,
                        tile_overlap_in_frames=tiled_config.tile_overlap_frames,
                    ),
                ),
            ):
                chunks.append(video_chunk)
            decoded = torch.cat(chunks, dim=2)
        else:
            decoded = self.decoder(unpatchified)
        self.decoder.to("cpu")
        decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        return decoded.detach()

    def _validate_video_shape(self, video: Tensor) -> None:
        _, _, frames, height, width = video.shape
        if height != self.height or width != self.width:
            raise ValueError(
                f"Video shape {height}x{width} does not match configured diffusion size {self.height}x{self.width}"
            )
        if frames != self.num_frames:
            raise ValueError(f"Video has {frames} frames, expected {self.num_frames}")

    def _build_stg_perturbation_config(self) -> BatchedPerturbationConfig:
        perturbations = [
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=self.generation_config.stg_blocks)
        ]
        if self.generation_config.stg_mode == "stg_av":
            perturbations.append(
                Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=self.generation_config.stg_blocks)
            )
        return BatchedPerturbationConfig([PerturbationConfig(perturbations=perturbations)])