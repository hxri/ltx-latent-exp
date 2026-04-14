"""Transfer-only workflow using a previously saved latent direction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console

from ltx_trainer.direction_discovery.experiments.run_discovery import _match_video_shape, _prepare_video_for_diffusion
from ltx_trainer.direction_discovery.latent.controller import LatentController
from ltx_trainer.direction_discovery.models import DiffusionGenerationConfig, DiffusionInterface, VAEInterface
from ltx_trainer.direction_discovery.utils.video_io import load_video, save_video
from ltx_trainer.model_loader import (
    load_embeddings_processor,
    load_model,
    load_video_vae_decoder,
    load_video_vae_encoder,
)

console = Console()


def _video_change_metrics(x0: torch.Tensor, x1: torch.Tensor) -> dict[str, float]:
    d = (x0.float() - x1.float()).cpu()
    mse = float(torch.mean(d * d).item())
    mae = float(torch.mean(torch.abs(d)).item())
    l2 = float(torch.norm(d).item())
    return {"mse": mse, "mae": mae, "l2": l2}


def run_transfer_saved_direction(
    direction_path: str,
    reference_video_path: str,
    transfer_video_paths: list[str],
    checkpoint_path: str = typer.Option(..., "--checkpoint-path"),
    text_encoder_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    prompt: str = "cinematic shot of the same scene",
    negative_prompt: str = "",
    alpha: float = -1.0,
    direction_target_rms: float = 0.1,
    device: str = "cuda",
    use_diffusion: bool = True,
    edit_after_diffusion: bool = True,
    num_inference_steps: int = 30,
    guidance_scale: float = 4.0,
    stg_scale: float = 0.0,
    diffusion_noise_scale: float = 0.2,
    seed: int = 123,
    save_difference_map: bool = True,
    difference_map_gain: float = 20.0,
) -> None:
    """Apply a saved direction tensor to multiple videos without rediscovering directions."""
    if len(transfer_video_paths) == 0:
        raise ValueError("Provide at least one transfer video path")

    direction_path = Path(direction_path)
    reference_video_path = Path(reference_video_path)
    checkpoint_path = Path(checkpoint_path)
    if text_encoder_path is not None:
        text_encoder_path = str(Path(text_encoder_path))

    if output_dir is None:
        output_dir = Path("outputs/direction_transfer")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not direction_path.exists():
        raise FileNotFoundError(f"Direction file not found: {direction_path}")
    if not reference_video_path.exists():
        raise FileNotFoundError(f"Reference video not found: {reference_video_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    direction = torch.load(direction_path, map_location="cpu")
    if not isinstance(direction, torch.Tensor):
        raise ValueError(f"Direction file does not contain a tensor: {direction_path}")

    console.print("[bold cyan]Saved Direction Transfer[/bold cyan]")
    console.print(f"Direction: {direction_path}")
    console.print(f"Reference video: {reference_video_path}")
    console.print(f"Transfer videos: {len(transfer_video_paths)}")
    console.print(f"Output: {output_dir}")
    console.print()

    # Prepare reference shape so transfer videos match token geometry of discovered direction.
    reference_video, _ = load_video(str(reference_video_path), max_frames=64)
    if use_diffusion:
        reference_video, _ = _prepare_video_for_diffusion(reference_video)
    ref_frames = reference_video.shape[1]
    ref_height = reference_video.shape[2]
    ref_width = reference_video.shape[3]

    torch_device = torch.device(device)

    with console.status("[bold]Loading models...[/bold]"):
        if use_diffusion:
            if text_encoder_path is None:
                raise ValueError("text_encoder_path is required when use_diffusion=True")

            components = load_model(
                checkpoint_path=str(checkpoint_path),
                text_encoder_path=text_encoder_path,
                device="cpu",
                dtype=torch.bfloat16,
                with_video_vae_encoder=True,
                with_video_vae_decoder=True,
                with_audio_vae_decoder=False,
                with_vocoder=False,
                with_text_encoder=True,
            )
            embeddings_processor = load_embeddings_processor(str(checkpoint_path), device="cpu", dtype=torch.bfloat16)
            vae = DiffusionInterface(
                transformer=components.transformer,
                encoder=components.video_vae_encoder,
                decoder=components.video_vae_decoder,
                text_encoder=components.text_encoder,
                embeddings_processor=embeddings_processor,
                height=ref_height,
                width=ref_width,
                num_frames=ref_frames,
                frame_rate=24.0,
                generation_config=DiffusionGenerationConfig(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    noise_scale=diffusion_noise_scale,
                    stg_scale=stg_scale,
                ),
                device=torch_device,
                dtype=torch.bfloat16,
            )
        else:
            encoder = load_video_vae_encoder(str(checkpoint_path), device=torch_device, dtype=torch.bfloat16)
            decoder = load_video_vae_decoder(str(checkpoint_path), device=torch_device, dtype=torch.bfloat16)
            vae = VAEInterface(encoder, decoder, device=torch_device, vae_tiling=False)

    summary: list[dict] = []

    for transfer_video_path in transfer_video_paths:
        vpath = Path(transfer_video_path)
        if not vpath.exists():
            console.print(f"[red]Skipping missing video:[/red] {vpath}")
            continue

        with console.status(f"[bold]Processing {vpath.name}...[/bold]"):
            video, fps = load_video(str(vpath), max_frames=64)
            if use_diffusion:
                video, notes = _prepare_video_for_diffusion(video)
                for note in notes:
                    console.print(f"[yellow]![/yellow] {vpath.name}: {note}")

            video = _match_video_shape(video, ref_frames, ref_height, ref_width)
            latents = vae.encode(video.unsqueeze(0))

            post_diffusion_mode = (
                edit_after_diffusion
                and hasattr(vae, "denoise_latents")
                and hasattr(vae, "decode_from_denoised_latents")
            )
            if post_diffusion_mode:
                base_latents = vae.denoise_latents(latents)
                edited_latents = LatentController.apply_direction(
                    base_latents,
                    direction.to(base_latents.device, dtype=base_latents.dtype),
                    alpha,
                    target_rms=direction_target_rms,
                )
                edited = vae.decode_from_denoised_latents(edited_latents)
                original_decoded = vae.decode_from_denoised_latents(base_latents)
            else:
                edited_latents = LatentController.apply_direction(
                    latents,
                    direction.to(latents.device, dtype=latents.dtype),
                    alpha,
                    target_rms=direction_target_rms,
                )
                edited = vae.decode(edited_latents)
                original_decoded = vae.decode(latents)

        stem = vpath.stem
        save_video(video, output_dir / f"{stem}_input_preprocessed.mp4", fps=fps)
        save_video(original_decoded, output_dir / f"{stem}_decoded_base.mp4", fps=fps)
        save_video(edited, output_dir / f"{stem}_edited.mp4", fps=fps)

        diff_map_path = None
        if save_difference_map:
            diff_map = (edited.float() - original_decoded.float()).abs() * difference_map_gain
            diff_map = diff_map.clamp(0.0, 1.0)
            diff_map_path = output_dir / f"{stem}_diff_map_x{difference_map_gain:g}.mp4"
            save_video(diff_map, diff_map_path, fps=fps)

        base_metrics = _video_change_metrics(original_decoded, edited)
        summary.append(
            {
                "video": str(vpath),
                "alpha": alpha,
                "direction_target_rms": direction_target_rms,
                "metrics_base_vs_edited": base_metrics,
                "difference_map": {
                    "enabled": save_difference_map,
                    "gain": difference_map_gain,
                    "path": str(diff_map_path) if diff_map_path is not None else None,
                },
            }
        )
        console.print(f"[green]✓[/green] Saved transfer outputs for {vpath.name}")

    summary_path = output_dir / "transfer_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "config": {
                    "direction_path": str(direction_path),
                    "reference_video_path": str(reference_video_path),
                    "alpha": alpha,
                    "direction_target_rms": direction_target_rms,
                    "use_diffusion": use_diffusion,
                    "edit_after_diffusion": edit_after_diffusion,
                    "seed": seed,
                    "diffusion_noise_scale": diffusion_noise_scale,
                    "save_difference_map": save_difference_map,
                    "difference_map_gain": difference_map_gain,
                },
                "results": summary,
            },
            f,
            indent=2,
        )

    console.print()
    console.print(f"[bold green]✓ Transfer complete[/bold green]: {output_dir}")
    console.print(f"Summary: {summary_path}")


if __name__ == "__main__":
    typer.run(run_transfer_saved_direction)
