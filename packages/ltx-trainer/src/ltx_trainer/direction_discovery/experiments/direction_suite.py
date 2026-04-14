"""Comprehensive direction suite experiment with comparison logging."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Optional

import torch
import typer
from rich.console import Console

from ltx_trainer.direction_discovery.experiments.run_discovery import (
    _match_video_shape,
    _prepare_video_for_diffusion,
    _to_grayscale_video,
    _zoom_in_video,
)
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


def _adjust_brightness_video(video: torch.Tensor, factor: float) -> torch.Tensor:
    return (video * factor).clamp(0.0, 1.0)


def _adjust_contrast_video(video: torch.Tensor, factor: float) -> torch.Tensor:
    # Contrast around per-frame RGB mean to avoid large global shifts.
    frame_mean = video.mean(dim=(0, 2, 3), keepdim=True)
    return (frame_mean + (video - frame_mean) * factor).clamp(0.0, 1.0)


def _video_metrics(x0: torch.Tensor, x1: torch.Tensor) -> dict[str, float]:
    d = (x0.float() - x1.float()).cpu()
    mse = float(torch.mean(d * d).item())
    mae = float(torch.mean(torch.abs(d)).item())
    l2 = float(torch.norm(d).item())
    return {"mse": mse, "mae": mae, "l2": l2}


def _sanitize_alpha(alpha: float) -> str:
    s = f"{alpha:+.3f}".replace("+", "p").replace("-", "m")
    return s.replace(".", "_")


def run_direction_suite(
    source_video_path: str,
    transfer_video_paths: list[str],
    checkpoint_path: str = typer.Option(..., "--checkpoint-path"),
    text_encoder_path: Optional[str] = typer.Option(None, "--text-encoder-path"),
    output_dir: Optional[str] = None,
    prompt: str = "",
    negative_prompt: str = "",
    alphas: str = "-1.0,-0.5,0.5,1.0",
    direction_target_rms: float = 0.05,
    seed: int = 123,
    device: str = "cuda",
    use_diffusion: bool = True,
    edit_after_diffusion: bool = True,
    diffusion_noise_scale: float = 0.1,
    guidance_scale: float = 1.0,
    num_inference_steps: int = 10,
    brightness_factor: float = 0.6,
    zoom_scale: float = 1.1,
    contrast_factor: float = 1.25,
    save_diff_maps: bool = True,
    diff_map_gain: float = 20.0,
) -> None:
    """Build 5 direction varieties and compare transfer behavior across videos and alphas."""
    if len(transfer_video_paths) == 0:
        raise ValueError("Provide at least one transfer video via positional TRANSFER_VIDEO_PATHS")

    if not (0.0 <= diffusion_noise_scale <= 1.0):
        raise ValueError(f"diffusion_noise_scale must be in [0,1], got {diffusion_noise_scale}")
    if brightness_factor <= 0.0 or brightness_factor > 1.0:
        raise ValueError(f"brightness_factor must be in (0,1], got {brightness_factor}")
    if zoom_scale <= 1.0:
        raise ValueError(f"zoom_scale must be > 1.0, got {zoom_scale}")
    if contrast_factor <= 0.0:
        raise ValueError(f"contrast_factor must be > 0, got {contrast_factor}")

    alpha_list = [float(a.strip()) for a in alphas.split(",") if a.strip()]
    if len(alpha_list) == 0:
        raise ValueError("alphas produced an empty list")

    source_video_path = Path(source_video_path)
    checkpoint_path = Path(checkpoint_path)
    if text_encoder_path is not None:
        text_encoder_path = str(Path(text_encoder_path))

    if output_dir is None:
        output_dir = Path("outputs/direction_suite")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_video_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_video_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    console.print("[bold cyan]Direction Suite Experiment[/bold cyan]")
    console.print(f"Source: {source_video_path}")
    console.print(f"Transfer videos: {len(transfer_video_paths)}")
    console.print(f"Alphas: {alpha_list}")
    console.print(f"Output: {output_dir}")

    source_video, source_fps = load_video(str(source_video_path), max_frames=64)
    if use_diffusion:
        source_video, notes = _prepare_video_for_diffusion(source_video)
        for note in notes:
            console.print(f"[yellow]![/yellow] Source preprocessing: {note}")

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
                height=source_video.shape[2],
                width=source_video.shape[3],
                num_frames=source_video.shape[1],
                frame_rate=source_fps,
                generation_config=DiffusionGenerationConfig(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    noise_scale=diffusion_noise_scale,
                ),
                device=torch_device,
                dtype=torch.bfloat16,
            )
        else:
            encoder = load_video_vae_encoder(str(checkpoint_path), device=torch_device, dtype=torch.bfloat16)
            decoder = load_video_vae_decoder(str(checkpoint_path), device=torch_device, dtype=torch.bfloat16)
            vae = VAEInterface(encoder, decoder, device=torch_device, vae_tiling=False)

    with torch.inference_mode():
        source_latents = vae.encode(source_video.unsqueeze(0))

        # Build 5 direction varieties.
        torch.manual_seed(seed)
        random_direction = torch.randn_like(source_latents.squeeze(0))

        gray_video = _to_grayscale_video(source_video)
        gray_latents = vae.encode(gray_video.unsqueeze(0))
        gray_direction = (gray_latents - source_latents).squeeze(0)

        darker_video = _adjust_brightness_video(source_video, brightness_factor)
        darker_latents = vae.encode(darker_video.unsqueeze(0))
        brightness_direction = (darker_latents - source_latents).squeeze(0)

        zoom_video = _zoom_in_video(source_video, zoom_scale)
        zoom_latents = vae.encode(zoom_video.unsqueeze(0))
        zoom_direction = (zoom_latents - source_latents).squeeze(0)

        contrast_video = _adjust_contrast_video(source_video, contrast_factor)
        contrast_latents = vae.encode(contrast_video.unsqueeze(0))
        contrast_direction = (contrast_latents - source_latents).squeeze(0)

    directions: dict[str, torch.Tensor] = {
        "random": random_direction,
        "grayscale": gray_direction,
        "brightness": brightness_direction,
        "zoom": zoom_direction,
        "contrast": contrast_direction,
    }

    refs_dir = output_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)
    save_video(source_video, refs_dir / "source_preprocessed.mp4", fps=source_fps)
    save_video(gray_video, refs_dir / "reference_grayscale.mp4", fps=source_fps)
    save_video(darker_video, refs_dir / "reference_brightness_reduced.mp4", fps=source_fps)
    save_video(zoom_video, refs_dir / "reference_zoomed.mp4", fps=source_fps)
    save_video(contrast_video, refs_dir / "reference_contrast_adjusted.mp4", fps=source_fps)

    directions_dir = output_dir / "directions"
    directions_dir.mkdir(parents=True, exist_ok=True)
    for name, direction in directions.items():
        torch.save(direction.detach().cpu(), directions_dir / f"{name}.pt")

    results: list[dict] = []
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

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

            video = _match_video_shape(
                video,
                target_frames=source_video.shape[1],
                target_height=source_video.shape[2],
                target_width=source_video.shape[3],
            )

            latents = vae.encode(video.unsqueeze(0))
            post_mode = (
                edit_after_diffusion
                and hasattr(vae, "denoise_latents")
                and hasattr(vae, "decode_from_denoised_latents")
            )
            if post_mode:
                base_latents = vae.denoise_latents(latents)
                base_video = vae.decode_from_denoised_latents(base_latents)
            else:
                base_latents = latents
                base_video = vae.decode(latents)

        stem = vpath.stem
        save_video(video, videos_dir / f"{stem}_input_preprocessed.mp4", fps=fps)
        save_video(base_video, videos_dir / f"{stem}_base.mp4", fps=fps)

        for direction_name, direction in directions.items():
            for alpha in alpha_list:
                edited_latents = LatentController.apply_direction(
                    base_latents,
                    direction.to(base_latents.device, dtype=base_latents.dtype),
                    alpha,
                    target_rms=direction_target_rms,
                )
                if post_mode:
                    edited = vae.decode_from_denoised_latents(edited_latents)
                else:
                    edited = vae.decode(edited_latents)

                alpha_tag = _sanitize_alpha(alpha)
                edited_path = videos_dir / f"{stem}_{direction_name}_a{alpha_tag}.mp4"
                save_video(edited, edited_path, fps=fps)

                diff_path = None
                if save_diff_maps:
                    diff = (edited.float() - base_video.float()).abs() * diff_map_gain
                    diff = diff.clamp(0.0, 1.0)
                    diff_path = videos_dir / f"{stem}_{direction_name}_a{alpha_tag}_diff_x{diff_map_gain:g}.mp4"
                    save_video(diff, diff_path, fps=fps)

                metrics = _video_metrics(base_video, edited)
                results.append(
                    {
                        "video": str(vpath),
                        "direction": direction_name,
                        "alpha": alpha,
                        "edited_video": str(edited_path),
                        "diff_map": str(diff_path) if diff_path is not None else None,
                        "metrics_base_vs_edited": metrics,
                    }
                )

    # Aggregate comparison by direction.
    by_direction: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_direction[row["direction"]].append(row["metrics_base_vs_edited"])

    comparison = []
    for direction_name, metric_rows in by_direction.items():
        comparison.append(
            {
                "direction": direction_name,
                "mean_mae": mean(r["mae"] for r in metric_rows),
                "mean_mse": mean(r["mse"] for r in metric_rows),
                "mean_l2": mean(r["l2"] for r in metric_rows),
                "samples": len(metric_rows),
            }
        )
    comparison = sorted(comparison, key=lambda x: x["mean_mae"], reverse=True)

    summary = {
        "config": {
            "source_video_path": str(source_video_path),
            "transfer_video_paths": transfer_video_paths,
            "alphas": alpha_list,
            "direction_target_rms": direction_target_rms,
            "seed": seed,
            "use_diffusion": use_diffusion,
            "edit_after_diffusion": edit_after_diffusion,
            "diffusion_noise_scale": diffusion_noise_scale,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "brightness_factor": brightness_factor,
            "zoom_scale": zoom_scale,
            "contrast_factor": contrast_factor,
            "save_diff_maps": save_diff_maps,
            "diff_map_gain": diff_map_gain,
        },
        "direction_files": {name: str(directions_dir / f"{name}.pt") for name in directions.keys()},
        "comparison_by_direction": comparison,
        "results": results,
    }

    summary_path = output_dir / "suite_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print()
    console.print(f"[bold green]✓ Direction suite complete[/bold green]: {output_dir}")
    console.print(f"Summary: {summary_path}")
    console.print("Top directions by mean MAE:")
    for idx, row in enumerate(comparison[:5], start=1):
        console.print(
            f"  {idx}. {row['direction']}: mean_mae={row['mean_mae']:.6f}, "
            f"mean_l2={row['mean_l2']:.4f} (n={row['samples']})"
        )


if __name__ == "__main__":
    typer.run(run_direction_suite)
