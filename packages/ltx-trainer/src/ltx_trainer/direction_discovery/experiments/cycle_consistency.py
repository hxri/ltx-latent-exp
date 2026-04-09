"""Cycle-consistency experiment for diffusion-backed VAE encode/decode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console

from ltx_trainer.direction_discovery.models import DiffusionGenerationConfig, DiffusionInterface
from ltx_trainer.direction_discovery.utils.video_io import load_video, save_video
from ltx_trainer.model_loader import load_embeddings_processor, load_model

console = Console()


def _prepare_video_for_diffusion(video: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    """Trim/crop video to LTX-compatible dimensions."""
    notes: list[str] = []
    _, frames, height, width = video.shape

    valid_frames = ((frames - 1) // 8) * 8 + 1
    if valid_frames != frames:
        video = video[:, :valid_frames]
        notes.append(f"trimmed frames from {frames} to {valid_frames}")

    valid_height = (height // 32) * 32
    valid_width = (width // 32) * 32
    if valid_height != height or valid_width != width:
        top = max((height - valid_height) // 2, 0)
        left = max((width - valid_width) // 2, 0)
        video = video[:, :, top : top + valid_height, left : left + valid_width]
        notes.append(f"center-cropped from {height}x{width} to {valid_height}x{valid_width}")

    if video.shape[1] < 9:
        raise ValueError("Diffusion-backed cycle consistency needs at least 9 frames (k*8 + 1)")

    return video, notes


def _video_metrics(xa: torch.Tensor, xb: torch.Tensor) -> dict[str, float]:
    """Compute simple scalar metrics between two videos in [0, 1]."""
    xa_f = xa.detach().float().cpu()
    xb_f = xb.detach().float().cpu()
    diff = xa_f - xb_f
    mse = float(torch.mean(diff * diff).item())
    mae = float(torch.mean(torch.abs(diff)).item())
    l2 = float(torch.norm(diff).item())
    psnr = 100.0 if mse == 0.0 else float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())
    return {
        "mse": mse,
        "mae": mae,
        "l2": l2,
        "psnr_db": psnr,
    }


def run_cycle_consistency(
    video_path: str,
    checkpoint_path: str,
    text_encoder_path: str,
    output_dir: Optional[str] = None,
    prompt: str = "cinematic shot of the same scene",
    negative_prompt: str = "",
    device: str = "cuda",
    num_inference_steps: int = 30,
    guidance_scale: float = 4.0,
    stg_scale: float = 0.0,
    max_frames: int = 64,
    seed: int = 42,
) -> None:
    """
    Run cycle consistency:

    1) video -> encoder -> z0 -> diffusion decode -> x1
    2) x1 -> encoder -> z1 -> diffusion decode -> x2

    Saves outputs and metrics to output_dir.
    """
    video_path = Path(video_path)
    checkpoint_path = Path(checkpoint_path)
    text_encoder_path = Path(text_encoder_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not text_encoder_path.exists():
        raise FileNotFoundError(f"Text encoder path not found: {text_encoder_path}")

    if output_dir is None:
        output_dir = Path("outputs/cycle_consistency")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]Cycle Consistency Experiment[/bold cyan]")
    console.print(f"Video: {video_path}")
    console.print(f"Output: {output_dir}")
    console.print(f"Steps: {num_inference_steps}, CFG: {guidance_scale}, STG: {stg_scale}")
    console.print()

    with console.status("[bold]Loading video...[/bold]"):
        video, fps = load_video(str(video_path), max_frames=max_frames)
        video, prep_notes = _prepare_video_for_diffusion(video)
        for note in prep_notes:
            console.print(f"[yellow]![/yellow] Diffusion preprocessing: {note}")
        console.print(f"[green]✓[/green] Loaded video: {video.shape}, {fps:.1f} FPS")

    torch_device = torch.device(device)

    with console.status("[bold]Loading models...[/bold]"):
        components = load_model(
            checkpoint_path=str(checkpoint_path),
            text_encoder_path=str(text_encoder_path),
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
            height=video.shape[2],
            width=video.shape[3],
            num_frames=video.shape[1],
            frame_rate=fps,
            generation_config=DiffusionGenerationConfig(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                stg_scale=stg_scale,
            ),
            device=torch_device,
            dtype=torch.bfloat16,
        )
        console.print("[green]✓[/green] Diffusion + VAE backend loaded")

    with torch.inference_mode():
        video_batch = video.unsqueeze(0)

        console.print("[bold]Running cycle...[/bold]")
        z0 = vae.encode(video_batch)
        x1 = vae.decode(z0)
        z1 = vae.encode(x1)
        x2 = vae.decode(z1)

    console.print(f"[green]✓[/green] z0 shape: {tuple(z0.shape)}")
    console.print(f"[green]✓[/green] z1 shape: {tuple(z1.shape)}")

    latent_l2 = float(torch.norm((z0 - z1).float()).item())
    latent_mae = float(torch.mean(torch.abs((z0 - z1).float())).item())

    metrics = {
        "config": {
            "video_path": str(video_path),
            "checkpoint_path": str(checkpoint_path),
            "text_encoder_path": str(text_encoder_path),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "device": device,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "stg_scale": stg_scale,
            "max_frames": max_frames,
            "seed": seed,
            "fps": fps,
            "video_shape": list(video.shape),
        },
        "latent_metrics": {
            "z0_vs_z1_l2": latent_l2,
            "z0_vs_z1_mae": latent_mae,
        },
        "video_metrics": {
            "ref_vs_x1": _video_metrics(video_batch, x1),
            "x1_vs_x2": _video_metrics(x1, x2),
            "ref_vs_x2": _video_metrics(video_batch, x2),
        },
    }

    console.print("[bold]Saving outputs...[/bold]")
    save_video(video, output_dir / "reference_preprocessed.mp4", fps=fps)
    save_video(x1, output_dir / "decoded_pass1.mp4", fps=fps)
    save_video(x2, output_dir / "decoded_pass2.mp4", fps=fps)
    torch.save(z0.detach().cpu(), output_dir / "latent_z0.pt")
    torch.save(z1.detach().cpu(), output_dir / "latent_z1.pt")

    metrics_path = output_dir / "cycle_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    console.print(f"[green]✓[/green] Saved videos and latents to: {output_dir}")
    console.print(f"[green]✓[/green] Saved metrics: {metrics_path}")
    console.print()
    console.print("[bold cyan]Summary[/bold cyan]")
    console.print(f"Latent L2 (z0 vs z1): {latent_l2:.6f}")
    console.print(f"Video L2 (x1 vs x2): {metrics['video_metrics']['x1_vs_x2']['l2']:.6f}")
    console.print(f"Video MAE (x1 vs x2): {metrics['video_metrics']['x1_vs_x2']['mae']:.6f}")


if __name__ == "__main__":
    typer.run(run_cycle_consistency)
