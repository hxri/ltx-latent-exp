"""Main experiment script for direction discovery."""

import json
from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.progress import track
from rich.table import Table

from ltx_trainer.direction_discovery.config import DirectionDiscoveryConfig
from ltx_trainer.direction_discovery.evaluation.metrics import create_metric
from ltx_trainer.direction_discovery.evaluation.stability import StabilityEvaluator
from ltx_trainer.direction_discovery.latent.controller import LatentController
from ltx_trainer.direction_discovery.latent.directions import create_direction_generator
from ltx_trainer.direction_discovery.models import DiffusionGenerationConfig, DiffusionInterface, VAEInterface
from ltx_trainer.direction_discovery.utils.video_io import load_video, save_video
from ltx_trainer.direction_discovery.utils.visualization import (
    create_direction_grid,
    create_metric_curve,
    create_reversibility_plot,
)
from ltx_trainer.model_loader import (
    load_embeddings_processor,
    load_model,
    load_video_vae_decoder,
    load_video_vae_encoder,
)

console = Console()


def _prepare_video_for_diffusion(video: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    """Trim/crop video to LTX-compatible dimensions without changing semantics more than needed."""
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
        raise ValueError("Diffusion-backed direction discovery needs at least 9 frames (k*8 + 1)")

    return video, notes


def _to_grayscale_video(video: torch.Tensor) -> torch.Tensor:
    """Convert RGB video [C, F, H, W] to 3-channel grayscale video."""
    if video.shape[0] == 1:
        return video.repeat(3, 1, 1, 1)
    if video.shape[0] < 3:
        raise ValueError(f"Expected at least 3 channels for RGB video, got shape {tuple(video.shape)}")

    # Standard luminance transform in linear RGB-like tensor space.
    gray = 0.299 * video[0] + 0.587 * video[1] + 0.114 * video[2]
    gray3 = gray.unsqueeze(0).repeat(3, 1, 1, 1)
    return gray3.clamp(0.0, 1.0)


def _adjust_brightness_video(video: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale video brightness by factor for [C, F, H, W] input in [0, 1]."""
    return (video * factor).clamp(0.0, 1.0)


def _zoom_in_video(video: torch.Tensor, zoom_scale: float) -> torch.Tensor:
    """Create a zoomed-in version of [C, F, H, W] while preserving output shape."""
    import torch.nn.functional as F

    if zoom_scale <= 1.0:
        raise ValueError(f"zoom_scale must be > 1.0 for zoom-in, got {zoom_scale}")

    c, f, h, w = video.shape
    up_h = max(int(round(h * zoom_scale)), h + 1)
    up_w = max(int(round(w * zoom_scale)), w + 1)

    video_2d = video.permute(1, 0, 2, 3)  # [F, C, H, W]
    zoomed = F.interpolate(video_2d, size=(up_h, up_w), mode="bilinear", align_corners=False)

    top = (up_h - h) // 2
    left = (up_w - w) // 2
    zoomed = zoomed[:, :, top : top + h, left : left + w]
    return zoomed.permute(1, 0, 2, 3).clamp(0.0, 1.0)


def _match_video_shape(video: torch.Tensor, target_frames: int, target_height: int, target_width: int) -> torch.Tensor:
    """Match video [C, F, H, W] to target shape via trim + optional spatial upsample + center crop."""
    import torch.nn.functional as F

    _, frames, height, width = video.shape

    if frames < target_frames:
        raise ValueError(f"Transfer video has {frames} frames, needs at least {target_frames}")
    if height < target_height or width < target_width:
        # Upsample spatially when transfer video is smaller than the required canvas.
        video_2d = video.permute(1, 0, 2, 3)  # [F, C, H, W]
        video_2d = F.interpolate(video_2d, size=(target_height, target_width), mode="bilinear", align_corners=False)
        video = video_2d.permute(1, 0, 2, 3)
        _, frames, height, width = video.shape

    video = video[:, :target_frames]

    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return video[:, :, top : top + target_height, left : left + target_width]


def run_direction_discovery(
    video_path: str,
    output_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    text_encoder_path: Optional[str] = None,
    prompt: str = "",
    negative_prompt: str = "",
    method: str = "random",
    num_directions: int = 5,
    metric: str = "l2",
    device: str = "cuda",
    vae_tiling: bool = False,
    use_diffusion: bool = True,
    num_inference_steps: int = 30,
    guidance_scale: float = 4.0,
    seed: int = 42,
    diffusion_noise_scale: float = 1.0,
    stg_scale: float = 0.0,
    edit_after_diffusion: bool = False,
    grayscale_difference_direction: bool = False,
    save_grayscale_reference: bool = False,
    brightness_difference_direction: bool = False,
    brightness_factor: float = 0.6,
    save_brightness_reference: bool = False,
    zoom_difference_direction: bool = False,
    zoom_scale: float = 1.1,
    save_zoom_reference: bool = False,
    transfer_video_path: Optional[str] = None,
    transfer_alpha: Optional[float] = None,
    single_alpha: Optional[float] = None,
    direction_target_rms: float = 0.05,
) -> None:
    """
    Run complete direction discovery experiment.

    Args:
        video_path: Path to input video
        output_dir: Output directory (default: ./outputs/direction_discovery)
        checkpoint_path: Path to LTX-2 checkpoint
        method: Direction generation method (random/difference/transformation)
        num_directions: Number of directions to generate
        metric: Distance metric (l2/ssim/lpips)
        device: Device to use
        vae_tiling: Enable VAE tiling
    """
    # Setup
    video_path = Path(video_path)
    if not video_path.exists():
        console.print(f"[red]Error: Video not found: {video_path}[/red]")
        return

    if output_dir is None:
        output_dir = Path("outputs/direction_discovery")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = DirectionDiscoveryConfig(
        output_dir=output_dir,
        device=device,
        vae_tiling=vae_tiling,
        checkpoint_path=Path(checkpoint_path) if checkpoint_path is not None else None,
        text_encoder_path=Path(text_encoder_path) if text_encoder_path is not None else None,
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_diffusion=use_diffusion,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        stg_scale=stg_scale,
    )
    config.direction_config.method = method
    config.direction_config.num_directions = num_directions
    config.direction_config.seed = seed
    config.evaluation_config.metric = metric
    if single_alpha is not None:
        config.evaluation_config.alphas = [single_alpha]

    if diffusion_noise_scale < 0.0 or diffusion_noise_scale > 1.0:
        raise ValueError(f"diffusion_noise_scale must be in [0, 1], got {diffusion_noise_scale}")
    if brightness_factor <= 0.0 or brightness_factor > 1.0:
        raise ValueError(f"brightness_factor must be in (0, 1], got {brightness_factor}")
    if zoom_scale <= 1.0:
        raise ValueError(f"zoom_scale must be > 1.0, got {zoom_scale}")

    enabled_sources = [
        grayscale_difference_direction,
        brightness_difference_direction,
        zoom_difference_direction,
    ]
    if sum(int(x) for x in enabled_sources) > 1:
        raise ValueError(
            "Use only one direction source: grayscale_difference_direction OR "
            "brightness_difference_direction OR zoom_difference_direction"
        )

    console.print("[bold cyan]Direction Discovery Experiment[/bold cyan]")
    console.print(f"Video: {video_path}")
    console.print(f"Output: {output_dir}")
    console.print(f"Method: {method}")
    console.print(f"Directions: {num_directions}")
    console.print(f"Alphas: {config.evaluation_config.alphas}")
    console.print(f"Direction target RMS: {direction_target_rms}")
    console.print(f"Seed: {seed}")
    console.print(f"Diffusion noise scale: {diffusion_noise_scale}")
    if use_diffusion and diffusion_noise_scale >= 0.95:
        console.print(
            "[yellow]![/yellow] High diffusion_noise_scale can overpower input latents and make outputs converge."
        )
    console.print(f"Edit stage: {'post-diffusion' if edit_after_diffusion else 'pre-diffusion'}")
    if grayscale_difference_direction:
        console.print("Difference direction source: original vs grayscale video latents")
    if brightness_difference_direction:
        console.print(
            f"Difference direction source: original vs brightness-reduced latents (factor={brightness_factor:.3f})"
        )
        console.print("[dim]Direction sign note: positive alpha will darken; negative alpha will brighten.[/dim]")
    if zoom_difference_direction:
        console.print(
            f"Difference direction source: original vs zoomed latents (zoom_scale={zoom_scale:.3f})"
        )
        console.print("[dim]Direction sign note: positive alpha will zoom in; negative alpha will zoom out.[/dim]")
    if transfer_video_path is not None:
        console.print(f"Transfer video: {transfer_video_path}")
    console.print(f"Backend: {'diffusion' if use_diffusion else 'vae-only'}")
    console.print()

    # Load video
    with console.status("[bold]Loading video...[/bold]"):
        video, fps = load_video(str(video_path), max_frames=64)  # Limit frames
        if use_diffusion:
            video, prep_notes = _prepare_video_for_diffusion(video)
            for note in prep_notes:
                console.print(f"[yellow]![/yellow] Diffusion preprocessing: {note}")
        console.print(f"[green]✓[/green] Loaded video: {video.shape}, {fps:.1f} FPS")

    torch_device = torch.device(device)

    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required")

    # Load backend
    with console.status("[bold]Loading models...[/bold]"):
        if use_diffusion:
            if text_encoder_path is None:
                raise ValueError("text_encoder_path is required when use_diffusion=True")

            components = load_model(
                checkpoint_path=checkpoint_path,
                text_encoder_path=text_encoder_path,
                device="cpu",
                dtype=torch.bfloat16,
                with_video_vae_encoder=True,
                with_video_vae_decoder=True,
                with_audio_vae_decoder=False,
                with_vocoder=False,
                with_text_encoder=True,
            )
            embeddings_processor = load_embeddings_processor(checkpoint_path, device="cpu", dtype=torch.bfloat16)
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
                    seed=config.direction_config.seed,
                    noise_scale=diffusion_noise_scale,
                    stg_scale=stg_scale,
                    stg_blocks=config.stg_blocks,
                    stg_mode=config.stg_mode,
                ),
                device=torch_device,
                dtype=torch.bfloat16,
            )
            console.print("[green]✓[/green] Diffusion + VAE backend loaded")
        else:
            encoder = load_video_vae_encoder(checkpoint_path, device=torch_device, dtype=torch.bfloat16)
            decoder = load_video_vae_decoder(checkpoint_path, device=torch_device, dtype=torch.bfloat16)
            vae = VAEInterface(encoder, decoder, device=torch_device, vae_tiling=vae_tiling)
            console.print("[green]✓[/green] VAE backend loaded")

    # Encode video to latent space
    with console.status("[bold]Encoding video to latent space...[/bold]"):
        video_batch = video.unsqueeze(0)
        latents = vae.encode(video_batch)
        console.print(f"[green]✓[/green] Latent shape: {latents.shape}")

    # Generate directions
    console.print()
    console.print("[bold]Generating directions...[/bold]")

    generator = create_direction_generator(method)
    if method == "transformation":
        directions = generator.generate(latents, vae=vae, num_directions=num_directions, seed=config.direction_config.seed)
    elif method == "difference" and grayscale_difference_direction:
        with console.status("[bold]Encoding grayscale reference for difference direction...[/bold]"):
            grayscale_video = _to_grayscale_video(video)
            if save_grayscale_reference:
                grayscale_path = output_dir / "grayscale_reference.mp4"
                save_video(grayscale_video, grayscale_path, fps=fps)
                console.print(f"[green]✓[/green] Saved grayscale reference video: {grayscale_path}")
            grayscale_latents = vae.encode(grayscale_video.unsqueeze(0))
            console.print(f"[green]✓[/green] Grayscale latent shape: {grayscale_latents.shape}")
        directions = generator.generate(
            latents,
            num_directions,
            seed=config.direction_config.seed,
            reference_latents=grayscale_latents,
        )
    elif method == "difference" and brightness_difference_direction:
        with console.status("[bold]Encoding brightness-reduced reference for difference direction...[/bold]"):
            darker_video = _adjust_brightness_video(video, brightness_factor)
            if save_brightness_reference:
                darker_path = output_dir / "brightness_reduced_reference.mp4"
                save_video(darker_video, darker_path, fps=fps)
                console.print(f"[green]✓[/green] Saved brightness-reduced reference video: {darker_path}")

            darker_latents = vae.encode(darker_video.unsqueeze(0))
            console.print(f"[green]✓[/green] Brightness-reduced latent shape: {darker_latents.shape}")

        # Explicit direction: v = z_dark - z_orig so positive alpha darkens.
        v = (darker_latents - latents).squeeze(0)
        directions = [v.clone() for _ in range(num_directions)]
    elif method == "difference" and zoom_difference_direction:
        with console.status("[bold]Encoding zoomed reference for difference direction...[/bold]"):
            zoomed_video = _zoom_in_video(video, zoom_scale)
            if save_zoom_reference:
                zoomed_path = output_dir / "zoom_reference.mp4"
                save_video(zoomed_video, zoomed_path, fps=fps)
                console.print(f"[green]✓[/green] Saved zoom reference video: {zoomed_path}")

            zoomed_latents = vae.encode(zoomed_video.unsqueeze(0))
            console.print(f"[green]✓[/green] Zoom latent shape: {zoomed_latents.shape}")

        # Explicit direction: v = z_zoom - z_orig so positive alpha tends to zoom in.
        v = (zoomed_latents - latents).squeeze(0)
        directions = [v.clone() for _ in range(num_directions)]
    else:
        directions = generator.generate(latents, num_directions, seed=config.direction_config.seed)
    console.print(f"[green]✓[/green] Generated {len(directions)} directions")

    # Setup evaluation
    metric_fn = create_metric(metric, device=torch_device)
    evaluator = StabilityEvaluator(
        vae,
        metric_fn,
        device=torch_device,
        edit_after_diffusion=edit_after_diffusion,
    )

    # Evaluate directions
    console.print()
    console.print("[bold]Evaluating directions...[/bold]")

    results = []
    total_steps = len(directions) * len(config.evaluation_config.alphas)
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Evaluating directions", total=total_steps)

        for dir_idx, direction in enumerate(directions):
            progress.update(task_id, description=f"Evaluating dir {dir_idx + 1}/{len(directions)}")

            def _on_alpha_done(alpha_idx: int, alpha_total: int, alpha: float) -> None:
                progress.advance(task_id, 1)
                progress.update(
                    task_id,
                    description=(
                        f"Evaluating dir {dir_idx + 1}/{len(directions)} "
                        f"alpha {alpha_idx}/{alpha_total} ({alpha:+.1f})"
                    ),
                )

            result = evaluator.evaluate_direction(
                latents,
                direction,
                config.evaluation_config.alphas,
                direction_id=dir_idx,
                target_rms=direction_target_rms,
                step_callback=_on_alpha_done,
            )
            results.append(result)
            console.print(
                "[dim]"
                f"dir {dir_idx}: strength={result.strength_score:.4f}, "
                f"reversibility={result.reversibility_score:.6f}, "
                f"quality={result.quality_score:.4f}"
                "[/dim]"
            )
    console.print(f"[green]✓[/green] Evaluated {len(results)} directions")

    # Save results
    console.print()
    console.print("[bold]Saving results...[/bold]")

    metrics_dict = {
        "config": {
            "method": method,
            "num_directions": num_directions,
            "metric": metric,
            "alphas": config.evaluation_config.alphas,
            "use_diffusion": use_diffusion,
            "edit_after_diffusion": edit_after_diffusion,
            "grayscale_difference_direction": grayscale_difference_direction,
            "brightness_difference_direction": brightness_difference_direction,
            "brightness_factor": brightness_factor,
            "zoom_difference_direction": zoom_difference_direction,
            "zoom_scale": zoom_scale,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "diffusion_noise_scale": diffusion_noise_scale,
        },
        "results": [r.to_dict() for r in results],
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    console.print(f"[green]✓[/green] Saved metrics: {metrics_path}")

    # Save directions for reuse/transfer experiments
    directions_dir = output_dir / "directions"
    directions_dir.mkdir(parents=True, exist_ok=True)
    for idx, direction in enumerate(directions):
        torch.save(direction.detach().cpu(), directions_dir / f"direction_{idx}.pt")
    console.print(f"[green]✓[/green] Saved direction tensors: {directions_dir}")

    # Generate visualizations
    console.print()
    console.print("[bold]Generating visualizations...[/bold]")

    # Stability plot
    viz_path = output_dir / "stability_plot.png"
    create_reversibility_plot(results, output_path=viz_path)
    console.print(f"[green]✓[/green] Saved stability plot: {viz_path}")

    # Generate and save example videos for top directions
    results_sorted = sorted(results, key=lambda r: r.quality_score, reverse=True)

    # Optional transfer test on a second video using the best discovered direction.
    if transfer_video_path is not None:
        transfer_path = Path(transfer_video_path)
        if not transfer_path.exists():
            raise FileNotFoundError(f"Transfer video not found: {transfer_path}")

        with console.status("[bold]Running transfer test on second video...[/bold]"):
            transfer_video, transfer_fps = load_video(str(transfer_path), max_frames=64)
            if use_diffusion:
                transfer_video, transfer_notes = _prepare_video_for_diffusion(transfer_video)
                for note in transfer_notes:
                    console.print(f"[yellow]![/yellow] Transfer preprocessing: {note}")

            transfer_video = _match_video_shape(
                transfer_video,
                target_frames=video.shape[1],
                target_height=video.shape[2],
                target_width=video.shape[3],
            )

            best_direction_idx = results_sorted[0].direction_id
            best_direction = directions[best_direction_idx]
            alpha_to_use = transfer_alpha if transfer_alpha is not None else config.evaluation_config.alphas[0]

            transfer_latents = vae.encode(transfer_video.unsqueeze(0))
            post_diffusion_mode = (
                edit_after_diffusion
                and hasattr(vae, "denoise_latents")
                and hasattr(vae, "decode_from_denoised_latents")
            )

            if post_diffusion_mode:
                transfer_base = vae.denoise_latents(transfer_latents)
                edited_latents = LatentController.apply_direction(
                    transfer_base,
                    best_direction,
                    alpha_to_use,
                    target_rms=direction_target_rms,
                )
                transfer_edited = vae.decode_from_denoised_latents(edited_latents)
            else:
                edited_latents = LatentController.apply_direction(
                    transfer_latents,
                    best_direction,
                    alpha_to_use,
                    target_rms=direction_target_rms,
                )
                transfer_edited = vae.decode(edited_latents)

            save_video(transfer_video, output_dir / "transfer_original_preprocessed.mp4", fps=transfer_fps)
            save_video(transfer_edited, output_dir / "transfer_edited.mp4", fps=transfer_fps)
            console.print(
                "[green]✓[/green] Saved transfer outputs: "
                f"{output_dir / 'transfer_original_preprocessed.mp4'} and {output_dir / 'transfer_edited.mp4'}"
            )

    for rank, result in enumerate(results_sorted[:3]):
        direction_idx = result.direction_id
        direction = directions[direction_idx]

        post_diffusion_mode = (
            edit_after_diffusion
            and hasattr(vae, "denoise_latents")
            and hasattr(vae, "decode_from_denoised_latents")
        )
        denoised_base = vae.denoise_latents(latents) if post_diffusion_mode else None

        # Generate videos at different alphas
        modified_videos = {}
        for alpha in config.evaluation_config.alphas:
            if post_diffusion_mode and denoised_base is not None:
                modified_latents = LatentController.apply_direction(
                    denoised_base,
                    direction,
                    alpha,
                    target_rms=direction_target_rms,
                )
                modified_video = vae.decode_from_denoised_latents(modified_latents)
            else:
                modified_latents = LatentController.apply_direction(
                    latents,
                    direction,
                    alpha,
                    target_rms=direction_target_rms,
                )
                modified_video = vae.decode(modified_latents)
            modified_videos[alpha] = modified_video.squeeze(0)  # Remove batch

        # Save original
        orig_video_path = output_dir / f"direction_{rank}_original.mp4"
        save_video(video, orig_video_path, fps=fps)

        # Save grid visualization
        grid_path = output_dir / f"direction_{rank}_grid.png"
        create_direction_grid(
            video,
            modified_videos,
            config.evaluation_config.alphas,
            output_path=grid_path,
            title=f"Direction {direction_idx} (Quality Score: {result.quality_score:.3f})",
        )

        # Save metric curve
        curve_path = output_dir / f"direction_{rank}_metrics.png"
        create_metric_curve(
            config.evaluation_config.alphas,
            result.distances_forward,
            result.distances_reversed,
            direction_id=direction_idx,
            output_path=curve_path,
        )

        console.print(f"[green]✓[/green] Saved direction {rank} visualizations")

    # Print summary table
    console.print()
    console.print("[bold cyan]Top Directions[/bold cyan]")

    table = Table(title="Stability Results (Top 5)")
    table.add_column("Rank", style="cyan")
    table.add_column("Dir ID", style="magenta")
    table.add_column("Quality Score", style="green")
    table.add_column("Strength", style="yellow")
    table.add_column("Reversibility", style="red")

    for rank, result in enumerate(results_sorted[:5]):
        table.add_row(
            str(rank + 1),
            str(result.direction_id),
            f"{result.quality_score:.4f}",
            f"{result.strength_score:.4f}",
            f"{result.reversibility_score:.6f}",
        )

    console.print(table)
    console.print()
    console.print(f"[bold green]✓ Experiment complete![/bold green]")
    console.print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    typer.run(run_direction_discovery)