"""Visualization utilities for direction discovery results."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec


def create_direction_grid(
    original_video: torch.Tensor,
    modified_videos: Dict[float, torch.Tensor],
    alphas: List[float],
    frame_indices: Optional[List[int]] = None,
    output_path: Optional[Path] = None,
    title: str = "Direction Applied",
    dpi: int = 100,
) -> Optional[np.ndarray]:
    """
    Create a grid visualization of direction application at different alphas.

    Args:
        original_video: Original video [C, F, H, W]
        modified_videos: Dictionary {alpha: modified_video}
        alphas: Sorted list of alpha values
        frame_indices: Indices of frames to display (default: first and last)
        output_path: Optional path to save figure
        title: Title for the plot
        dpi: DPI for saved figure

    Returns:
        Numpy array of the figure if output_path is None
    """
    if frame_indices is None:
        frame_indices = [0, original_video.shape[1] - 1]

    # Create figure
    fig = plt.figure(figsize=(len(alphas) + 1, len(frame_indices) + 1), dpi=dpi)
    gs = GridSpec(len(frame_indices) + 1, len(alphas) + 1, figure=fig)

    # Title row
    ax = fig.add_subplot(gs[0, :])
    ax.text(0.5, 0.5, title, transform=ax.transAxes, fontsize=16, fontweight="bold", ha="center", va="center")
    ax.axis("off")

    # Process frames
    for row_idx, frame_idx in enumerate(frame_indices):
        if frame_idx < 0:
            frame_idx = original_video.shape[1] + frame_idx

        # Original frame
        orig_frame = original_video[:, frame_idx].detach().float().cpu().permute(1, 2, 0).numpy()
        orig_frame = np.clip(orig_frame, 0, 1)

        ax = fig.add_subplot(gs[row_idx + 1, 0])
        ax.imshow(orig_frame)
        ax.set_ylabel(f"Frame {frame_idx}", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Modified frames at different alphas
        for col_idx, alpha in enumerate(alphas):
            if alpha not in modified_videos:
                continue

            mod_video = modified_videos[alpha]
            mod_frame = mod_video[:, frame_idx].detach().float().cpu().permute(1, 2, 0).numpy()
            mod_frame = np.clip(mod_frame, 0, 1)

            ax = fig.add_subplot(gs[row_idx + 1, col_idx + 1])
            ax.imshow(mod_frame)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(f"α={alpha:.2f}", fontweight="bold")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return None
    else:
        # Return as numpy array
        canvas = fig.canvas
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return image


def create_reversibility_plot(
    results: List,
    output_path: Optional[Path] = None,
    dpi: int = 100,
) -> None:
    """
    Create scatter plot of strength vs. reversibility.

    Args:
        results: List of StabilityResults objects
        output_path: Path to save figure
        dpi: DPI for saved figure
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    # Extract data
    strengths = [r.strength_score for r in results]
    reversibilities = [r.reversibility_score for r in results]
    quality_scores = [r.quality_score for r in results]

    # Color by quality score
    scatter = ax.scatter(strengths, reversibilities, c=quality_scores, s=100, cmap="viridis", alpha=0.6)

    # Add labels
    ax.set_xlabel("Strength Score (L2 from original)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Reversibility Error (lower is better)", fontsize=12, fontweight="bold")
    ax.set_title("Direction Stability Analysis", fontsize=14, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Quality Score", fontsize=11, fontweight="bold")

    # Add diagonal reference
    max_val = max(max(strengths), max(reversibilities))
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Equal strength/reversibility")
    ax.legend()

    ax.grid(True, alpha=0.3)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()


def create_metric_curve(
    alphas: List[float],
    distances_forward: Dict[float, float],
    distances_reversed: Dict[float, float],
    direction_id: int = 0,
    output_path: Optional[Path] = None,
    dpi: int = 100,
) -> None:
    """
    Create plot of distance metrics vs. alpha.

    Args:
        alphas: List of tested alpha values
        distances_forward: Dict {alpha: distance}
        distances_reversed: Dict {alpha: distance}
        direction_id: Direction identifier
        output_path: Path to save figure
        dpi: DPI for saved figure
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    # Sort alphas
    alphas_sorted = sorted(alphas)
    forward = [distances_forward[a] for a in alphas_sorted]
    reversed_vals = [distances_reversed[a] for a in alphas_sorted]

    # Plot
    ax.plot(alphas_sorted, forward, "o-", linewidth=2, label="Forward (applied direction)", markersize=8)
    ax.plot(alphas_sorted, reversed_vals, "s-", linewidth=2, label="Reversed (after undoing)", markersize=8)

    ax.set_xlabel("Alpha (Strength Multiplier)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance (L2)", fontsize=12, fontweight="bold")
    ax.set_title(f"Stability Metrics - Direction {direction_id}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()