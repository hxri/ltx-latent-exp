"""Video I/O utilities."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torchvision.io import read_video, write_video

try:
    import cv2
except ImportError:  # pragma: no cover - environment dependent
    cv2 = None


def load_video(
    video_path: str,
    target_fps: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Load video file and return frames and fps.

    Args:
        video_path: Path to video file
        target_fps: Resample to this FPS (if different from source)
        max_frames: Maximum frames to load

    Returns:
        Tuple of (video tensor [C, F, H, W], fps)
    """
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
            frame_idx += 1

        cap.release()

        if not frames:
            raise ValueError(f"No frames loaded from {video_path}")

        video = torch.stack(frames).permute(1, 0, 2, 3)
        return video, float(fps)

    video_frames, _, metadata = read_video(str(video_path), pts_unit="sec")
    if video_frames.numel() == 0:
        raise ValueError(f"No frames loaded from {video_path}")

    if max_frames is not None:
        video_frames = video_frames[:max_frames]

    fps = float(metadata.get("video_fps", 0.0) or 0.0)
    video = video_frames.permute(3, 0, 1, 2).float() / 255.0  # [C, F, H, W]
    return video, fps


def save_video(
    video: torch.Tensor,
    output_path: Path,
    fps: float = 24.0,
    codec: str = "libx264",
) -> None:
    """
    Save video tensor to file.

    Args:
        video: Video tensor [B, C, F, H, W] or [C, F, H, W], range [0, 1]
        output_path: Output file path
        fps: Frames per second
        codec: Video codec
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove batch dimension if present
    if video.dim() == 5:
        video = video.squeeze(0)

    video = video.detach().cpu()

    # Convert to [F, H, W, C] for torchvision.write_video
    if video.shape[0] in [1, 3, 4]:
        video = video.permute(1, 2, 3, 0)
    elif video.shape[1] in [1, 3, 4]:
        video = video.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Unsupported video tensor shape: {tuple(video.shape)}")

    # Clamp to [0, 1] and convert to uint8
    video = torch.clamp(video, 0, 1)
    video = (video * 255).to(torch.uint8)

    # Write video
    write_video(str(output_path), video, fps=int(fps))


def extract_frames(
    video: torch.Tensor,
    indices: Optional[list[int]] = None,
) -> list[torch.Tensor]:
    """
    Extract specific frames from video.

    Args:
        video: Video tensor [C, F, H, W], range [0, 1]
        indices: Frame indices to extract (default: first and last)

    Returns:
        List of frame tensors [C, H, W]
    """
    if indices is None:
        indices = [0, video.shape[1] - 1]

    frames = []
    for idx in indices:
        if idx < 0:
            idx = video.shape[1] + idx
        frames.append(video[:, idx])

    return frames


def video_to_frames_array(video: torch.Tensor) -> np.ndarray:
    """
    Convert video tensor to numpy array of frames.

    Args:
        video: [C, F, H, W], range [0, 1]

    Returns:
        Numpy array [F, H, W, C] in [0, 255]
    """
    # [C, F, H, W] -> [F, H, W, C]
    video = video.permute(1, 2, 3, 0).numpy()
    # [0, 1] -> [0, 255]
    video = (video * 255).astype(np.uint8)
    return video