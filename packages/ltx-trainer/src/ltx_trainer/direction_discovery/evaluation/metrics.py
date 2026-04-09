"""Distance metrics for evaluating direction stability."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class DistanceMetric(ABC):
    """Base class for distance metrics."""

    @abstractmethod
    def compute(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """
        Compute distance between two tensors.

        Args:
            x1: First tensor
            x2: Second tensor

        Returns:
            Scalar distance value
        """
        pass


class L2Metric(DistanceMetric):
    """L2 (Euclidean) distance metric."""

    def compute(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """Compute L2 distance."""
        return float(torch.norm(x1 - x2).item())


class SSIMMetric(DistanceMetric):
    """Structural Similarity Index (SSIM) metric."""

    def __init__(self, data_range: float = 1.0, window_size: int = 11):
        """
        Initialize SSIM metric.

        Args:
            data_range: Range of input data
            window_size: Gaussian window size
        """
        self.data_range = data_range
        self.window_size = window_size

    def compute(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """
        Compute SSIM distance (1 - SSIM).

        Args:
            x1: First tensor [B, C, F, H, W] or [B, C, H, W]
            x2: Second tensor [B, C, F, H, W] or [B, C, H, W]

        Returns:
            Distance value in [0, 2]
        """
        # Average over frames and batch for consistency
        if x1.dim() == 5:  # [B, C, F, H, W]
            x1 = x1.mean(dim=2)  # Average over frames
            x2 = x2.mean(dim=2)

        if x1.shape[0] > 1:
            x1 = x1.mean(dim=0, keepdim=True)
            x2 = x2.mean(dim=0, keepdim=True)

        return float(1.0 - self._ssim(x1, x2).item())

    def _ssim(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute structural similarity."""
        C1 = 0.01**2
        C2 = 0.03**2

        # Compute means
        mu1 = self._gauss_blur(x1)
        mu2 = self._gauss_blur(x2)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        # Compute variances
        sigma1_sq = self._gauss_blur(x1**2) - mu1_sq
        sigma2_sq = self._gauss_blur(x2**2) - mu2_sq
        sigma12 = self._gauss_blur(x1 * x2) - mu1_mu2

        # SSIM formula
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim.mean()

    def _gauss_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Simple Gaussian blur."""
        import torch.nn.functional as F

        kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=x.device)
        kernel = (kernel / kernel.sum()).view(1, 1, 1, 5)
        kernel2d = kernel * kernel.transpose(2, 3)
        kernel2d = kernel2d.repeat(x.shape[1], 1, 1, 1)

        x_blurred = F.conv2d(x, kernel2d, padding=2, groups=x.shape[1])
        return x_blurred


class LPIPSMetric(DistanceMetric):
    """LPIPS (Learned Perceptual Image Patch Similarity) metric."""

    def __init__(self, device: torch.device = torch.device("cpu")):
        """
        Initialize LPIPS metric.

        Args:
            device: Device for computation
        """
        self.device = device
        self._lpips_model = None

    @property
    def lpips_model(self):
        """Lazy load LPIPS model."""
        if self._lpips_model is None:
            try:
                import lpips

                self._lpips_model = lpips.LPIPS(net="alex", version="0.1").to(self.device).eval()
            except ImportError:
                self._lpips_model = False
        return self._lpips_model

    def compute(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """
        Compute LPIPS distance.

        Args:
            x1: First tensor [B, C, F, H, W] or [B, C, H, W], range [-1, 1]
            x2: Second tensor

        Returns:
            Distance value
        """
        # Average over frames if 5D
        if x1.dim() == 5:
            x1 = x1.mean(dim=2)
            x2 = x2.mean(dim=2)

        # Normalize to [-1, 1]
        x1 = self._normalize(x1)
        x2 = self._normalize(x2)

        if self.lpips_model is False:
            return float(torch.norm(x1 - x2).item())

        with torch.inference_mode():
            dist = self.lpips_model(x1.to(self.device), x2.to(self.device))

        return float(dist.mean().item())

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [-1, 1] range."""
        if x.min() >= 0:
            # [0, 1] -> [-1, 1]
            return 2 * x - 1
        return x


def create_metric(metric_name: str, device: torch.device = torch.device("cpu")) -> DistanceMetric:
    """Factory function to create metric."""
    metrics = {
        "l2": L2Metric,
        "ssim": SSIMMetric,
        "lpips": lambda: LPIPSMetric(device),
    }

    if metric_name not in metrics:
        raise ValueError(f"Unknown metric: {metric_name}")

    return metrics[metric_name]()