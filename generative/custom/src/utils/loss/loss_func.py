import torch
import torch.nn.functional as F
from kornia.metrics import SSIM


ssim = SSIM(3)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def l1_smooth_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target)


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1 - ssim(pred, target)).mean()


def gaussian_nll_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.gaussian_nll_loss(pred, target, torch.ones_like(pred, requires_grad=True))


def cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(pred, target)


calculate_loss = mse_loss
