import torch
import torch.nn as nn
import torch.nn.functional as F



def psnr(pred, target):
    mse = nn.functional.mse_loss(pred, target)
    return float('inf') if mse == 0 else 20 * torch.log10(1.0 / torch.sqrt(mse))

def batch_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def ubrmse(pred, gt, eps=1e-6):
    """Unbiased Root Mean Square Error — removes mean bias before computing RMSE."""
    mp = pred.mean(dim=[1, 2, 3], keepdim=True)
    mg = gt.mean(dim=[1, 2, 3], keepdim=True)
    return torch.sqrt(((pred - mp - (gt - mg)) ** 2).mean(dim=[1, 2, 3]) + eps).mean()


def block_mean(tensor, block_size=9):
    """
    Compute non-overlapping block means.

    tensor : [B, 1, 54, 54]
    returns: [B, 1,  6,  6]   (each value = mean of the corresponding 9×9 block)

    Uses AvgPool2d which is differentiable → gradients flow back through this.
    """
    return F.avg_pool2d(tensor, kernel_size=block_size, stride=block_size)


def sr_loss(pred_1km, y_1km, y_9km, alpha=0.0, lam=1.0,
            y1_mean=0.0, y1_std=1.0, y9_mean=0.0, y9_std=1.0):
    """
    Dual-scale super-resolution loss.

    Parameters
    ----------
    pred_1km : [B, 1, 54, 54]  — model prediction at 1 km
    y_1km    : [B, 1, 54, 54]  — 1 km proxy target
    y_9km    : [B, 1,  6,  6]  — original 9 km SMAP observation
    alpha    : float            — weight on ubrmse term
    lam      : float (λ)       — weight on block-consistency loss

    Returns
    -------
    total_loss : scalar tensor

    The block loss is what makes this fundamentally SR:
      it ties the fine prediction back to the coarse observation,
      enforcing cross-scale physical consistency.

    Normalization:
    The model output pred_1km is trained against y_1km, so it learns to 
    produce values in y1-normalized space. When you block-average pred_1km 
    to get pred_9km, that result is still in y1-space. But y_9km was normalized 
    with y9 stats. You're comparing apples to oranges.
    
    All inputs are in their respective normalized spaces:
    pred_1km : [B, 1, 54, 54]  — model output (y1-normalized space)
    y_1km    : [B, 1, 54, 54]  — target (y1-normalized space)
    y_9km    : [B, 1,  6,  6]  — coarse obs (y9-normalized space)  ← different!

    The block loss requires both sides in the same space.
    We convert y_9km: y9-norm → physical → y1-norm.
    """
    # ── Pixel-level loss (1 km) ──
    loss_pixel = F.mse_loss(pred_1km, y_1km)
    if alpha > 0:
        loss_pixel = loss_pixel + alpha * ubrmse(pred_1km, y_1km)

    # ── Block loss ──
    pred_9km = block_mean(pred_1km, block_size=9)  # [B, 1, 6, 6] in y1-norm space

    # Convert y_9km from y9-normalized → physical → y1-normalized
    #   physical = y_9km * y9_std + y9_mean
    #   y1_norm  = (physical - y1_mean) / y1_std
    #   combined = y_9km * (y9_std / y1_std) + (y9_mean - y1_mean) / y1_std
    y_9km_in_y1_space = y_9km * (y9_std / (y1_std + 1e-6)) + (y9_mean - y1_mean) / (y1_std + 1e-6)

    loss_block = F.mse_loss(pred_9km, y_9km_in_y1_space)

    return loss_pixel + lam * loss_block