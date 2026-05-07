"""
Ablation Study: Input Band Masking (Feature Importance)
========================================================
For each ablation variant, the specified input band(s) are replaced
with their global mean (from training stats) at inference time.
This isolates each predictor's contribution without retraining.

Band layout (19 channels after one-hot expansion):
  Band  0    : SMAP 1km-R (bilinear broadcast)
  Band  1    : LST (MYD11A1)
  Band  2    : DEM (SRTM)
  Bands 3–N  : LULC one-hot  (from original band 4, categorical)
  Bands N+1–M: STC one-hot   (from original band 5, categorical)

Ablation variants:
  1. Mask SMAP only        → how much does the coarse SM skip contribute?
  2. Mask LST only         → quantifies thermal guidance (title claim)
  3. Mask DEM only
  4. Mask LULC only        → all one-hot channels for LULC
  5. Mask STC only         → all one-hot channels for STC
  6. Mask ALL aux (keep SMAP only) → pure residual baseline
  7. Mask SMAP (keep aux only)     → aux-only prediction
  8. Full model (no masking)       → reference
"""

import os
import json
import yaml
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

from utils import SRCvDataset, compare_images, mean_r
from model import SRCNN_Residual


# ──────────────────────────────────────────────────────────────────
# 1. Determine band ranges from stats
# ──────────────────────────────────────────────────────────────────

def get_band_ranges(stats):
    """
    Parse the stats JSON to determine which expanded channels
    correspond to each original input band.

    Returns:
        dict mapping human-readable names to list of channel indices
        in the 19-channel expanded tensor.
    """
    x1_stats = stats["x1"]
    ranges = OrderedDict()
    ch = 0

    for band_key in sorted(x1_stats.keys(), key=lambda k: int(k.split("_")[1])):
        band_info = x1_stats[band_key]
        band_idx = int(band_key.split("_")[1])

        # Map original band index to name
        name_map = {1: "SMAP", 2: "LST", 3: "DEM", 4: "LULC", 5: "STC"}
        name = name_map.get(band_idx, f"Band_{band_idx}")

        if band_info["type"] == "continuous":
            ranges[name] = list(range(ch, ch + 1))
            ch += 1
        elif band_info["type"] == "categorical":
            num_classes = band_info["num_classes"]
            ranges[name] = list(range(ch, ch + num_classes))
            ch += num_classes

    return ranges


# ──────────────────────────────────────────────────────────────────
# 2. Masking function
# ──────────────────────────────────────────────────────────────────

def mask_bands(x, channels_to_mask, fill_value=0.0):
    """
    Replace specified channels with fill_value (default 0.0 = global
    mean in z-score normalized space).

    Args:
        x : tensor [B, C, H, W]
        channels_to_mask : list of int channel indices
        fill_value : replacement value (0.0 for z-score normalized data)

    Returns:
        masked copy of x
    """
    x_masked = x.clone()
    for ch in channels_to_mask:
        x_masked[:, ch, :, :] = fill_value
    return x_masked


# ──────────────────────────────────────────────────────────────────
# 3. Evaluation with masking
# ──────────────────────────────────────────────────────────────────

def evaluate_masked(model, loader, device, stats, channels_to_mask=None):
    """
    Run inference with optional band masking and compute metrics
    in physical (denormalized) space.
    """
    all_metrics = {"PSNR": [], "MSE": [], "Bias": [], "ubRMSE": [], "PearsonR": []}

    y1_mean = stats["y1"]["band_1"]["mean"]
    y1_std  = stats["y1"]["band_1"]["std"]

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x_1km, y_1km, y_9km, *_ = batch
            x_1km = x_1km.to(device)
            y_1km = y_1km.to(device)

            # Apply masking if specified
            if channels_to_mask is not None and len(channels_to_mask) > 0:
                x_1km = mask_bands(x_1km, channels_to_mask, fill_value=0.0)

            pred_1km = model(x_1km)

            # Denormalize
            pred_dn = (pred_1km * y1_std + y1_mean).cpu().numpy()[0, 0]
            y_dn    = (y_1km * y1_std + y1_mean).cpu().numpy()[0, 0]

            data_range = y_dn.max() - y_dn.min()
            if data_range < 1e-6:
                data_range = 0.8

            metrics = compare_images(pred_dn, y_dn, data_range=data_range)
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

    results = {
        k: float(mean_r(v)) if k == "PearsonR" else float(np.mean(v))
        for k, v in all_metrics.items()
    }
    return results


# ──────────────────────────────────────────────────────────────────
# 4. Define ablation experiments
# ──────────────────────────────────────────────────────────────────

def build_ablation_table(band_ranges):
    """
    Build the list of ablation experiments.

    Each entry: (experiment_name, channels_to_mask)
    """
    all_channels = []
    for chs in band_ranges.values():
        all_channels.extend(chs)

    all_aux = []
    for name, chs in band_ranges.items():
        if name != "SMAP":
            all_aux.extend(chs)

    experiments = OrderedDict()

    # Reference (no masking)
    experiments["Full model (no mask)"] = []

    # Individual drops
    for name, chs in band_ranges.items():
        experiments[f"Drop {name}"] = chs

    # Group drops
    experiments["Drop ALL aux (SMAP only)"] = all_aux
    experiments["Drop SMAP (aux only)"] = band_ranges.get("SMAP", [])

    return experiments


# ──────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────

def run_ablation(cfg_path="configs/config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # ── Load stats ──
    stats_path = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["stats_path"])
    with open(stats_path, "r") as fp:
        stats = json.load(fp)

    # ── Band ranges ──
    band_ranges = get_band_ranges(stats)
    print("\n═══ Expanded Band Layout ═══")
    for name, chs in band_ranges.items():
        print(f"  {name:8s} → channels {chs}")
    total_ch = sum(len(v) for v in band_ranges.values())
    print(f"  Total expanded channels: {total_ch}\n")

    # ── Dataset ──
    dataset = SRCvDataset(
        in_dir=cfg["paths"]["in_dir"],
        coarse_dir=cfg["paths"]["coarse_dir"],
        fine_dir=cfg["paths"]["fine_dir"],
        coarse_res=cfg["data"]["coarse_res"],
        fine_res=cfg["data"]["fine_res"],
        scale_factor=cfg["data"]["scale_factor"],
        stats_path=stats_path,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ── Model ──
    model = SRCNN_Residual(in_channels=total_ch).to(device)
    model_path = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["model_path"])
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from: {model_path}\n")

    # ── Run ablations ──
    experiments = build_ablation_table(band_ranges)

    print("═══ Ablation Study: Input Band Masking ═══\n")
    print(f"{'Experiment':<30s} {'PSNR':>8s} {'MSE':>10s} {'Bias':>8s} {'ubRMSE':>8s} {'R':>8s}")
    print("─" * 76)

    all_results = {}
    for exp_name, mask_chs in experiments.items():
        results = evaluate_masked(model, loader, device, stats, mask_chs)
        all_results[exp_name] = results

        print(
            f"{exp_name:<30s} "
            f"{results['PSNR']:8.4f} "
            f"{results['MSE']:10.6f} "
            f"{results['Bias']:8.4f} "
            f"{results['ubRMSE']:8.4f} "
            f"{results['PearsonR']:8.4f}"
        )

    # ── Compute relative degradation from full model ──
    ref = all_results["Full model (no mask)"]

    print("\n═══ Relative Degradation (Δ from full model) ═══\n")
    print(f"{'Experiment':<30s} {'ΔPSNR':>8s} {'ΔMSE':>10s} {'ΔubRMSE':>8s} {'ΔR':>8s}")
    print("─" * 68)

    for exp_name, results in all_results.items():
        if exp_name == "Full model (no mask)":
            continue
        print(
            f"{exp_name:<30s} "
            f"{results['PSNR'] - ref['PSNR']:+8.4f} "
            f"{results['MSE'] - ref['MSE']:+10.6f} "
            f"{results['ubRMSE'] - ref['ubRMSE']:+8.4f} "
            f"{results['PearsonR'] - ref['PearsonR']:+8.4f}"
        )

    # ── Save results ──
    out_path = os.path.join(cfg["paths"]["save_dir"], "ablation_band_masking.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run_ablation()