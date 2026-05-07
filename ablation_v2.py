"""
Ablation Study: Noise Sensitivity Analysis (v2)
=================================================
- Continuous bands: Gaussian noise (additive, z-score space)
- Categorical bands: Random class-flip noise (probability-based)

This ensures both band types are corrupted in a physically
meaningful way at comparable severity levels.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import SRCvDataset, compare_images, mean_r
from model import SRCNN_Residual


# ──────────────────────────────────────────────────────────────────
# 1. Band layout from stats
# ──────────────────────────────────────────────────────────────────

def get_band_ranges(stats):
    x1_stats = stats["x1"]
    ranges = OrderedDict()
    band_types = OrderedDict()
    ch = 0
    for band_key in sorted(x1_stats.keys(), key=lambda k: int(k.split("_")[1])):
        band_info = x1_stats[band_key]
        band_idx = int(band_key.split("_")[1])
        name_map = {1: "SMAP", 2: "LST", 3: "DEM", 4: "LULC", 5: "STC"}
        name = name_map.get(band_idx, f"Band_{band_idx}")
        if band_info["type"] == "continuous":
            ranges[name] = list(range(ch, ch + 1))
            band_types[name] = "continuous"
            ch += 1
        elif band_info["type"] == "categorical":
            num_classes = band_info["num_classes"]
            ranges[name] = list(range(ch, ch + num_classes))
            band_types[name] = "categorical"
            ch += num_classes
    return ranges, band_types


# ──────────────────────────────────────────────────────────────────
# 2. Noise injection (type-aware)
# ──────────────────────────────────────────────────────────────────

def add_noise_continuous(x, channels, noise_std):
    x_noisy = x.clone()
    for ch in channels:
        noise = torch.randn_like(x_noisy[:, ch:ch+1, :, :]) * noise_std
        x_noisy[:, ch:ch+1, :, :] += noise
    return x_noisy


def add_noise_categorical(x, channels, flip_prob):
    x_noisy = x.clone()
    B, _, H, W = x.shape
    num_classes = len(channels)

    oh_block = x_noisy[:, channels, :, :].clone()
    flip_mask = (torch.rand(B, 1, H, W, device=x.device) < flip_prob)
    random_classes = torch.randint(0, num_classes, (B, H, W), device=x.device)

    new_oh = torch.zeros_like(oh_block)
    new_oh.scatter_(1, random_classes.unsqueeze(1), 1.0)

    flip_mask_expanded = flip_mask.expand_as(oh_block)
    oh_block[flip_mask_expanded] = new_oh[flip_mask_expanded]

    x_noisy[:, channels, :, :] = oh_block
    return x_noisy


def severity_to_flip_prob(severity):
    if severity == 0.0:
        return 0.0
    return min(1.0 - np.exp(-0.5 * severity), 0.95)


def corrupt_input(x, channels, band_type, severity):
    if severity == 0.0:
        return x
    if band_type == "continuous":
        return add_noise_continuous(x, channels, noise_std=severity)
    elif band_type == "categorical":
        flip_prob = severity_to_flip_prob(severity)
        return add_noise_categorical(x, channels, flip_prob=flip_prob)
    return x


# ──────────────────────────────────────────────────────────────────
# 3. Evaluation
# ──────────────────────────────────────────────────────────────────

def evaluate_noisy(model, loader, device, stats, channels, band_type, severity):
    all_metrics = {"PSNR": [], "MSE": [], "Bias": [], "ubRMSE": [], "PearsonR": []}

    y1_mean = stats["y1"]["band_1"]["mean"]
    y1_std  = stats["y1"]["band_1"]["std"]

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_1km, y_1km, y_9km, *_ = batch
            x_1km = x_1km.to(device)
            y_1km = y_1km.to(device)

            x_1km = corrupt_input(x_1km, channels, band_type, severity)

            pred_1km = model(x_1km)

            pred_dn = (pred_1km * y1_std + y1_mean).cpu().numpy()[0, 0]
            y_dn    = (y_1km * y1_std + y1_mean).cpu().numpy()[0, 0]

            data_range = y_dn.max() - y_dn.min()
            if data_range < 1e-6:
                data_range = 0.8

            metrics = compare_images(pred_dn, y_dn, data_range=data_range)
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

    return {
        k: float(mean_r(v)) if k == "PearsonR" else float(np.mean(v))
        for k, v in all_metrics.items()
    }


# ──────────────────────────────────────────────────────────────────
# 4. Plotting (log-scale x-axis, dual x-labels)
# ──────────────────────────────────────────────────────────────────

KNOWN_CATEGORICAL = {"LULC", "STC"}

def get_band_type_from_name(band_name):
    return "categorical" if band_name in KNOWN_CATEGORICAL else "continuous"


def plot_metric(all_results, severity_levels, metric_key, ylabel, save_dir):
    colors = {
        "SMAP": "#e63946",
        "LST":  "#f4a261",
        "DEM":  "#2a9d8f",
        "LULC": "#264653",
        "STC":  "#7209b7",
    }
    markers = {
        "SMAP": "o",
        "LST":  "s",
        "DEM":  "^",
        "LULC": "D",
        "STC":  "v",
    }

    flip_percentages = [severity_to_flip_prob(s) * 100 for s in severity_levels]

    EPSILON = 0.05
    x_values = [s if s > 0 else EPSILON for s in severity_levels]

    fig, ax = plt.subplots(figsize=(9, 6))

    all_lines = []

    for band_name, results_by_severity in all_results.items():
        btype = get_band_type_from_name(band_name)
        metric_values = [results_by_severity[s][metric_key] for s in severity_levels]

        color  = colors.get(band_name, "#333333")
        marker = markers.get(band_name, "o")
        linestyle = "--" if btype == "categorical" else "-"

        line, = ax.plot(
            x_values, metric_values,
            marker=marker,
            color=color,
            linewidth=2,
            markersize=7,
            linestyle=linestyle,
            label=band_name,
        )
        all_lines.append(line)

    # ── Log scale ──
    ax.set_xscale("log")

    ax.set_xticks(x_values)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda val, pos: "0" if val <= EPSILON else
                         (f"{int(val)}" if val == int(val) else f"{val:.1f}")
    ))
    ax.minorticks_off()

    ax.set_xlabel("Noise Standard Deviation (σ)", fontsize=13, color="black")
    ax.tick_params(axis="x", labelcolor="black", labelsize=11)

    # ── Secondary x-axis: Class Perturbation Rate ──
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x_values)
    ax2.set_xticklabels([f"{int(round(p))}" for p in flip_percentages], fontsize=10)
    ax2.minorticks_off()

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 40))
    ax2.set_xlabel("Class Perturbation Rate (%)", fontsize=13, color="black")
    ax2.tick_params(axis="x", labelcolor="black", labelsize=10)

    # ── y-axis ──
    ax.set_ylabel(ylabel, fontsize=13, color="black")
    ax.tick_params(axis="y", labelcolor="black", labelsize=11)

    # ── Title ──
    ax.set_title(
        f"Input Corruption Sensitivity: {metric_key}",
        fontsize=14, fontweight="bold",
    )

    # ── Legend ──
    all_labels = [l.get_label() for l in all_lines]
    ax.legend(
        all_lines, all_labels,
        loc="best",
        fontsize=10,
        framealpha=0.9,
    )

    ax.grid(True, alpha=0.3, which="major")
    fig.tight_layout()

    out_path = os.path.join(save_dir, f"ablation_noise_{metric_key}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_all_metrics(all_results, severity_levels, save_dir):
    metric_map = {
        "PSNR":     "PSNR (dB)",
        "PearsonR": "Pearson Correlation (R)",
        "ubRMSE":   "ubRMSE (m³/m³)",
    }
    for mk, ylabel in metric_map.items():
        plot_metric(all_results, severity_levels, mk, ylabel, save_dir)


# ──────────────────────────────────────────────────────────────────
# 5. Console summary
# ──────────────────────────────────────────────────────────────────

def print_summary(all_results, severity_levels):
    metrics = ["PSNR", "PearsonR", "ubRMSE"]

    print(f"\n{'Band':<8s}", end="")
    for s in severity_levels:
        print(f"  σ={s:<4.1f}", end="")
    print()
    print("─" * (8 + len(severity_levels) * 9))

    for mk in metrics:
        print(f"\n  {mk}:")
        for band_name in all_results:
            print(f"  {band_name:<6s}", end="")
            for s in severity_levels:
                val = all_results[band_name][s][mk]
                print(f"  {val:7.4f}", end="")
            print()


# ──────────────────────────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────────────────────────

def run_noise_ablation(cfg_path="configs/config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    stats_path = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["stats_path"])
    with open(stats_path, "r") as fp:
        stats = json.load(fp)

    band_ranges, band_types = get_band_ranges(stats)
    total_ch = sum(len(v) for v in band_ranges.values())

    print("\n═══ Band Layout ═══")
    for name, chs in band_ranges.items():
        print(f"  {name:8s} → channels {str(chs):30s}  type={band_types[name]}")
    print(f"  Total: {total_ch}\n")

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

    model = SRCNN_Residual(in_channels=total_ch).to(device)
    model_path = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["model_path"])
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded: {model_path}\n")

    severity_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    all_results = OrderedDict()

    for band_name, channels in band_ranges.items():
        btype = band_types[band_name]
        print(f"═══ Corrupting: {band_name} ({btype}, channels {channels}) ═══")
        all_results[band_name] = OrderedDict()

        for sev in severity_levels:
            if btype == "categorical" and sev > 0:
                fp = severity_to_flip_prob(sev)
                suffix = f"  (flip_prob={fp:.2f})"
            else:
                suffix = ""

            results = evaluate_noisy(
                model, loader, device, stats, channels, btype, sev
            )
            all_results[band_name][sev] = results
            print(
                f"  σ={sev:<4.1f}{suffix:20s}  "
                f"PSNR={results['PSNR']:7.3f}  "
                f"R={results['PearsonR']:.4f}  "
                f"ubRMSE={results['ubRMSE']:.4f}"
            )

    # ── Plot ──
    save_dir = cfg["paths"]["save_dir"]
    print("\n═══ Generating Plots ═══")
    plot_all_metrics(all_results, severity_levels, save_dir)

    # ── Summary ──
    print_summary(all_results, severity_levels)

    # ── Save JSON ──
    json_results = {
        band: {str(s): m for s, m in by_sev.items()}
        for band, by_sev in all_results.items()
    }
    out_path = os.path.join(save_dir, "ablation_noise_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run_noise_ablation()