import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import yaml
import json
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

from utils import SRCvDataset
from model import SRCNN_Shuffle, SRCNN_Residual
from utils import plot_predictions, plot_predictions_4x


def focal_mean_nan(arr, size=7):
    """
    Apply focal mean / moving-window average similar to GRASS r.neighbours
    with operation='average' and neighborhood size = size.

    Supports arrays shaped as:
        H x W
        C x H x W
        B x C x H x W

    NaN values are ignored during averaging.
    """

    arr = np.asarray(arr)

    valid = np.isfinite(arr)
    arr_filled = np.where(valid, arr, 0.0)

    # Apply the filter only over the spatial dimensions H and W
    filter_size = [1] * arr.ndim
    filter_size[-2:] = [size, size]

    kernel_area = size * size

    local_sum = uniform_filter(
        arr_filled,
        size=filter_size,
        mode="constant",
        cval=0.0,
    ) * kernel_area

    local_count = uniform_filter(
        valid.astype(np.float32),
        size=filter_size,
        mode="constant",
        cval=0.0,
    ) * kernel_area

    out = local_sum / np.maximum(local_count, 1e-6)
    out[local_count == 0] = np.nan

    return out.astype(arr.dtype, copy=False)


def format_date_label(date_str):
    """
    Safely format date labels coming from the dataset.
    Handles tensor, int, string, and YYYYMMDD-style values.
    """

    date_val = date_str[0]

    if torch.is_tensor(date_val):
        date_val = date_val.item()

    date_label = str(date_val)

    if isinstance(date_val, (int, np.integer)) and len(str(date_val)) == 8:
        d = str(date_val)
        date_label = f"{d[:4]}-{d[4:6]}-{d[6:]}"

    elif isinstance(date_val, str) and date_val.isdigit() and len(date_val) == 8:
        d = date_val
        date_label = f"{d[:4]}-{d[4:6]}-{d[6:]}"

    return date_label


def run(cfg, device):
    stats_path = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["stats_path"])

    with open(stats_path, "r") as fp:
        stats = json.load(fp)

    in_channels_model = 19

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

    geomodel = cfg["model"]["geomodel"]
    arch = cfg["model"]["arch"]

    # ─────────────────────────────
    # Load model
    # ─────────────────────────────
    if arch == "SRCNN_Shuffle":
        model = SRCNN_Shuffle(in_channels=in_channels_model).to(device)

    elif arch == "SRCNN_Residual":
        model = SRCNN_Residual(in_channels=in_channels_model).to(device)

    else:
        raise ValueError(f"Unsupported model architecture: {arch}")

    model_path = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["model_path"])

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    # ─────────────────────────────
    # Denormalization stats
    # ─────────────────────────────
    x1_b1_mean = stats["x1"]["band_1"]["mean"]
    x1_b1_std = stats["x1"]["band_1"]["std"]

    y1_mean = stats["y1"]["band_1"]["mean"]
    y1_std = stats["y1"]["band_1"]["std"]

    y9_mean = stats["y9"]["band_1"]["mean"]
    y9_std = stats["y9"]["band_1"]["std"]

    # ─────────────────────────────
    # Inference settings
    # ─────────────────────────────
    inference_cfg = cfg.get("inference", {})

    num_samples = inference_cfg.get("num_samples", 5)

    # Visualization-only smoothing settings
    smooth_for_display = inference_cfg.get("smooth_for_display", True)
    smooth_size = inference_cfg.get("smooth_size", 7)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            x_1km, y_1km, y_9km, center_lon, center_lat, date_str, *_ = batch

            x_1km = x_1km.to(device)

            # ─────────────────────────────
            # Forward pass
            # ─────────────────────────────
            if geomodel:
                center_lon = center_lon.to(device)
                center_lat = center_lat.to(device)

                pred_1km = model(
                    x_1km,
                    center_lat,
                    center_lon,
                    date_str,
                )
            else:
                pred_1km = model(x_1km)

            # ─────────────────────────────
            # Denormalize everything to physical units
            # ─────────────────────────────
            # Band 0 of x_1km = bilinear-interpolated SMAP
            x1_band0_dn = (
                x_1km[:, 0:1].cpu().numpy() * x1_b1_std + x1_b1_mean
            )

            pred_dn = (
                pred_1km.cpu().numpy() * y1_std + y1_mean
            )

            y_1km_dn = (
                y_1km.numpy() * y1_std + y1_mean
            )

            y_9km_dn = (
                y_9km.numpy() * y9_std + y9_mean
            )

            # ─────────────────────────────
            # Visualization-only smoothing
            # Equivalent to r.neighbours average size=7
            # ─────────────────────────────
            if smooth_for_display:
                x1_plot = focal_mean_nan(x1_band0_dn, size=smooth_size)
                pred_plot = focal_mean_nan(pred_dn, size=smooth_size)
                y1_plot = focal_mean_nan(y_1km_dn, size=smooth_size)

                # Usually keep original 9 km SMAP unchanged,
                # because it is already coarse.
                y9_plot = y_9km_dn

                smoothing_note = f"{smooth_size}x{smooth_size} focal mean for visualization"

            else:
                x1_plot = x1_band0_dn
                pred_plot = pred_dn
                y1_plot = y_1km_dn
                y9_plot = y_9km_dn

                smoothing_note = "raw unsmoothed display"

            # ─────────────────────────────
            # Date label
            # ─────────────────────────────
            date_label = format_date_label(date_str)

            # ─────────────────────────────
            # Plot
            # ─────────────────────────────
            plot_predictions_4x(
                x_1km_dn=x1_plot,
                y_9km_dn=y9_plot,
                pred_dn=pred_plot,
                y_1km_dn=y1_plot,
                idx=0,
                title=f"Sample #{idx} | Date: {date_label} | {smoothing_note}",
            )

            if idx >= num_samples - 1:
                break


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg["device"] if torch.cuda.is_available() else "cpu"
    )

    run(cfg, device)