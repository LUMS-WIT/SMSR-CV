import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import yaml, json
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import compare_images, mean_r, SRCvDataset, logs_display, calculate_dataset_stats_cv
from model import SRCNN_Shuffle, SRCNN_Residual

def evaluate_model(model, loader, device, geomodel, stats):
    """
    Evaluate the trained SR model on test data.

    All metrics are computed in PHYSICAL (denormalized) space so they
    are interpretable and comparable with the baseline.

    Uses training-set stats for denormalization (standard practice):
    the model was trained on that distribution, so test data must be
    inverse-transformed with the same parameters.
    """
    all_metrics = {"PSNR": [], "MSE": [], "Bias": [], "ubRMSE": [], "PearsonR": []}

    y1_mean = stats["y1"]["band_1"]["mean"]
    y1_std  = stats["y1"]["band_1"]["std"]

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Model Evaluation"):
            x_1km, y_1km, y_9km, center_lon, center_lat, date_str, *_ = batch

            x_1km = x_1km.to(device)
            y_1km = y_1km.to(device)

            if geomodel:
                center_lon = center_lon.to(device)
                center_lat = center_lat.to(device)
                pred_1km = model(x_1km, center_lat, center_lon, date_str)
            else:
                pred_1km = model(x_1km)

            # ── Denormalize to physical soil moisture (m³/m³) ──
            pred_dn = (pred_1km * y1_std + y1_mean).cpu().numpy()[0, 0]   # [H, W]
            y_dn    = (y_1km * y1_std + y1_mean).cpu().numpy()[0, 0]      # [H, W]

            # data_range from the ground truth for this sample
            data_range = y_dn.max() - y_dn.min()

            metrics = compare_images(pred_dn, y_dn, data_range=data_range)
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

    results = {k: mean_r(v) if k == "PearsonR" else np.mean(v) for k, v in all_metrics.items()}
    return results


def evaluate_model_adapted(model, loader, device, geomodel, train_stats, test_stats):
    """
    Evaluate with simple domain adaptation:
    Shift model predictions to align with test region's statistics.

    Parameters
    ----------
    model        : trained SR model
    loader       : DataLoader for test set
    device       : torch device
    geomodel     : bool
    train_stats  : dict — stats JSON from training set
    test_stats   : dict — stats JSON from test set
    """
    # ── Training set stats (used for normalization during training) ──
    y1_train_mean = train_stats["y1"]["band_1"]["mean"]
    y1_train_std  = train_stats["y1"]["band_1"]["std"]

    # ── Test set stats (the test region's distribution) ──
    y1_test_mean = test_stats["y1"]["band_1"]["mean"]
    y1_test_std  = test_stats["y1"]["band_1"]["std"]

    # In normalized space:
    #   train data → mean ≈ 0, std ≈ 1  (by definition of z-score)
    #   test data  → mean = (y1_test_mean - y1_train_mean) / y1_train_std
    #                std  = y1_test_std / y1_train_std
    # If these differ from 0 and 1, there's a distribution shift.

    norm_test_mean = (y1_test_mean - y1_train_mean) / (y1_train_std + 1e-6)
    norm_test_std  = y1_test_std / (y1_train_std + 1e-6)

    print(f"Train y1 (physical): mean={y1_train_mean:.4f}, std={y1_train_std:.4f}")
    print(f"Test  y1 (physical): mean={y1_test_mean:.4f}, std={y1_test_std:.4f}")
    print(f"Test  y1 (in train-normalized space): mean={norm_test_mean:.4f}, std={norm_test_std:.4f}")

    # ── Evaluate with mean-shift correction ──
    all_metrics = {"PSNR": [], "MSE": [], "Bias": [], "ubRMSE": [], "PearsonR": []}
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Adapted Evaluation"):
            x_1km, y_1km, y_9km, center_lon, center_lat, date_str, *_ = batch

            x_1km = x_1km.to(device)
            y_1km = y_1km.to(device)

            if geomodel:
                center_lon = center_lon.to(device)
                center_lat = center_lat.to(device)
                pred_1km = model(x_1km, center_lat, center_lon, date_str)
            else:
                pred_1km = model(x_1km)

            # Shift prediction in normalized space to correct for mean offset
            # Model predicts in train-normalized space (mean≈0),
            # but test targets have mean≈norm_test_mean in that space.
            pred_1km_adapted = pred_1km + norm_test_mean

            # Denormalize using TRAIN stats (model weights are calibrated to these)
            pred_dn = (pred_1km_adapted * y1_train_std + y1_train_mean).cpu().numpy()[0, 0]
            y_dn    = (y_1km * y1_train_std + y1_train_mean).cpu().numpy()[0, 0]

            data_range = y_dn.max() - y_dn.min()
            if data_range < 1e-6:
                data_range = 0.8

            metrics = compare_images(pred_dn, y_dn, data_range=data_range)
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

    results = {k: mean_r(v) if k == "PearsonR" else np.mean(v) for k, v in all_metrics.items()}
    return results

def evaluate_baseline(loader, stats):
    """
    Baseline: compare band 0 of x_1km (bilinear-interpolated SMAP at 1km)
    against y_1km (the 1km proxy target).

    Both are denormalized to physical space using their respective
    training-set stats before comparison.

    Band 0 of x_1km was normalized with stats["x1"]["band_1"],
    y_1km was normalized with stats["y1"]["band_1"].
    """
    all_metrics = {"PSNR": [], "MSE": [], "Bias": [], "ubRMSE": [], "PearsonR": []}

    # Stats for the first band of x_1km (the bilinear SMAP channel)
    x1_b1_mean = stats["x1"]["band_1"]["mean"]
    x1_b1_std  = stats["x1"]["band_1"]["std"]

    # Stats for y_1km
    y1_mean = stats["y1"]["band_1"]["mean"]
    y1_std  = stats["y1"]["band_1"]["std"]

    for batch in tqdm(loader, desc="Baseline Evaluation"):
        x_1km, y_1km, y_9km, *_ = batch

        # ── Denormalize band 0 of x_1km (bilinear SMAP) ──
        baseline_dn = (x_1km[0, 0].numpy() * x1_b1_std + x1_b1_mean)   # [54, 54]

        # ── Denormalize y_1km ──
        y_dn = (y_1km[0, 0].numpy() * y1_std + y1_mean)                 # [54, 54]

        data_range = y_dn.max() - y_dn.min()

        metrics = compare_images(baseline_dn, y_dn, data_range=data_range)
        for k in all_metrics:
            all_metrics[k].append(metrics[k])

    results = {k: mean_r(v) if k == "PearsonR" else np.mean(v) for k, v in all_metrics.items()}
    return results


def run(cfg, device):
    stats_path = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["stats_path"])

    with open(stats_path, "r") as fp:
        stats = json.load(fp)

    stats_test, _ = calculate_dataset_stats_cv(
        x1_dir=cfg["paths"]["in_dir"],
        y9_dir=cfg["paths"]["coarse_dir"],
        y1_dir=cfg["paths"]["fine_dir"],
        x1_bands=cfg["data"]["in_channels"],
        json_path=None,
    )

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

    model_path = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["model_path"])

    # ── Load model ──
    if arch == "SRCNN_Shuffle":
        model = SRCNN_Shuffle(in_channels=in_channels_model).to(device)
    elif arch == "SRCNN_Residual":
        model = SRCNN_Residual(in_channels=in_channels_model).to(device)
    
    model_path = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["model_path"])
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # ── Model evaluation ──
    model_scores = evaluate_model(model, loader, device, geomodel, stats)
    # model_scores = evaluate_model_adapted(model, loader, device, geomodel, stats, stats_test)  # Adapted evaluation with test stats
    print(f"\nModel: {arch}:")
    for k, v in model_scores.items():
        print(f"  {k}: {v:.4f}")

    # ── Baseline evaluation (reuses same loader — no extra dataset) ──
    baseline_scores = evaluate_baseline(loader, stats)
    print("\nBaseline on Test Set:")
    for k, v in baseline_scores.items():
        print(f"  {k}: {v:.4f}")

    # ── Training logs ──
    log_path = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["training_log"])
    logs_display(log_path)

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    run(cfg, device)
