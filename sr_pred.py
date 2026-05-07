import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch, yaml, json, re
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import SRCvDataset
from model import SRCNN_Shuffle, SRCNN_Residual


def save_prediction_tiff(pred_array, reference_tiff_path, output_path):
    """
    Save a 2D prediction array as a GeoTIFF, copying the CRS and
    geotransform from the reference TIFF (the original 1km input).

    Parameters
    ----------
    pred_array         : np.ndarray [H, W] — denormalized prediction
    reference_tiff_path: str — path to original 1km TIFF (for CRS + transform)
    output_path        : str — where to save the output GeoTIFF
    """
    with rasterio.open(reference_tiff_path) as src:
        profile = src.profile.copy()

    # Update profile for single-band float32 output
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw',
        nodata=np.nan,
    )

    # Ensure dimensions match
    assert pred_array.shape == (profile['height'], profile['width']), \
        f"Shape mismatch: pred {pred_array.shape} vs ref ({profile['height']}, {profile['width']})"

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(pred_array.astype(np.float32), 1)


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

    # ── Load model ──
    if arch == "SRCNN_Shuffle":
        model = SRCNN_Shuffle(in_channels=in_channels_model).to(device)
    elif arch == "SRCNN_Residual":
        model = SRCNN_Residual(in_channels=in_channels_model).to(device)
    model_path = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["model_path"])
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # ── Denormalization stats ──
    x1_b1_mean = stats["x1"]["band_1"]["mean"]
    x1_b1_std  = stats["x1"]["band_1"]["std"]
    y1_mean    = stats["y1"]["band_1"]["mean"]
    y1_std     = stats["y1"]["band_1"]["std"]
    y9_mean    = stats["y9"]["band_1"]["mean"]
    y9_std     = stats["y9"]["band_1"]["std"]

    # ── Output directory for GeoTIFFs ──
    # tiff_out_dir = os.path.join(cfg["paths"]["save_dir"], "predictions_tiff")
    tiff_out_dir = os.path.join(cfg["validation"]["raster_out"])
    os.makedirs(tiff_out_dir, exist_ok=True)

    num_samples = cfg.get("inference", {}).get("num_samples", len(dataset))
    plot_samples = cfg.get("inference", {}).get("plot_samples", 1)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            x_1km, y_1km, y_9km, center_lon, center_lat, date_str, *_ = batch
            x_1km = x_1km.to(device)

            if geomodel:
                center_lon = center_lon.to(device)
                center_lat = center_lat.to(device)
                pred_1km = model(x_1km, center_lat, center_lon, date_str)
            else:
                pred_1km = model(x_1km)

            # ── Denormalize to physical units ──
            pred_dn  = (pred_1km.cpu().numpy() * y1_std + y1_mean)       # [1, 1, 54, 54]
            y_1km_dn = (y_1km.numpy() * y1_std + y1_mean)
            y_9km_dn = (y_9km.numpy() * y9_std + y9_mean)
            x1_band0_dn = (x_1km[:, 0:1].cpu().numpy() * x1_b1_std + x1_b1_mean)

            # ── Get the reference TIFF path (same file the dataloader read) ──
            # The dataloader's file list is indexed by the original dataset index.
            # Since shuffle=False and batch_size=1, idx maps directly.
            original_file = dataset.files[idx]
            # Extract the key to find the corresponding 1km input TIFF

            key = re.search(r'(\d{8}_\d+)', os.path.basename(original_file)).group(1)
            reference_tiff = os.path.join(
                cfg["paths"]["in_dir"], f"SMAP-E_1km_AM_{key}.tif"
            )

            # ── Save prediction as GeoTIFF ──
            pred_2d = pred_dn[0, 0]  # [54, 54]
            out_filename = f"pred_1km_AM_{key}.tif"
            out_path = os.path.join(tiff_out_dir, out_filename)
            save_prediction_tiff(pred_2d, reference_tiff, out_path)

            # ── Extract date for title / logging ──
            date_val = date_str[0].item() if torch.is_tensor(date_str[0]) else date_str[0]
            date_label = str(date_val)
            if isinstance(date_val, (int, np.integer)) and len(str(date_val)) == 8:
                d = str(date_val)
                date_label = f"{d[:4]}-{d[4:6]}-{d[6:]}"

            print(f"[{idx+1}/{min(num_samples, len(dataset))}] "
                  f"Saved: {out_filename}  |  Date: {date_label}")

            # # ── Plot first few samples ──
            # if idx < plot_samples:
            #     plot_predictions_4x(
            #         x_1km_dn=x1_band0_dn,
            #         y_9km_dn=y_9km_dn,
            #         pred_dn=pred_dn,
            #         y_1km_dn=y_1km_dn,
            #         idx=0,
            #         title=f"Sample #{idx}  |  Date: {date_label}",
            #     )

            # if idx >= num_samples - 1:
            #     break

    print(f"\nAll predictions saved to: {tiff_out_dir}")
    print(f"Total files: {min(num_samples, len(dataset))}")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    run(cfg, device)