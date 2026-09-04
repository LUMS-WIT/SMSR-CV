import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
import yaml
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

from utils import SRUsDataset
from model import SRCNN_Shuffle, SRCNN_Residual


def save_prediction_tiff(pred_array, reference_tiff_path, output_path):
    """Save a 2D denormalized prediction as a single-band GeoTIFF."""

    with rasterio.open(reference_tiff_path) as src:
        profile = src.profile.copy()

    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress="lzw",
        nodata=np.nan,
    )

    if pred_array.shape != (profile["height"], profile["width"]):
        raise ValueError(
            f"Shape mismatch: prediction={pred_array.shape}, "
            f"reference=({profile['height']}, {profile['width']})"
        )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred_array.astype(np.float32), 1)


def run(cfg, device):
    stats_path = os.path.join(
        cfg["paths"]["save_dir"],
        cfg["paths"]["stats_path"]
    )

    with open(stats_path, "r") as fp:
        stats = json.load(fp)

    # Must match the input-channel count used during model training.
    in_channels_model = 19

    # Prediction-only dataset: uses 1 km R-M inputs and 9 km SMAP files.
    dataset = SRUsDataset(
        in_dir=cfg["paths"]["in_dir"],
        coarse_dir=cfg["paths"]["coarse_dir"],
        coarse_res=cfg["data"]["coarse_res"],
        fine_res=cfg["data"]["fine_res"],
        scale_factor=cfg["data"]["scale_factor"],
        stats_path=stats_path,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    geomodel = cfg["model"]["geomodel"]
    arch = cfg["model"]["arch"]

    if arch == "SRCNN_Shuffle":
        model = SRCNN_Shuffle(in_channels=in_channels_model).to(device)
    elif arch == "SRCNN_Residual":
        model = SRCNN_Residual(in_channels=in_channels_model).to(device)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model_path = os.path.join(
        cfg["paths"]["save_dir"],
        cfg["paths"]["model_path"]
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # The model output remains normalized using the training 1 km SMAP statistics.
    y1_mean = np.float32(stats["y1"]["band_1"]["mean"])
    y1_std = np.float32(stats["y1"]["band_1"]["std"])

    tiff_out_dir = cfg["validation"]["raster_out"]
    os.makedirs(tiff_out_dir, exist_ok=True)

    num_samples = max(
        cfg.get("inference", {}).get("num_samples", len(dataset)),
        len(dataset),
    )

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            # SRUsDataset output:
            # x_1km, y_9km, center_lon, center_lat, date, placeholder
            x_1km, _, center_lon, center_lat, date, _ = batch
            x_1km = x_1km.to(device)

            if geomodel:
                center_lon = center_lon.to(device)
                center_lat = center_lat.to(device)
                pred_1km = model(x_1km, center_lat, center_lon, date)
            else:
                pred_1km = model(x_1km)

            # Convert model output back to physical soil-moisture units.
            pred_dn = pred_1km.cpu().numpy() * y1_std + y1_mean

            # Find the 1 km R-M raster used as this sample's input.
            coarse_file = dataset.files[idx]
            key = re.search(
                r"(\d{8}_\d+)",
                os.path.basename(coarse_file)
            ).group(1)

            reference_tiff = os.path.join(
                cfg["paths"]["in_dir"],
                f"SMAP-E_1km_AM_{key}.tif"
            )

            pred_2d = pred_dn[0, 0]
            out_filename = f"pred_1km_AM_{key}.tif"
            out_path = os.path.join(tiff_out_dir, out_filename)

            save_prediction_tiff(
                pred_array=pred_2d,
                reference_tiff_path=reference_tiff,
                output_path=out_path,
            )

            date_value = date[0].item()
            date_label = str(date_value)
            if len(date_label) == 8:
                date_label = (
                    f"{date_label[:4]}-{date_label[4:6]}-{date_label[6:]}"
                )

            print(
                f"[{idx + 1}/{num_samples}] Saved: {out_filename} "
                f"| Date: {date_label}"
            )

            if idx + 1 >= num_samples:
                break

    print(f"\nAll predictions saved to: {tiff_out_dir}")
    print(f"Total files: {num_samples}")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg["device"] if torch.cuda.is_available() else "cpu"
    )

    run(cfg, device)