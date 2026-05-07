import os, time, torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import LossLogger, SRCvDataset, calculate_dataset_stats_cv
from loss import sr_loss, batch_psnr
from model import SRCNN_Shuffle, SRCNN_Residual
import yaml
import json


# ── Forward helper ──────────────────────────────────────────────────

def model_forward(model, batch, geomodel, device):
    """
    Dataloader yields:
        x_1km          [B, C, 54, 54]   — input multiband
        y_1km          [B, 1, 54, 54]   — 1 km target
        y_9km          [B, 1,  6,  6]   — 9 km SMAP observation
        center_lon     [B]
        center_lat     [B]
        date           list[str] of length B
        None           (placeholder, ignored)

    Returns: pred_1km, y_1km, y_9km
    """
    x_1km, y_1km, y_9km, center_lon, center_lat, date_str, *_ = batch

    # Move tensors to device (strings/None are left as-is)
    x_1km = x_1km.to(device)
    y_1km = y_1km.to(device)
    y_9km = y_9km.to(device)

    if geomodel:
        center_lon = center_lon.to(device)
        center_lat = center_lat.to(device)
        pred_1km = model(x_1km, center_lat, center_lon, date_str)
    else:
        pred_1km = model(x_1km)

    return pred_1km, y_1km, y_9km


# ── Training loop ──────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, geomodel, stats, lam=1.0):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_samples = 0

    y1_mean = stats["y1"]["band_1"]["mean"]
    y1_std  = stats["y1"]["band_1"]["std"]
    y9_mean = stats["y9"]["band_1"]["mean"]
    y9_std  = stats["y9"]["band_1"]["std"]

    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        pred_1km, y_1km, y_9km = model_forward(model, batch, geomodel, device)

        loss = sr_loss(
            pred_1km, y_1km, y_9km,
            alpha=0.0, lam=lam,
            y1_mean=y1_mean, y1_std=y1_std,
            y9_mean=y9_mean, y9_std=y9_std,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ---- PSNR on denormalized values (no grad) ----
        with torch.no_grad():
            pred_dn = pred_1km * y1_std + y1_mean
            y_dn    = y_1km * y1_std + y1_mean

            # max_val = actual range of this batch's target
            data_range = y_dn.max() - y_dn.min()
            if data_range < 1e-6:
                data_range = 0.8  # fallback for constant patches

            total_psnr += batch_psnr(pred_dn, y_dn, max_val=data_range).sum().item()
            total_samples += pred_1km.size(0)

    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / total_samples
    return avg_loss, avg_psnr


# ── Evaluation loop ────────────────────────────────────────────────

def evaluate(model, loader, device, geomodel, stats, lam=1.0):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_samples = 0

    y1_mean = stats["y1"]["band_1"]["mean"]
    y1_std  = stats["y1"]["band_1"]["std"]

    y9_mean = stats["y9"]["band_1"]["mean"]
    y9_std  = stats["y9"]["band_1"]["std"]

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pred_1km, y_1km, y_9km = model_forward(model, batch, geomodel, device)

            # Report denormalized metrics
            pred_dn = pred_1km * y1_std + y1_mean
            y_dn    = y_1km * y1_std + y1_mean
            # y9_dn   = y_9km * y9_std + y9_mean  # if y_9km was also normalized

            # max_val = actual range of this batch's target
            data_range = y_dn.max() - y_dn.min()
            if data_range < 1e-6:
                data_range = 0.8  # fallback for constant patches
                
            total_loss += sr_loss(
                pred_1km, y_1km, y_9km,
                lam=lam,
                y1_mean=y1_mean, y1_std=y1_std,
                y9_mean=y9_mean, y9_std=y9_std,
            ).item()

            total_psnr += batch_psnr(pred_dn, y_dn, max_val=data_range).sum().item()
            total_samples += pred_1km.size(0)

    return total_loss / len(loader), total_psnr / total_samples

def run(cfg, device):
    os.makedirs(cfg["paths"]["save_dir"], exist_ok=True)

    stats_path = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["stats_path"])

    stats, in_channels_model = calculate_dataset_stats_cv(
        x1_dir=cfg["paths"]["in_dir"],
        y9_dir=cfg["paths"]["coarse_dir"],
        y1_dir=cfg["paths"]["fine_dir"],
        x1_bands=cfg["data"]["in_channels"],
        json_path=stats_path,
    )
    # if stats_path is not None:
    #     with open(stats_path, "r") as fp:
    #         stats = json.load(fp)
    # # print("Number of input channels for the model:", in_channels_model)
    # in_channels_model= 19

    dataset = SRCvDataset(
        in_dir=cfg["paths"]["in_dir"],
        coarse_dir=cfg["paths"]["coarse_dir"],
        fine_dir=cfg["paths"]["fine_dir"],
        coarse_res=cfg["data"]["coarse_res"],       # redundant ?
        fine_res=cfg["data"]["fine_res"],           # redundant ?
        scale_factor=cfg["data"]["scale_factor"],   # redundant ?
        stats_path=stats_path,
    )

    val_split = cfg["training"]["val_split"]
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ── Model ──
    geomodel = cfg["model"]["geomodel"]

    arch = cfg["model"]["arch"]
    scale_factor = cfg["data"]["scale_factor"]

    # === Model ===
    if arch == "SRCNN_Shuffle":
        model = SRCNN_Shuffle(in_channels=in_channels_model).to(device)
    elif arch == "SRCNN_Residual":
        model = SRCNN_Residual(in_channels=in_channels_model).to(device)

    geomodel = cfg["model"]["geomodel"]
    lam = cfg["training"]["lambda"]   # λ for block loss

    optimizer = optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]),
                           weight_decay=float(cfg["training"]["weight_decay"]))
    best_psnr = 10.0
    best_model_path = os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["model_path"])

    logging_on = cfg["training"]["log"]
    if logging_on:
        logger = LossLogger(
            os.path.join(cfg["paths"]["save_dir"], cfg["paths"]["training_log"]),
            overwrite=True,
        )

    # ── Early stopping config ──
    max_epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["patience"]
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=int(patience/2), min_lr=1e-6
    )

    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")

        train_loss, train_psnr = train_one_epoch(
            model, train_loader, optimizer, device, geomodel, stats, lam=lam
        )
        val_loss, val_psnr = evaluate(
            model, val_loader, device, geomodel, stats, lam=lam
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} | "
            f"Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f}"
        )
        
        if logging_on:
            logger.log(train_loss, train_psnr, val_loss, val_psnr)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! (PSNR: {best_psnr:.2f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs")

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}. "
                      f"No PSNR improvement for {patience} consecutive epochs.")
                break
        
        scheduler.step(val_psnr)

    print(f"\nTraining done. Best model: {best_model_path} (PSNR: {best_psnr:.2f})")

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    run(cfg, device)
