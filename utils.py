import numpy as np
import math
import json
import os
import glob
import re
import torch
from collections import Counter
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio
from skimage.metrics import peak_signal_noise_ratio
from scipy.ndimage import generic_filter
from scipy.stats import pearsonr
from pyproj import Transformer
from scipy.ndimage import uniform_filter, distance_transform_edt
from tqdm import tqdm
from torch.nn.functional import one_hot

########################################################################################
# Data loader operations
########################################################################################

class SRCvDataset(Dataset):
    """
    Super-resolution dataset:
      Inputs  : SMAP_P_E (9km → 1km broadcast) + Aux (1km)
      Targets : SMAP_1km proxy (training only)
      Physics : SMAP_P_E 9km (block-mean constraint)
    """

    def __init__(self, in_dir, coarse_dir, fine_dir, coarse_res = "9km", fine_res="1km", scale_factor=9,
                 stats_path=None, transform=None):

        self.in_dir = in_dir
        self.coarse_dir = coarse_dir
        self.fine_dir = fine_dir
        self.coarse_res = coarse_res
        self.fine_res = fine_res
        self.scale_factor = scale_factor
        # self.in_channels = in_channels
        self.transform = transform

        if stats_path is not None:
            with open(stats_path, "r") as fp:
                self.stats = json.load(fp)
        else:
            self.stats = None
        
        self.files = sorted(glob.glob(os.path.join(coarse_dir, "*.tif")))

    def __len__(self):
        return len(self.files)

    def _read(self, path, geoloc=False):
        with rasterio.open(path) as src:
            if not geoloc:
                arr = src.read().astype(np.float32)
                center_lon, center_lat = None, None 
            else:
                arr = src.read().astype(np.float32)
                transform = src.transform
                width = src.width          # (pixel_width: 54 or 6)
                height = src.height
                crs = src.crs

                # Compute center coordinates (pixel-wise)
                center_x = transform * (width // 2, height // 2)
                center_x = torch.tensor(center_x, dtype=torch.float32)

                try:
                    # Convert to lat/lon if CRS is not EPSG:4326
                    if crs and crs.to_epsg() != 4326:
                        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                        center_lon, center_lat = transformer.transform(center_x[0], center_x[1])
                    else:
                        center_lon, center_lat = center_x
                except Exception:
                    center_lon, center_lat = None, None                
        
        return arr, center_lon, center_lat

    def fill_nan_continuous(self, arr, size=3):
        """
        Fill NaNs in continuous geophysical fields using:
        - local mean (ignoring NaNs)
        - nearest-neighbor fallback
        """
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Continuous fill expects 2D array")

        mask = np.isnan(arr)
        if not mask.any():
            return arr

        valid = (~mask).astype(np.float32)
        arr0  = np.where(mask, 0.0, arr)

        num = uniform_filter(arr0, size=size, mode="reflect")
        den = uniform_filter(valid, size=size, mode="reflect")

        out = arr.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            out[mask] = num[mask] / den[mask]

        leftover = np.isnan(out)
        if leftover.any():
            _, idx = distance_transform_edt(leftover, return_indices=True)
            out[leftover] = out[tuple(idx[:, leftover])]

        return out


    def fill_nan_categorical(self, arr):
        """
        Fill NaNs in categorical rasters using nearest valid neighbor.
        Preserves class labels.
        """
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError("Categorical fill expects 2D array")

        mask = np.isnan(arr)
        if not mask.any():
            return arr

        out = arr.copy()
        _, idx = distance_transform_edt(mask, return_indices=True)
        out[mask] = out[tuple(idx[:, mask])]

        assert not np.isnan(out).any()

        return out

    def fill_nan(self, x, size=3):
        """
        Automatically fill NaNs for:
        - single-band (2D): continuous
        - multi-band (C,H,W): last 2 bands categorical, rest continuous

        Args:
            x : numpy array of shape (H,W) or (C,H,W)

        Returns:
            Filled array with same shape
        """
        x = np.asarray(x)

        # -----------------------
        # Case 1: Single-band
        # -----------------------
        if x.ndim == 2:
            return self.fill_nan_continuous(x, size=size)

        # -----------------------
        # Case 2: Multi-band
        # -----------------------
        if x.ndim != 3:
            raise ValueError("Expected input shape (H,W) or (C,H,W)")

        C = x.shape[0]
        if C < 3:
            # Safety: treat all as continuous if too few bands
            return np.stack(
                [self.fill_nan_continuous(x[i], size=size) for i in range(C)],
                axis=0
            )

        out = np.empty_like(x)

        # Continuous bands
        for i in range(C - 2):
            out[i] = self.fill_nan_continuous(x[i], size=size)

        # Categorical bands (last 2)
        out[C - 2] = self.fill_nan_categorical(x[C - 2])
        out[C - 1] = self.fill_nan_categorical(x[C - 1])
        
        assert not np.isnan(out).any()
        return out

    def normalize_multiband(self,x, stats_x):
        """
        Normalize a multi-band tensor using stats metadata.
        Continuous bands → z-score
        Categorical bands → one-hot encoding

        Args:
            x : np.ndarray (C, H, W)
            stats_x : dict (stats["x"])

        Returns:
            np.ndarray : normalized tensor (C', H, W)
        """
        normalized = []

        for i in range(x.shape[0]):
            band_idx = i + 1
            band_stats = stats_x[f"band_{band_idx}"]

            # -------------------------
            # Continuous band
            # -------------------------
            if band_stats["type"] == "continuous":
                mean = np.float32(band_stats["mean"])
                std  = np.float32(band_stats["std"])
                normalized.append((x[i] - mean) / (std + 1e-6))

            # -------------------------
            # Categorical band
            # -------------------------
            elif band_stats["type"] == "categorical":
                band_int = x[i].astype(np.int32)

                mapping = {int(k): v for k, v in band_stats["mapping"].items()}
                num_classes = band_stats["num_classes"]

                # vectorized remap
                remapped = np.vectorize(mapping.get)(band_int).astype(np.int64)

                # one-hot → (H, W, C)
                oh = one_hot(
                    torch.from_numpy(remapped),
                    num_classes=num_classes
                )

                # → (C, H, W)
                oh = oh.permute(2, 0, 1).numpy()
                normalized.extend(oh)

            else:
                raise ValueError(f"Unknown band type: {band_stats['type']}")

        return np.stack(normalized, axis=0)


    def __getitem__(self, idx):

        f9 = self.files[idx]
        # 'training/temporal/train/9km\\SMAP-E_9km_AM_20150403_0.tif' --> key = '20150403_0'
        key = re.search(r'(\d{8}_\d+)', os.path.basename(f9)).group(1)

        # Extract date
        date_str = key.split('_')[0]  # '20150403' 
        date = torch.tensor(int(date_str), dtype=torch.int32) # YYYYMMDD → int

        f1r = os.path.join(self.in_dir, f"SMAP-E_1km_AM_{key}.tif")
        f1  = os.path.join(self.fine_dir,   f"SMAP-E_1km_AM_{key}.tif")

        # -------------------------
        # Read data
        # -------------------------
        x_1km, _, _ = self._read(f1r)            # (C, H, W) [SM, LST, DEM, LULC, STC]
        y_1km, center_lon, center_lat = self._read(f1, geoloc=True)           # (H, W)
        y_9km, _, _ = self._read(f9, geoloc=True)           # (h, w)

        # NOTE: center_lon and lat are same for all 3

        # -------------------------
        # NaN handling
        # -------------------------
        x_1km = self.fill_nan(x_1km)
        y_1km = self.fill_nan(y_1km)
        y_9km = self.fill_nan(y_9km)

        # -------------------------
        # Normalize
        # -------------------------
        x_1km = self.normalize_multiband(x_1km, self.stats["x1"])

        y1_mean = np.float32(self.stats["y1"]["band_1"]["mean"])
        y1_std  = np.float32(self.stats["y1"]["band_1"]["std"])
        y_1km = (y_1km - y1_mean) / (y1_std + 1e-6)

        y9_mean = np.float32(self.stats["y9"]["band_1"]["mean"])
        y9_std  = np.float32(self.stats["y9"]["band_1"]["std"])
        y_9km = (y_9km - y9_mean) / (y9_std + 1e-6)

        # -------------------------
        # Torch
        # -------------------------
        x_1km = torch.from_numpy(x_1km).float()
        y_1km = torch.from_numpy(y_1km).float()  
        y_9km = torch.from_numpy(y_9km).float()  

        # return {
        #     "x_1km": x_1km,   # model input
        #     "y_1km": y_1km,   # proxy (pixel loss)
        #     "y_9km": y_9km,    # physics loss
        #     "date": date      # auxiliary metadata
        # }

        return x_1km, y_1km, y_9km, center_lon, center_lat, date, 'this is a placeholder'

class SREncDataset(Dataset):
    def __init__(self, coarse_dir, fine_dir, coarse_res = "9km", fine_res="1km", scale_factor=9, 
                 in_channels=1, stats_path=None, upsample= True, legacy=False, SMAP=True, transform=None):
        self.coarse_dir = coarse_dir
        self.fine_dir = fine_dir
        self.coarse_res = coarse_res
        self.fine_res = fine_res
        self.upsample = upsample
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.transform = transform
        self.legacy = legacy
        self.SMAP = SMAP

        if stats_path is not None:
            with open(stats_path, "r") as fp:
                self.stats = json.load(fp)
        else:
            self.stats = None

        if not SMAP:
            # Default logic
            self.coarse_files = sorted(glob.glob(os.path.join(coarse_dir, '*.tif')))
            self.fine_files = sorted([
                os.path.join(fine_dir, os.path.basename(f).replace(coarse_res, fine_res))
                for f in self.coarse_files
            ])

            # Filter out unmatched files
            self.coarse_files, self.fine_files = zip(*[
                (c, f) for c, f in zip(self.coarse_files, self.fine_files) if os.path.exists(f)
            ])

        else:
            # SMAP logic: match by YYYYMMDD_id
            coarse_files_all = sorted(glob.glob(os.path.join(coarse_dir, '*.tif')))
            fine_files_dict = {
                re.search(r'(\d{8}_\d+)', os.path.basename(f)).group(1): f
                for f in glob.glob(os.path.join(fine_dir, '*.tif'))
                if re.search(r'(\d{8}_\d+)', os.path.basename(f))
            }

            matched_pairs = []
            for cf in coarse_files_all:
                m = re.search(r'(\d{8}_\d+)', os.path.basename(cf))
                if m:
                    key = m.group(1)
                    if key in fine_files_dict:
                        matched_pairs.append((cf, fine_files_dict[key]))

            if not matched_pairs:
                raise RuntimeError("No matching coarse/fine SMAP file pairs found.")

            self.coarse_files, self.fine_files = zip(*matched_pairs)

    def __len__(self):
        return len(self.coarse_files)

    def upsample_coarse(self, coarse_tensor, scale_factor, mode='bilinear'):
        """
        Upsample a coarse-resolution tensor by a fixed scale factor.

        Args:
            coarse_tensor (Tensor): shape (C, H, W)
            scale_factor (int): Upscaling factor (e.g., 3 for 9km → 3km)
            mode (str): Interpolation mode ('bilinear', 'bicubic', etc.)

        Returns:
            upsampled_tensor (Tensor): shape (C, H * scale_factor, W * scale_factor)
        """
        # Add batch dimension → (1, C, H, W)
        coarse_tensor = coarse_tensor.unsqueeze(0)
        up = F.interpolate(
            coarse_tensor,  # [1, C, H, W]
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False if mode != "nearest" else None
        )
        return up.squeeze(0)  # return [1, H', W']


    def parse_metadata(self, filename):
        basename = os.path.basename(filename)
        # Example (old): SMAP-HB_3km_daily_mean_20150331_0.tif
        # Example (new): SMAP-E_1km_AM_20150403_0.tif
        parts = basename.replace(".tif", "").split("_")

        platform = parts[0]                           # "SMAP_HB"
        res = parts[1]                                # "3km"
        date = parts[3]                               # "20150331"
        block_id = parts[4]
        # block_id = parts[5] if len(parts) > 5 else "0"

        return {
            "platform": platform,
            "res": res,
            "date": date,
            "id": block_id,
            "filename": basename
        }
   
    def fill_nan_legacy(self, arr, size=3):
        """
        Fill NaNs in the array using local mean filtering.
        """
        def mean_filter(window):
            valid = window[~np.isnan(window)]
            return valid.mean() if valid.size > 0 else np.nan
        return generic_filter(arr, mean_filter, size=size, mode='mirror')

    def fill_nan(self, arr, size=3):
        """
        Fill NaNs using local-mean (ignoring NaNs) with nearest-pixel fallback.
        Accepts only numpy arrays or torch tensors; never triggers dataset access.
        """
        arr = np.array(arr, dtype=np.float32, copy=True)

        if arr.ndim != 2:
            raise ValueError(f"fill_nan_fast expects 2D array, got shape={arr.shape}")

        mask = np.isnan(arr)
        if not mask.any():
            return arr

        valid = (~mask).astype(np.float32)
        arr0  = np.where(mask, 0.0, arr)

        num = uniform_filter(arr0, size=size, mode='reflect')
        den = uniform_filter(valid, size=size, mode='reflect')

        out = arr.copy()
        with np.errstate(invalid='ignore', divide='ignore'):
            out[mask] = num[mask] / den[mask]

        # leftover = mask & (den == 0)
        leftover = np.isnan(out)
        if leftover.any():
            # Nearest valid pixel fallback → guarantees no NaNs if any valid pixel exists
            _, idx = distance_transform_edt(leftover, return_distances=True, return_indices=True)
            out[leftover] = out[tuple(idx[:, leftover])]

        return out

    def __getitem__(self, idx):
        coarse_path = self.coarse_files[idx]
        fine_path = self.fine_files[idx]

        # --------------------------------------------------
        # 1. READ COARSE
        # --------------------------------------------------
        with rasterio.open(coarse_path) as csrc:
            # coarse = csrc.read(1).astype(np.float32)
            if self.in_channels == 1:
                coarse = csrc.read(1).astype(np.float32)  # (H, W)
                coarse = coarse[None, ...]  # (1, H, W)
            else:
                coarse = csrc.read().astype(np.float32)
            transform = csrc.transform
            width = csrc.width          # (pixel_width: 8)
            height = csrc.height
            crs = csrc.crs
            res = csrc.res  

            # Compute center coordinates (pixel-wise)
            center_x = transform * (width // 2, height // 2)
            center_x = torch.tensor(center_x, dtype=torch.float32)
            try:
                # Convert to lat/lon if CRS is not EPSG:4326
                if crs and crs.to_epsg() != 4326:
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    center_lon, center_lat = transformer.transform(center_x[0], center_x[1])
                else:
                    center_lon, center_lat = center_x
            except Exception:
                center_lon, center_lat = None, None

        # --------------------------------------------------
        # 2. READ FINE (ALWAYS SINGLE-BAND)
        # --------------------------------------------------
        with rasterio.open(fine_path) as fsrc:
            fine = fsrc.read(1).astype(np.float32)  # (H,W)

        # --------------------------------------------------
        # 3. FILL NaNs
        # --------------------------------------------------

        if np.isnan(fine).any():
            fine = self.fill_nan(fine)
            total_nans = np.isnan(fine).sum()

        assert not np.isnan(fine).any(), f"Fine file at index {idx} contains {np.isnan(fine).sum()} / {total_nans} NaN values after processing."

        # if self.in_channels == 1:
        #     if np.isnan(coarse).any():
        #         coarse = self.fill_nan(coarse)
        #     assert not np.isnan(coarse).any(), f"Coarse file at index {idx} contains {np.isnan(coarse).sum()} NaN values after processing."
        # else:
        #     if np.isnan(coarse[0]).any():
        #         coarse[0] = self.fill_nan(coarse[0])
        #     assert not np.isnan(coarse[0]).any(), f"Coarse file at index {idx} contains {np.isnan(coarse).sum()} NaN values after processing."


        if np.isnan(coarse[0]).any():
            coarse[0] = self.fill_nan(coarse[0])
        assert not np.isnan(coarse[0]).any(), f"Coarse file at index {idx} contains {np.isnan(coarse).sum()} NaN values after processing."


        # --------------------------------------------------
        # 5. NORMALIZATION + ONE-HOT
        # --------------------------------------------------
        if self.in_channels == 1:
            mean = self.stats["coarse"]["band_1"]["mean"]
            std  = self.stats["coarse"]["band_1"]["std"]
            coarse = (coarse - mean) / (std + 1e-6)

        else:
            categorical_bands = [3, 4]  # 1-indexed
            normalized = []

            for i in range(coarse.shape[0]):
                band_idx = i + 1

                if band_idx in categorical_bands:
                    band_int = coarse[i].astype(np.int32)

                    raw_mapping = self.stats["coarse"][f"band_{band_idx}"]["mapping"]
                    num_classes = self.stats["coarse"][f"band_{band_idx}"]["num_classes"]
                    mapping = {int(k): v for k, v in raw_mapping.items()}

                    remapped = np.vectorize(mapping.__getitem__)(band_int).astype(np.int64)

                    oh = one_hot(
                        torch.from_numpy(remapped),
                        num_classes=num_classes
                    )  # (H,W,C)

                    oh = oh.permute(2, 0, 1).numpy()  # (C,H,W)
                    normalized.extend(oh)

                else:
                    mean = self.stats["coarse"][f"band_{band_idx}"]["mean"]
                    std  = self.stats["coarse"][f"band_{band_idx}"]["std"]
                    normalized.append((coarse[i] - mean) / (std + 1e-6))

            coarse = np.stack(normalized, axis=0)

        # --------------------------------------------------
        # 6. NORMALIZE FINE
        # --------------------------------------------------
        fine = (fine - self.stats["fine"]["band_1"]["mean"]) / (
            self.stats["fine"]["band_1"]["std"] + 1e-6
        )


        # --------------------------------------------------
        # TODO: check this part for SRCNN
        # 4. TORCH + UPSAMPLE (BEFORE NORMALIZATION): 
        # --------------------------------------------------
        coarse = torch.from_numpy(coarse)  # (C,H,W)

        if self.upsample:
            coarse = self.upsample_coarse(
                coarse,
                self.scale_factor,
                mode="nearest"   # categorical-safe
            )

        coarse = coarse.numpy()  # back to numpy for normalization

        # --------------------------------------------------
        # 7. FINAL TORCH CONVERSION (FLOAT32)
        # --------------------------------------------------
        coarse = torch.from_numpy(coarse.astype(np.float32))
        fine   = torch.from_numpy(fine.astype(np.float32)).unsqueeze(0)

        # Normalize (optional)
        # coarse = (coarse - np.nanmin(coarse)) / (np.nanmax(coarse) - np.nanmin(coarse) + 1e-6)
        # fine = (fine - np.nanmin(fine)) / (np.nanmax(fine) - np.nanmin(fine) + 1e-6)

        # # *** Apply Normalization Here ***
        # if self.in_channels == 1:
        #     # Normalize single-band (SMAP)
        #     mean = self.stats["fine"]["band_1"]["mean"]
        #     std = self.stats["fine"]["band_1"]["std"]
        #     coarse = (coarse - mean) / (std + 1e-6)
        # else:
        #     # Normalize multi-band (coarse)
        #     categorical_bands = [3, 4]  # 1: SMAP_AM, 2: DEM, 3: LULC, 4: STC, 5: WCpF2, 6: WCpF42, 7: WCres, 8: WCsat
        #     normalized_coarse = []
        #     for i in range(coarse.shape[0]):
        #         band_index = i + 1  # Bands are 1-indexed
        #         if band_index in categorical_bands:
        #             band_int = coarse[i].astype(np.int32)

        #             raw_mapping = self.stats["coarse"][f"band_{band_index}"]["mapping"]
        #             num_classes = self.stats["coarse"][f"band_{band_index}"]["num_classes"]

        #             mapping = {int(k): v for k, v in raw_mapping.items()}

        #             try:
        #                 remapped_band = np.vectorize(mapping.__getitem__)(band_int)
        #             except KeyError as e:
        #                 raise ValueError(f"Unknown categorical value {e} in band {band_index}")

        #             remapped_band = remapped_band.astype(np.int64)

        #             one_hot_band = one_hot(
        #                 torch.from_numpy(remapped_band),
        #                 num_classes=num_classes
        #             )  # (H, W, C)

        #             one_hot_band = one_hot_band.permute(2, 0, 1).numpy()  # (C, H, W)
        #             normalized_coarse.extend(one_hot_band)
        #         else:
        #             # Z-Score Normalization for continuous bands
        #             mean = self.stats["coarse"][f"band_{band_index}"]["mean"]
        #             std = self.stats["coarse"][f"band_{band_index}"]["std"]
        #             band = (coarse[i] - mean) / (std + 1e-6)
        #             normalized_coarse.append(band)
            
        #     coarse = np.stack(normalized_coarse, axis=0).astype(np.float32)


        # # Normalize fine data (always single-band)
        # fine_mean = self.stats["fine"]["band_1"]["mean"]
        # fine_std = self.stats["fine"]["band_1"]["std"]
        # fine = (fine - fine_mean) / (fine_std + 1e-6)
        # fine = fine.astype(np.float32)

        # # Add channel dimension
        # if self.in_channels == 1:
        #     coarse = torch.from_numpy(coarse).unsqueeze(0)
        # else:
        #     coarse = torch.from_numpy(coarse)
        # fine = torch.from_numpy(fine).unsqueeze(0)

        # if self.upsample:
        #     coarse = self.upsample_coarse(coarse, self.scale_factor)

        # if self.transform:
        #     coarse, fine = self.transform((coarse, fine))

        
        if self.legacy:
            return coarse, fine
        else:
            meta = self.parse_metadata(coarse_path)
            date_str = meta['date']            # e.g. "20150331"
            date = torch.tensor(int(date_str), dtype=torch.int32) # YYYYMMDD → int
            # date  = datetime.strptime(date_str, "%Y%m%d")

            # Encode path (as UTF-8 bytes → uint8 tensor)
            path_bytes = coarse_path.encode("utf-8")
            coarse_path = torch.tensor(list(path_bytes), dtype=torch.uint8)
            # coarse_path = meta['filename']

            return coarse, fine, center_lon, center_lat, date, coarse_path
    
    def get_metadata(self, idx):

        """Return only metadata for the given index."""
        coarse_path = self.coarse_files[idx]
        fine_path = self.fine_files[idx]

        metadata_coarse = self.parse_metadata(coarse_path)
        metadata_fine = self.parse_metadata(fine_path)

        return metadata_coarse, metadata_fine


def plot_difference(fine, pred, idx=0, fine_res='', cmap='bwr', title=None, show_colorbar=True):
    """
    Plot the difference map between predicted and ground truth, with ubrmse in title.

    Args:
        pred (Tensor/ndarray): predicted, [B,1,H,W], [1,H,W], or [H,W]
        gt (Tensor/ndarray): ground truth, [B,1,H,W], [1,H,W], or [H,W]
        idx (int): batch index
        fine_res (str): resolution string for title
        cmap (str): colormap, default 'bwr' (blue-white-red)
        title (str): Optional user-supplied title
        show_colorbar (bool): whether to show colorbar
    """

    f_img = fine[idx, 0].numpy()
    p_img = pred[idx, 0].numpy()

    rmse_map = np.sqrt((p_img - f_img) ** 2)
    scalar_rmse = np.sqrt(np.mean((p_img - f_img) ** 2))
    u_rmse = ubrmse(f_img, p_img)

    full_title = f"RMSE Map (per-pixel) [RMSE={scalar_rmse:.4f}, ubRMSE={u_rmse:.4f}]"
    if fine_res:
        full_title += f" | Res: {fine_res}"
    if title:
        full_title = title + " | " + full_title

    plt.figure(figsize=(6, 5))
    im = plt.imshow(rmse_map, cmap=cmap)
    plt.title(full_title)
    plt.axis("off")
    if show_colorbar:
        plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.show()



def plot_sample(coarse, fine, idx=0, title=None, cmap='viridis'):
    """
    Plot coarse and fine resolution tensors side by side.
    
    Args:
        coarse (Tensor): shape [B, 1, H, W]
        fine (Tensor): shape [B, 1, H, W]
        idx (int): Index in the batch to visualize
        title (str): Optional plot title
        cmap (str): Matplotlib colormap
    """
    c_img = coarse[idx, 0].numpy()
    f_img = fine[idx, 0].numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(c_img, cmap=cmap)
    plt.title("Coarse")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(f_img, cmap=cmap)
    plt.title("Fine")
    plt.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()

def plot_predictions_legacy(coarse, fine, pred, idx=0, coarse_res='9km', 
                     fine_res='1km', title='Model Prediction', cmap='viridis'):
    """
    Plot coarse, groundtruth, and predicted images side by side.

    Accepts [B, 1, H, W], [1, H, W], or [H, W] arrays/tensors.
    """
    c_img = coarse[idx, 0].numpy()
    f_img = fine[idx, 0].numpy()
    p_img = pred[idx, 0].numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(c_img, cmap=cmap)
    plt.title(f"Coarse\n({coarse_res})")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(f_img, cmap=cmap)
    plt.title(f"Ground Truth\n({fine_res})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(p_img, cmap=cmap)
    plt.title(f"Predicted\n({fine_res})")
    plt.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()

def plot_predictions(y_9km_dn, pred_dn, y_1km_dn, idx=0,
                     title='Model Prediction', cmap='viridis',
                     vmin=0.0, vmax=0.0):
    """
    Plot 3 panels in physical soil moisture units (m³/m³):

      1. SMAP 9km (y_9km)        — [6, 6]   coarse observation
      2. Predicted 1km            — [54, 54]  model output
      3. Ground Truth 1km (y_1km) — [54, 54]  proxy target

    Parameters
    ----------
    vmin, vmax : float
        User-defined colorbar range. Default 0.0–0.6 m³/m³.
    """
    img_9km  = y_9km_dn[idx, 0]
    img_pred = pred_dn[idx, 0]
    img_1km  = y_1km_dn[idx, 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if vmin == 0.0 and vmax == 0.0:
        # Shared color range across all 4 panels
        all_vals = np.concatenate([img_9km.ravel(), img_pred.ravel(),
                                img_1km.ravel()])
        vmin, vmax = np.nanpercentile(all_vals, [2, 98])

    panels = [
        (img_9km,  "SMAP 9 km\n(Coarse Observation)",  "6×6"),
        (img_pred, "Predicted 1 km\n(Model Output)",    "54×54"),
        (img_1km,  "Ground Truth 1 km\n(Proxy Target)", "54×54"),
    ]

    for ax, (img, label, res) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation='nearest')
        ax.set_title(f"{label}\n[{res}]", fontsize=11)
        ax.axis("off")

    # Dedicated colorbar axis — positioned to the right with no overlap
    # [left, bottom, width, height] in figure coordinates
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
    fig.colorbar(im, cax=cbar_ax, label="Soil Moisture (m³/m³)")

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    fig.subplots_adjust(top=0.85, bottom=0.05, left=0.02, right=0.91,
                        wspace=0.08)
    plt.show()


# def plot_predictions_4x(x_1km_dn, y_9km_dn, pred_dn, y_1km_dn, idx=0,
#                      title='Model Prediction', cmap='viridis'):
def plot_predictions_4x(x_1km_dn, y_9km_dn, pred_dn, y_1km_dn, idx=0,
                     title='Model Prediction', cmap='Spectral'):
    """
    Plot 4 panels showing all components in physical soil moisture units (m³/m³):

      1. SMAP 9km (y_9km)        — [B, 1, 6, 6]   coarse observation
      2. SMAP 1km-R (x_1km b0)   — [B, 1, 54, 54]  bilinear interpolated input
      3. Predicted 1km            — [B, 1, 54, 54]  model output
      4. Ground Truth 1km (y_1km) — [B, 1, 54, 54]  proxy target

    All values are denormalized — colorbar shows real soil moisture.
    A shared colorbar range (vmin/vmax) is used across all panels so
    colors are directly comparable.
    """
    img_9km  = y_9km_dn[idx, 0]     # [6, 6]
    img_1kr  = x_1km_dn[idx, 0]     # [54, 54]
    img_pred = pred_dn[idx, 0]      # [54, 54]
    img_1km  = y_1km_dn[idx, 0]     # [54, 54]

    # Shared color range across all 4 panels
    all_vals = np.concatenate([img_9km.ravel(), img_1kr.ravel(),
                               img_pred.ravel(), img_1km.ravel()])
    vmin, vmax = np.nanpercentile(all_vals, [2, 98])
    # vmin, vmax = min(vmin, 0.1), max(vmax, 0.5) 

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # panels = [
    #     (img_9km,  "SMAP 9 km\n(Coarse Observation)",    "6×6"),
    #     (img_1kr,  "SMAP 1 km-R\n(Bilinear Interpolated)", "54×54"),
    #     (img_pred, "Predicted 1 km\n(Model Output)",       "54×54"),
    #     (img_1km,  "High Resolution Reference (1km)",    "54×54"),
    # ]

    # for ax, (img, label, res) in zip(axes, panels):
    #     im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
    #                     interpolation='nearest')
    #     ax.set_title(f"{label}\n[{res}]", fontsize=11)
    #     ax.axis("off")

    panels = [
        (img_9km,  "Coarse Resolution Input (9km)"),
        (img_1kr,  "Coarse Interpolated (1km-R)"),
        (img_pred, "Predicted (1km)"),
        (img_1km,  "High Resolution Reference (1km)"),
    ]

    for ax, (img, label) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation='nearest')
        ax.set_title(f"{label}", fontsize=11)
        ax.axis("off")

    # # Single shared colorbar
    # cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    # cbar.set_label("Soil Moisture (m³/m³)", fontsize=11)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
    fig.colorbar(im, cax=cbar_ax, label="Soil Moisture (m³/m³)")

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    # plt.tight_layout()
    fig.subplots_adjust(top=0.85, bottom=0.05, left=0.02, right=0.92,
                        wspace=0.08)
    plt.show()


def plot_predictions_4x_new(
    y_9km_dn,
    pred_dn,
    y_1km_dn,
    lst_dn,
    idx=0,
    title="",
    cmap_sm="Spectral",
    cmap_lst="inferno",
    save_path=None,
    display_dpi=100,
    save_dpi=300,
):
    """
    4-panel plot:

      1. SMAP 9km
      2. Predicted 1km
      3. SMAP 1km
      4. MODIS LST

    First 3 panels share one common soil-moisture colorbar on the LEFT.
    LST has its own colorbar on the RIGHT.

    Display stays readable.
    Saved figure is exported at 300 dpi.
    """

    img_9km  = y_9km_dn[idx, 0]
    img_pred = pred_dn[idx, 0]
    img_1km  = y_1km_dn[idx, 0]
    img_lst  = lst_dn[idx, 0]

    # Shared soil-moisture color range
    sm_vals = np.concatenate([
        img_9km.ravel(),
        img_pred.ravel(),
        img_1km.ravel(),
    ])
    sm_vmin, sm_vmax = np.nanpercentile(sm_vals, [2, 98])

    # LST color range
    lst_vmin, lst_vmax = np.nanpercentile(img_lst, [2, 98])

    with plt.rc_context({
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 10,
    }):

        fig, axes = plt.subplots(
            1, 4,
            figsize=(12, 3.4),
            dpi=display_dpi
        )

        # Tighter / more standard spacing between panels
        fig.subplots_adjust(
            left=0.10,
            right=0.90,
            bottom=0.14,
            top=0.80,
            wspace=0.10
        )

        # --- Plot first three SM panels ---
        sm_panels = [
            (img_9km,  "SMAP SM 9km"),
            (img_pred, "Predicted SM 1km"),
            (img_1km,  "SMAP SM 1km"),
        ]

        for ax, (img, label) in zip(axes[:3], sm_panels):
            im_sm = ax.imshow(
                img,
                cmap=cmap_sm,
                vmin=sm_vmin,
                vmax=sm_vmax,
                interpolation="nearest",
                aspect="equal"
            )
            ax.set_title(label, pad=6)
            ax.axis("off")

        # --- Plot LST panel ---
        im_lst = axes[3].imshow(
            img_lst,
            cmap=cmap_lst,
            vmin=lst_vmin,
            vmax=lst_vmax,
            interpolation="nearest",
            aspect="equal"
        )
        axes[3].set_title("MODIS LST 1km", pad=6)
        axes[3].axis("off")

        # ------------------------------------------------------------------
        # Add colorbars with SMALLER gap from the plots
        # ------------------------------------------------------------------
        pos_left  = axes[0].get_position()
        pos_right = axes[3].get_position()

        cbar_width = 0.012
        cbar_gap = 0.006   # smaller gap than before

        # Left common colorbar for soil moisture
        cbar_sm_ax = fig.add_axes([
            pos_left.x0 - cbar_gap - cbar_width,
            pos_left.y0,
            cbar_width,
            pos_left.height
        ])
        cbar_sm = fig.colorbar(im_sm, cax=cbar_sm_ax)
        cbar_sm.set_label("Soil Moisture (m³/m³)", fontsize=8, labelpad=6)
        cbar_sm.ax.tick_params(labelsize=8)
        cbar_sm_ax.yaxis.set_ticks_position("left")
        cbar_sm_ax.yaxis.set_label_position("left")

        # Right colorbar for LST
        cbar_lst_ax = fig.add_axes([
            pos_right.x1 + cbar_gap,
            pos_right.y0,
            cbar_width,
            pos_right.height
        ])
        cbar_lst = fig.colorbar(im_lst, cax=cbar_lst_ax)
        cbar_lst.set_label("LST (K)", fontsize=8, labelpad=6)
        cbar_lst.ax.tick_params(labelsize=8)
        cbar_lst_ax.yaxis.set_ticks_position("right")
        cbar_lst_ax.yaxis.set_label_position("right")

        if title:
            fig.suptitle(title, fontsize=10, fontweight="bold", y=0.93)

        if save_path is not None:
            fig.savefig(
                save_path,
                dpi=save_dpi,
                bbox_inches="tight",
                pad_inches=0.05
            )

        plt.show()

def plot_predictions_5x(
    y_9km_dn,
    pred_dn,
    y_1km_dn,
    lst_dn,
    dem_dn,
    idx=0,
    title="Model Prediction",
    cmap_sm="Spectral",
    cmap_lst="inferno",
    cmap_dem="terrain",
    save_path=None,
    display_dpi=100,
    save_dpi=300,
):
    """
    5-panel plot:

      1. SMAP 9km                 Soil moisture, m³/m³
      2. Predicted 1km            Soil moisture, m³/m³
      3. SMAP 1km                 Soil moisture, m³/m³
      4. MODIS LST                Kelvin
      5. DEM                      meters

    First 3 panels share one common soil-moisture colorbar on the LEFT.
    LST has its own colorbar.
    DEM has its own colorbar on the RIGHT.

    Display dpi stays readable.
    Saved figure is exported at 300 dpi.
    """

    img_9km  = y_9km_dn[idx, 0]
    img_pred = pred_dn[idx, 0]
    img_1km  = y_1km_dn[idx, 0]
    img_lst  = lst_dn[idx, 0]
    img_dem  = dem_dn[idx, 0]

    sm_vals = np.concatenate([
        img_9km.ravel(),
        img_pred.ravel(),
        img_1km.ravel(),
    ])

    sm_vmin, sm_vmax = np.nanpercentile(sm_vals, [2, 98])
    lst_vmin, lst_vmax = np.nanpercentile(img_lst, [2, 98])
    dem_vmin, dem_vmax = np.nanpercentile(img_dem, [2, 98])

    with plt.rc_context({
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 10,
    }):

        fig = plt.figure(figsize=(14.5, 3.4), dpi=display_dpi)

        bottom = 0.18
        height = 0.58

        img_w = 0.135
        gap = 0.025
        cbar_w = 0.010

        x_start = 0.085

        ax1 = fig.add_axes([x_start + 0 * (img_w + gap), bottom, img_w, height])
        ax2 = fig.add_axes([x_start + 1 * (img_w + gap), bottom, img_w, height])
        ax3 = fig.add_axes([x_start + 2 * (img_w + gap), bottom, img_w, height])
        ax4 = fig.add_axes([x_start + 3 * (img_w + gap), bottom, img_w, height])

        # LST colorbar between LST and DEM
        cbar_lst_ax = fig.add_axes([
            x_start + 4 * (img_w + gap) - 0.015,
            bottom,
            cbar_w,
            height
        ])

        ax5 = fig.add_axes([
            x_start + 4 * (img_w + gap) + 0.010,
            bottom,
            img_w,
            height
        ])

        # Soil moisture colorbar on far left
        cbar_sm_ax = fig.add_axes([0.040, bottom, cbar_w, height])

        # DEM colorbar on far right
        cbar_dem_ax = fig.add_axes([0.940, bottom, cbar_w, height])

        sm_panels = [
            (ax1, img_9km,  "SMAP 9km"),
            (ax2, img_pred, "Predicted 1km"),
            (ax3, img_1km,  "SMAP 1km"),
        ]

        for ax, img, label in sm_panels:
            im_sm = ax.imshow(
                img,
                cmap=cmap_sm,
                vmin=sm_vmin,
                vmax=sm_vmax,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_title(label, fontsize=10, pad=6)
            ax.axis("off")

        im_lst = ax4.imshow(
            img_lst,
            cmap=cmap_lst,
            vmin=lst_vmin,
            vmax=lst_vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax4.set_title("MODIS LST", fontsize=10, pad=6)
        ax4.axis("off")

        im_dem = ax5.imshow(
            img_dem,
            cmap=cmap_dem,
            vmin=dem_vmin,
            vmax=dem_vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax5.set_title("DEM", fontsize=10, pad=6)
        ax5.axis("off")

        # Common soil moisture colorbar
        cbar_sm = fig.colorbar(im_sm, cax=cbar_sm_ax)
        cbar_sm.set_label("Soil Moisture (m³/m³)", fontsize=8, labelpad=6)
        cbar_sm.ax.tick_params(labelsize=8)
        cbar_sm_ax.yaxis.set_ticks_position("left")
        cbar_sm_ax.yaxis.set_label_position("left")

        # LST colorbar
        cbar_lst = fig.colorbar(im_lst, cax=cbar_lst_ax)
        cbar_lst.set_label("LST (K)", fontsize=8, labelpad=6)
        cbar_lst.ax.tick_params(labelsize=8)

        # DEM colorbar
        cbar_dem = fig.colorbar(im_dem, cax=cbar_dem_ax)
        cbar_dem.set_label("Elevation (m)", fontsize=8, labelpad=6)
        cbar_dem.ax.tick_params(labelsize=8)
        cbar_dem_ax.yaxis.set_ticks_position("right")
        cbar_dem_ax.yaxis.set_label_position("right")

        if title:
            fig.suptitle(
                title,
                fontsize=10,
                fontweight="bold",
                y=0.96,
            )

        if save_path is not None:
            fig.savefig(
                save_path,
                dpi=save_dpi,
                bbox_inches="tight",
                pad_inches=0.08,
            )

        plt.show()

def plot_predictions_6x(
    x_1km_dn, y_9km_dn, pred_dn, y_1km_dn,
    lst_dn, dem_dn,
    idx=0,
    title="Model Prediction",
    cmap_sm="Spectral",
    cmap_lst="inferno",
    cmap_dem="terrain"
):
    """
    Plot 6 panels:

      1. Coarse Resolution Input 9km      Soil moisture, m³/m³
      2. Coarse Interpolated 1km-R        Soil moisture, m³/m³
      3. Predicted 1km                    Soil moisture, m³/m³
      4. High Resolution Reference 1km    Soil moisture, m³/m³
      5. MODIS LST                        Kelvin
      6. DEM                              meters

    First 4 panels share one soil-moisture colorbar on the left.
    LST and DEM each have their own colorbar on the left of their panel.
    """

    img_9km  = y_9km_dn[idx, 0]
    img_1kr  = x_1km_dn[idx, 0]
    img_pred = pred_dn[idx, 0]
    img_1km  = y_1km_dn[idx, 0]
    img_lst  = lst_dn[idx, 0]
    img_dem  = dem_dn[idx, 0]

    sm_vals = np.concatenate([
        img_9km.ravel(),
        img_1kr.ravel(),
        img_pred.ravel(),
        img_1km.ravel()
    ])

    sm_vmin, sm_vmax = np.nanpercentile(sm_vals, [2, 98])
    lst_vmin, lst_vmax = np.nanpercentile(img_lst, [2, 98])
    dem_vmin, dem_vmax = np.nanpercentile(img_dem, [2, 98])

    fig, axes = plt.subplots(1, 6, figsize=(28, 5))

    sm_panels = [
        (img_9km,  "Coarse Resolution Input (9km)"),
        (img_1kr,  "Coarse Interpolated (1km-R)"),
        (img_pred, "Predicted (1km)"),
        (img_1km,  "High Resolution Reference (1km)")
    ]

    for ax, (img, label) in zip(axes[:4], sm_panels):
        im_sm = ax.imshow(
            img,
            cmap=cmap_sm,
            vmin=sm_vmin,
            vmax=sm_vmax,
            interpolation="nearest"
        )
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    im_lst = axes[4].imshow(
        img_lst,
        cmap=cmap_lst,
        vmin=lst_vmin,
        vmax=lst_vmax,
        interpolation="nearest"
    )
    axes[4].set_title("MODIS LST (K)", fontsize=11)
    axes[4].axis("off")

    im_dem = axes[5].imshow(
        img_dem,
        cmap=cmap_dem,
        vmin=dem_vmin,
        vmax=dem_vmax,
        interpolation="nearest"
    )
    axes[5].set_title("DEM (m)", fontsize=11)
    axes[5].axis("off")

    fig.subplots_adjust(
        top=0.85,
        bottom=0.05,
        left=0.08,
        right=0.98,
        wspace=0.12
    )

    # Shared soil moisture colorbar on the far left
    cbar_sm_ax = fig.add_axes([0.025, 0.18, 0.012, 0.58])
    cbar_sm = fig.colorbar(im_sm, cax=cbar_sm_ax)
    cbar_sm.set_label("Soil Moisture (m³/m³)", fontsize=11)
    cbar_sm_ax.yaxis.set_ticks_position("left")
    cbar_sm_ax.yaxis.set_label_position("left")

    # LST colorbar on left side of LST panel
    cbar_lst = fig.colorbar(im_lst, ax=axes[4], fraction=0.046, pad=0.04)
    cbar_lst.set_label("LST (K)", fontsize=11)

    # DEM colorbar on left side of DEM panel
    cbar_dem = fig.colorbar(im_dem, ax=axes[5], fraction=0.046, pad=0.04)
    cbar_dem.set_label("Elevation (m)", fontsize=11)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.show()

########################################################################################
## Metrics
########################################################################################

def psnr(target, ref):
    target_data = target.astype(np.float32)
    ref_data = ref.astype(np.float32)

    mse = np.mean((ref_data - target_data) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * math.log10(1.0 / math.sqrt(mse))  # assuming values in [0, 1]

def mse_metric(target, ref):
    return np.mean((ref.astype('float32') - target.astype('float32')) ** 2)

def bias(target, ref):
    return np.mean(target) - np.mean(ref)

def ubrmse(target, ref):
    x = target - np.mean(target)
    y = ref - np.mean(ref)
    return np.sqrt(np.mean((x - y) ** 2))

def pearson_corr(target, ref):
    """
    Computes the Pearson correlation coefficient between target and ref.
    Returns just the correlation value (not the p-value).
    """
    return pearsonr(target.flatten(), ref.flatten())[0]


def compare_images(target, ref, data_range=None):
    """
    Args:
        target: NumPy array [H, W] — prediction (or baseline)
        ref:    NumPy array [H, W] — ground truth
        data_range: max - min of the reference signal (for PSNR)
    Returns:
        dict with PSNR, MSE, Bias, ubRMSE, Pearson Correlation
    """
    if data_range is None:
        data_range = ref.max() - ref.min()
        if data_range < 1e-6:
            data_range = 1.0  # fallback for constant patches

    return {
        "PSNR": peak_signal_noise_ratio(ref, target, data_range=data_range),
        "MSE": mse_metric(target, ref),
        "Bias": bias(target, ref),
        "ubRMSE": ubrmse(target, ref),
        "PearsonR": pearson_corr(target, ref),
    }

def fisher_z(r):
    r = np.array(r)
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def inverse_fisher_z(z):
    return np.tanh(z)

def mean_r(correlations):
    z_values = fisher_z(correlations)
    mean_z = np.nanmean(z_values)
    return inverse_fisher_z(mean_z)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# logger.py

class LossLogger:
    def __init__(self, log_path="training_log.json", overwrite=False):
        self.log_path = log_path
        self.logs = {
            "train_loss": [],
            "train_psnr": [],
            "val_loss": [],
            "val_psnr": []
        }
        if not overwrite and os.path.exists(log_path):
            with open(log_path, "r") as f:
                self.logs = json.load(f)
        else:
            self._save()

    def log(self, train_loss, train_psnr, val_loss, val_psnr):
        self.logs["train_loss"].append(train_loss)
        self.logs["train_psnr"].append(train_psnr)
        self.logs["val_loss"].append(val_loss)
        self.logs["val_psnr"].append(val_psnr)
        self._save()

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.logs, f, indent=2)

    def get_logs(self):
        return self.logs

def logs_display(log_path):

    logger = LossLogger(log_path)
    logs = logger.get_logs()
    epochs = range(1, len(logs["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, logs["train_loss"], label="Train Loss")
    plt.plot(epochs, logs["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Train/Val Loss")

    plt.figure()
    plt.plot(epochs, logs["train_psnr"], label="Train PSNR")
    plt.plot(epochs, logs["val_psnr"], label="Val PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.title("Train/Val PSNR") 
    plt.show()

def compute_stats_and_plot(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Extract the sub-region IDs (the last character before '.tif') from file names
    region_ids = []
    for file in files:
        if file.endswith(".tif"):
            try:
                # Extracting the sub-region ID
                region_id = int(file.split('_')[-1].split(".")[0])
                region_ids.append(region_id)
            except ValueError:
                print(f"Warning: Could not extract region ID from {file}")

    # Count occurrences of each region ID
    region_counts = Counter(region_ids)

    # Total number of files
    total_files = sum(region_counts.values())

    # Compute percentages of each region
    region_percentages = {region: (count / total_files) * 100 for region, count in region_counts.items()}

    # Print statistics
    print("Sub-region Statistics:")
    for region, count in region_counts.items():
        print(f"Region {region}: {count} files ({region_percentages[region]:.2f}%)")

    # Plot statistics
    regions = list(region_counts.keys())
    counts = list(region_counts.values())
    percentages = [region_percentages[r] for r in regions]

    # Bar plot for counts
    plt.figure(figsize=(12, 6))
    plt.bar(regions, counts, color='steelblue', alpha=0.8)
    plt.xlabel("Region ID")
    plt.ylabel("Count")
    plt.title("File Count per Sub-region")
    plt.xticks(regions)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

# # Example folder path (Replace this with the actual path to your folder containing TIFF files)
# folder_path = "D:/SM-DeepLearning/datasets/Central valley/training/9km-r"

# # Compute stats and generate plots
# compute_stats_and_plot(folder_path)

def calculate_dataset_stats(coarse_dir, fine_dir, coarse_bands=8, fine_bands=1, json_path="dataset_stats.json"):
    """
    Calculate the mean, standard deviation, and class information for coarse and fine datasets.

    Args:
        coarse_dir (str): Path to the directory containing all coarse-resolution files.
        fine_dir (str): Path to the directory containing all fine-resolution files.
        coarse_bands (int): Number of bands in the coarse dataset.
        fine_bands (int): Number of bands in the fine dataset (usually 1).
        json_path (str): Path to save or load the stats JSON file.

    Returns:
        stats (dict): A dictionary with stats for coarse and fine datasets.
    """

    # Initialize accumulators for coarse and fine datasets (for mean, std calculations)
    coarse_sums = np.zeros(coarse_bands, dtype=np.float64)
    coarse_squared_sums = np.zeros(coarse_bands, dtype=np.float64)
    coarse_pixel_counts = np.zeros(coarse_bands, dtype=np.int64)

    fine_sums = np.zeros(fine_bands, dtype=np.float64)
    fine_squared_sums = np.zeros(fine_bands, dtype=np.float64)
    fine_pixel_counts = np.zeros(fine_bands, dtype=np.int64)

    # Categorical band-specific storage for classes (bands 3 and 4)
    categorical_classes = {3: set(), 4: set()}  # Use sets to collect unique classes

    # Process coarse-resolution files
    raster_files_coarse = glob.glob(f"{coarse_dir}/*.tif")
    for raster_file in tqdm(raster_files_coarse, desc="Processing coarse files"):
        with rasterio.open(raster_file) as src:
            for i in range(coarse_bands):  # Coarse bands can have multiple channels
                band = src.read(i + 1).astype(np.float64)
                band = band[~np.isnan(band)]  # Remove NaN values

                # Handle categorical bands (e.g., band 3 and 4)
                if (i + 1) in categorical_classes:
                    categorical_classes[i + 1].update(np.unique(band.astype(np.int32)))

                # Continuous band calculation
                coarse_sums[i] += band.sum()
                coarse_squared_sums[i] += np.square(band).sum()
                coarse_pixel_counts[i] += band.size

    # Process fine-resolution files
    raster_files_fine = glob.glob(f"{fine_dir}/*.tif")
    for raster_file in tqdm(raster_files_fine, desc="Processing fine files"):
        with rasterio.open(raster_file) as src:
            for i in range(fine_bands):  # Fine bands typically have one channel
                band = src.read(i + 1).astype(np.float64)
                band = band[~np.isnan(band)]  # Remove NaN values

                fine_sums[i] += band.sum()
                fine_squared_sums[i] += np.square(band).sum()
                fine_pixel_counts[i] += band.size

    # Calculate stats (mean and std) for continuous bands
    coarse_means = coarse_sums / coarse_pixel_counts
    coarse_stds = np.sqrt(coarse_squared_sums / coarse_pixel_counts - np.square(coarse_means))
    fine_means = fine_sums / fine_pixel_counts
    fine_stds = np.sqrt(fine_squared_sums / fine_pixel_counts - np.square(fine_means))

    # Prepare dictionary with stats
    stats = {
        "coarse": {},
        "fine": {},
    }

    # Add mean and std for continuous bands to coarse stats
    for i in range(coarse_bands):
        if (i + 1) in categorical_classes:
            # For categorical bands, save the number of classes, unique class values, and mapping
            classes = sorted(map(int, categorical_classes[i + 1]))  # Convert np.int32 to Python int
            mapping = {value: idx for idx, value in enumerate(classes)}  # Create mapping from raw class to contiguous indices
            stats["coarse"][f"band_{i+1}"] = {
                "num_classes": len(classes),
                "classes": classes,
                "mapping": mapping  # Save the mapping
            }
        else:
            # For continuous bands, save mean and std
            stats["coarse"][f"band_{i+1}"] = {"mean": float(coarse_means[i]), "std": float(coarse_stds[i])}

    # Add mean and std for fine bands
    for i in range(fine_bands):
        stats["fine"][f"band_{i+1}"] = {"mean": float(fine_means[i]), "std": float(fine_stds[i])}

    # Save stats to a JSON file
    with open(json_path, "w") as fp:
        json.dump(stats, fp)

    return stats

# Example usage
# coarse_dir = "training/spatial/0/train/9km-m"  # Replace with the path to your dataset
# fine_dir = "training/spatial/0/train/1km"
# stats = calculate_dataset_stats(coarse_dir, fine_dir, coarse_bands=8, fine_bands=1)

# # Check the resulting stats JSON
# print(stats)

def calculate_dataset_stats_cv(
    x1_dir,
    y9_dir,
    y1_dir,
    x1_bands=5,
    json_path="dataset_stats.json"
):
    """
    Compute dataset statistics for SR soil moisture framework.

    Inputs:
      - x1: 1km multiband input (5 bands, last 2 categorical)
      - y9: native SMAP_P_E (1 band, 9km)
      - y1: proxy SMAP_1km (1 band)

    Output:
      JSON stats usable directly by SREncDataset
    """

    stats = {
        "x1": {},
        "y9": {},
        "y1": {}
    }

    # -----------------------------
    # Helper accumulators
    # -----------------------------
    def init_acc(n):
        return (
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.int64)
        )

    fr1_sum, fr1_sq, fr1_cnt = init_acc(x1_bands)
    coarse_sum, coarse_sq, coarse_cnt = init_acc(1)
    fine_sum, fine_sq, fine_cnt = init_acc(1)

    # categorical collectors (last 2 bands)
    categorical_classes = {
        x1_bands - 1: set(),
        x1_bands: set()
    }

    # -----------------------------
    # Process FR1 (1km multiband)
    # -----------------------------
    fr1_files = glob.glob(f"{x1_dir}/*.tif")
    for f in tqdm(fr1_files, desc="Processing fr1 (1km inputs)"):
        with rasterio.open(f) as src:
            for i in range(x1_bands):
                band = src.read(i + 1).astype(np.float64)
                band = band[~np.isnan(band)]

                if band.size == 0:
                    continue

                band_idx = i + 1

                if band_idx in categorical_classes:
                    categorical_classes[band_idx].update(
                        np.unique(band.astype(np.int32))
                    )
                else:
                    fr1_sum[i] += band.sum()
                    fr1_sq[i] += np.square(band).sum()
                    fr1_cnt[i] += band.size

    # -----------------------------
    # Process COARSE (9km SMAP_P_E)
    # -----------------------------
    coarse_files = glob.glob(f"{y9_dir}/*.tif")
    for f in tqdm(coarse_files, desc="Processing coarse (9km)"):
        with rasterio.open(f) as src:
            band = src.read(1).astype(np.float64)
            band = band[~np.isnan(band)]
            if band.size == 0:
                continue
            coarse_sum[0] += band.sum()
            coarse_sq[0] += np.square(band).sum()
            coarse_cnt[0] += band.size

    # -----------------------------
    # Process FINE (1km proxy)
    # -----------------------------
    fine_files = glob.glob(f"{y1_dir}/*.tif")
    for f in tqdm(fine_files, desc="Processing fine (1km target)"):
        with rasterio.open(f) as src:
            band = src.read(1).astype(np.float64)
            band = band[~np.isnan(band)]
            if band.size == 0:
                continue
            fine_sum[0] += band.sum()
            fine_sq[0] += np.square(band).sum()
            fine_cnt[0] += band.size

    # -----------------------------
    # Finalize FR1 stats
    # -----------------------------
    fr1_bands_count = 0

    for i in range(x1_bands):
        band_idx = i + 1
        if band_idx in categorical_classes:
            classes = sorted(map(int, categorical_classes[band_idx]))
            mapping = {v: k for k, v in enumerate(classes)}
            num_classes = len(classes)
            stats["x1"][f"band_{band_idx}"] = {
                "type": "categorical",
                "num_classes": num_classes,
                "classes": classes,
                "mapping": mapping
            }
            fr1_bands_count += num_classes
        else:
            mean = fr1_sum[i] / fr1_cnt[i]
            std = np.sqrt(fr1_sq[i] / fr1_cnt[i] - mean**2)
            stats["x1"][f"band_{band_idx}"] = {
                "type": "continuous",
                "mean": float(mean),
                "std": float(std)
            }
            fr1_bands_count += 1

    # -----------------------------
    # Coarse & fine stats
    # -----------------------------
    for name, s, sq, c in [
        ("y9", coarse_sum, coarse_sq, coarse_cnt),
        ("y1", fine_sum, fine_sq, fine_cnt)
    ]:
        mean = s[0] / c[0]
        std = np.sqrt(sq[0] / c[0] - mean**2)
        stats[name]["band_1"] = {
            "type": "continuous",
            "mean": float(mean),
            "std": float(std)
        }

    # -----------------------------
    # Save
    # -----------------------------
    if json_path:
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)

    return stats, fr1_bands_count

# Example usage:
# coarse_dir = "training/temporal/train/9km"  # Replace with the path to your dataset
# fine_dir = "training/temporal/train/1km"
# f1_dir = "training/temporal/train/1km-m"
# stats_path = "data_stats1.json"

# stats, fr1_bands = calculate_dataset_stats_cv(
#     x1_dir=f1_dir,
#     y9_dir=coarse_dir,
#     y1_dir=fine_dir,
#     x1_bands=5,
#     json_path=stats_path
# )

# print("Dataset statistics saved to", stats)
# print("Number of FR1 bands:", fr1_bands)
