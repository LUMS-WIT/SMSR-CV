import os
import glob
import re
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path


def get_raster_value_at_point(tiff_path, lat, lon):
    """
    Extract the pixel value from a GeoTIFF at a given lat/lon.
    Returns NaN if the point is outside the raster or the value is nodata.
    """
    try:
        with rasterio.open(tiff_path) as src:
            row, col = src.index(lon, lat)
            if row < 0 or col < 0 or row >= src.height or col >= src.width:
                return np.nan
            val = src.read(1)[row, col]
            if src.nodata is not None and val == src.nodata:
                return np.nan
            return float(val)
    except Exception as e:
        print(f"    ⚠ Error reading {tiff_path}: {e}")
        return np.nan


def parse_raster_date(filename):
    """
    Extract the date string from a raster filename.
    Handles both SMAP and model formats:
      SMAP-E_1km_AM_20210315_001.tiff  -> '2021-03-15'
      pred_1km_AM_20210315_001.tiff    -> '2021-03-15'
    """
    match = re.search(r'AM_(\d{8})_', filename)
    if match:
        d = match.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    return None


def build_raster_date_index(raster_dir):
    """
    Scan a raster directory and build a dict: date_str -> filepath.
    """
    date_to_file = {}
    if not os.path.isdir(raster_dir):
        print(f"  ⚠ Raster directory not found: {raster_dir}")
        return date_to_file
    for f in sorted(os.listdir(raster_dir)):
        if f.lower().endswith(('.tif', '.tiff')):
            date_str = parse_raster_date(f)
            if date_str:
                date_to_file[date_str] = os.path.join(raster_dir, f)
    return date_to_file


def parse_ismn_filename(filename):
    """
    Parse an ISMN CSV filename to extract lat and lon.
    """
    name = filename.replace(".csv", "")
    parts = name.rsplit("_", 2)
    try:
        lat = float(parts[-2])
        lon = float(parts[-1])
        return lat, lon
    except (ValueError, IndexError):
        return None, None


def extract_and_merge_raster(
    ismn_dir="/content/ismn_sub",
    raster_root="/content/raster",
    model_root="/content/model",          # NEW: root for y1km predictions
    output_dir="/content/ismn_sub_with_smap",
    resolutions=("1km", "9km")
):
    """
    For each ISMN CSV:
      1. Extract SMAP 1km and 9km values (from raster_root)
      2. Extract model y1km values (from model_root/out/)
      3. Add '1km', '9km', 'y1km' columns
      4. Print per-file overlap counts + three-way overlap (swc & 1km & y1km)
      5. Print overall sum of three-way overlaps at the end
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Build date indices for SMAP resolutions ---
    print("Building raster date indices...")
    raster_indices = {}
    for res in resolutions:
        res_dir = os.path.join(raster_root, res)
        raster_indices[res] = build_raster_date_index(res_dir)
        print(f"  {res}: {len(raster_indices[res])} raster files indexed")

    # --- Build date index for y1km (model predictions) ---
    y1km_dir = model_root
    raster_indices["y1km"] = build_raster_date_index(y1km_dir)
    print(f"  y1km: {len(raster_indices['y1km'])} raster files indexed")
    assert len(raster_indices["y1km"]) > 0, "No y1km prediction rasters found! Check model_root path."

    # All columns to extract (SMAP + model)
    all_columns = list(resolutions) + ["y1km"]

    print(f"\n{'='*70}")
    total_csvs = 0
    grand_total_three_way = 0  # overall sum of three-way overlaps

    for subroi_folder in sorted(os.listdir(ismn_dir)):
        subroi_path = os.path.join(ismn_dir, subroi_folder)
        if not os.path.isdir(subroi_path) or not subroi_folder.startswith("subroi_"):
            continue

        print(f"\n{'='*70}")
        print(f"Processing: {subroi_folder}")
        print(f"{'='*70}")

        out_subroi_dir = os.path.join(output_dir, subroi_folder)
        os.makedirs(out_subroi_dir, exist_ok=True)

        for csv_file in sorted(os.listdir(subroi_path)):
            if not csv_file.endswith(".csv"):
                continue

            csv_path = os.path.join(subroi_path, csv_file)
            lat, lon = parse_ismn_filename(csv_file)

            if lat is None or lon is None:
                print(f"\n  ⚠ Could not parse lat/lon from: {csv_file}, skipping.")
                continue

            print(f"\n  📄 {csv_file}")
            print(f"     Lat: {lat}, Lon: {lon}")

            df = pd.read_csv(csv_path)

            # Initialize all new columns to NaN
            for col in all_columns:
                df[col] = np.nan

            overlap_counts = {col: 0 for col in all_columns}

            for idx, row in df.iterrows():
                date_str = row["date"]

                for col in all_columns:
                    if date_str in raster_indices[col]:
                        tiff_path = raster_indices[col][date_str]
                        val = get_raster_value_at_point(tiff_path, lat, lon)
                        if not np.isnan(val):
                            df.at[idx, col] = val
                            overlap_counts[col] += 1

            # --- Per-column overlap stats ---
            print(f"     ISMN dates: {len(df)}")
            for col in all_columns:
                total_rasters = len(raster_indices[col])
                matched = overlap_counts[col]
                pct = matched / len(df) * 100 if len(df) > 0 else 0
                print(f"     {col}: {matched}/{len(df)} dates matched "
                      f"({pct:.1f}% overlap, "
                      f"{total_rasters} rasters available)")

            # --- Three-way overlap: rows where swc AND 1km AND y1km all exist ---
            three_way_mask = (
                df["swc"].notna() &
                df["1km"].notna() &
                df["y1km"].notna()
            )
            three_way_count = three_way_mask.sum()
            grand_total_three_way += three_way_count
            print(f"     🔗 Three-way overlap (swc & 1km & y1km): {three_way_count}/{len(df)} rows")

            # Save
            out_path = os.path.join(out_subroi_dir, csv_file)
            df.to_csv(out_path, index=False)
            total_csvs += 1
            print(f"     ✅ Saved: {subroi_folder}/{csv_file}")

    print(f"\n{'='*70}")
    print(f"Total merged CSVs: {total_csvs}")
    print(f"Overall three-way overlap (swc & 1km & y1km) across ALL files: {grand_total_three_way}")
    print(f"Output directory: {output_dir}")


# --- Run ---
extract_and_merge_raster(
    ismn_dir="Dataset\\ismn_data_subroi",
    raster_root= "Dataset",         # contains 1km/ and 9km/ subfolders with SMAP rasters# contains 1km/ and 9km/ subfolders
    model_root="Output\\out_full",           # contains out/ subfolder with pred_1km_AM_*.tiff
    output_dir="Output\\sm_full",
    resolutions=("1km", "9km")
)
