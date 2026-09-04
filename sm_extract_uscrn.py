import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
SITES_CSV = Path(r"D:\SM-DeepLearning\SMSR-CV\Output\ISMN_USCRN_updated.csv")

RASTER_DIR = Path(
    r"D:\SM-DeepLearning\SMSR-CV\Output\out_full_uscrn"
)

OUTPUT_DIR = Path(
    r"D:\SM-DeepLearning\SMSR-CV\Output\sm_full_uscrn"
)


def parse_raster_date(filename):
    """
    Extract YYYY-MM-DD from prediction filenames such as:
    pred_1km_AM_20210315_001.tif
    """
    match = re.search(r"(?<!\d)(\d{8})(?!\d)", filename)

    if not match:
        return None

    date_raw = match.group(1)

    try:
        return pd.to_datetime(date_raw, format="%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def extract_value_from_open_raster(src, lat, lon):
    """
    Extract a valid raster value at WGS84 latitude/longitude.

    Returns None when:
    - the point lies outside the raster;
    - the point is nodata/masked;
    - the raster value is NaN or infinite.
    """
    try:
        # Convert WGS84 coordinates to the raster CRS when necessary.
        if src.crs is not None:
            x_values, y_values = transform(
                "EPSG:4326",
                src.crs,
                [float(lon)],
                [float(lat)],
            )
            x, y = x_values[0], y_values[0]
        else:
            # Assumes coordinates are already longitude/latitude.
            x, y = float(lon), float(lat)

        bounds = src.bounds
        if not (
            bounds.left <= x <= bounds.right
            and bounds.bottom <= y <= bounds.top
        ):
            return None

        # Reads only one pixel rather than loading the full raster.
        sampled_value = next(src.sample([(x, y)], indexes=1, masked=True))

        if np.any(np.ma.getmaskarray(sampled_value)):
            return None

        value = float(sampled_value[0])

        if not np.isfinite(value):
            return None

        if src.nodata is not None and np.isclose(value, src.nodata):
            return None

        return value

    except Exception as error:
        print(f"    Could not sample point ({lat}, {lon}): {error}")
        return None


def safe_filename_component(value):
    """Keep readable names while removing characters invalid in Windows filenames."""
    value = str(value).strip()
    return re.sub(r'[<>:"/\\|?*]', "_", value)


def format_coordinate(value):
    """Format coordinates cleanly for output filenames."""
    value = float(value)
    return f"{value:.10f}".rstrip("0").rstrip(".")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(SITES_CSV)

    required_columns = {"id", "network", "station", "lon", "lat"}
    missing_columns = required_columns - set(sites.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns in site CSV: {sorted(missing_columns)}"
        )

    raster_files = sorted(
        [
            path for path in RASTER_DIR.iterdir()
            if path.suffix.lower() in {".tif", ".tiff"}
        ]
    )

    if not raster_files:
        raise FileNotFoundError(f"No TIFF files found in: {RASTER_DIR}")

    # One result list for each input site row.
    site_records = [[] for _ in range(len(sites))]

    print(f"Sites found: {len(sites)}")
    print(f"Raster files found: {len(raster_files)}")

    for raster_number, raster_path in enumerate(raster_files, start=1):
        date = parse_raster_date(raster_path.name)

        if date is None:
            print(f"Skipped (no YYYYMMDD date in filename): {raster_path.name}")
            continue

        try:
            with rasterio.open(raster_path) as src:
                for site_index, site in sites.iterrows():
                    value = extract_value_from_open_raster(
                        src=src,
                        lat=site["lat"],
                        lon=site["lon"],
                    )

                    # Skip rasters where the site is outside the raster
                    # or has a nodata / NaN value.
                    if value is None:
                        continue

                    site_records[site_index].append(
                        {
                            "date": date,
                            "y1km": value,
                        }
                    )

        except Exception as error:
            print(f"Could not read {raster_path.name}: {error}")

        print(f"[{raster_number}/{len(raster_files)}] Processed: {raster_path.name}")

    # Write one date/y1km CSV file for each site.
    for site_index, site in sites.iterrows():
        site_id = safe_filename_component(site["id"])
        network = safe_filename_component(site["network"])
        station = safe_filename_component(site["station"])
        lon = format_coordinate(site["lon"])
        lat = format_coordinate(site["lat"])

        output_name = f"{site_id}_{network}_{station}_{lon}_{lat}.csv"
        output_path = OUTPUT_DIR / output_name

        output_df = pd.DataFrame(site_records[site_index], columns=["date", "y1km"])
        output_df = output_df.sort_values("date").reset_index(drop=True)

        output_df.to_csv(output_path, index=False)

        print(
            f"Saved {output_name} "
            f"({len(output_df)} valid raster values)"
        )

    print(f"\nFinished. Site CSV files saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()