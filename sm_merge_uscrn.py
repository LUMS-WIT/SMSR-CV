import re
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
SITES_CSV = Path(r"D:\SM-DeepLearning\SMSR-CV\Output\ISMN_USCRN_updated.csv")

# Folder containing the per-site prediction CSVs created previously.
PREDICTION_CSV_DIR = Path(
    r"D:\SM-DeepLearning\SMSR-CV\Output\sm_full_uscrn"
)

# Change this to your actual USCRN / ISMN root directory.
# It must contain folders such as: Avondale-2-N, Bedford-5-WNW, ...
USCRN_ROOT_DIR = Path(
    r"D:\SM-DeepLearning\datasets\Central valley\datasets\USCRN_validations\ISMN_USCRN\USCRN"
)


def safe_filename_component(value):
    return re.sub(r'[<>:"/\\|?*]', "_", str(value).strip())


def format_coordinate(value):
    value = float(value)
    return f"{value:.10f}".rstrip("0").rstrip(".")


def normalize_name(value):
    """Makes Avondale-2-N and Avondale_2_N comparable."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_station_directory(uscrn_root_dir, station_name):
    """Find the site folder, allowing hyphen/underscore naming differences."""
    exact_path = uscrn_root_dir / str(station_name)

    if exact_path.is_dir():
        return exact_path

    target_name = normalize_name(station_name)

    matches = [
        folder
        for folder in uscrn_root_dir.iterdir()
        if folder.is_dir() and normalize_name(folder.name) == target_name
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"More than one possible folder found for station: {station_name}"
        )

    return None


def get_depth_from_stm_filename(stm_path):
    """
    Example:
    USCRN_USCRN_Avondale-2-N_sm_0.050000_0.050000_....stm
                                  ^^^^^^^^
    """
    match = re.search(r"_sm_([0-9.]+)_", stm_path.name, re.IGNORECASE)

    if not match:
        return None

    return float(match.group(1))


def depth_to_column_name(depth):
    """
    0.050000 -> sm_d0_05
    0.100000 -> sm_d0_1
    """
    depth_text = f"{depth:.6f}".rstrip("0").rstrip(".")
    return f"sm_d{depth_text.replace('.', '_')}"


def read_stm_observations(stm_path, depth):
    """
    Read all valid observations from one STM file.

    Expected lines:
    2014/04/23 00:00 0.249 G M
    """
    records = []

    with open(stm_path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            parts = line.strip().split()

            # Ignore metadata/header lines and malformed records.
            if len(parts) < 3:
                continue

            if not re.match(r"^\d{4}/\d{2}/\d{2}$", parts[0]):
                continue

            try:
                timestamp = pd.to_datetime(
                    f"{parts[0]} {parts[1]}",
                    format="%Y/%m/%d %H:%M",
                    errors="raise",
                )
                sm_value = float(parts[2])
            except (ValueError, TypeError):
                continue

            # Typical missing-value markers are negative values such as -9999.
            if not np.isfinite(sm_value) or sm_value <= -999:
                continue

            records.append(
                {
                    "date": timestamp.normalize(),
                    "depth": depth,
                    "sm_value": sm_value,
                }
            )

    return records


def get_daily_ismn_soil_moisture(station_dir):
    """
    Read every STM file for a station and return daily mean SM values
    by sensor depth.
    """
    all_records = []

    for stm_path in sorted(station_dir.rglob("*.stm")):
        depth = get_depth_from_stm_filename(stm_path)

        if depth is None:
            print(f"  Skipped STM without identifiable depth: {stm_path.name}")
            continue

        all_records.extend(read_stm_observations(stm_path, depth))

    if not all_records:
        return pd.DataFrame(columns=["date"])

    observations = pd.DataFrame(all_records)

    # Average all hourly/sub-daily measurements for each date and depth.
    daily_means = (
        observations
        .groupby(["date", "depth"], as_index=False)["sm_value"]
        .mean()
    )

    daily_means["column_name"] = daily_means["depth"].apply(
        depth_to_column_name
    )

    # Convert e.g. date / depth / value into date / sm_d0_05 / sm_d0_1.
    daily_wide = (
        daily_means
        .pivot(index="date", columns="column_name", values="sm_value")
        .reset_index()
    )

    daily_wide.columns.name = None
    return daily_wide


def main():
    if not USCRN_ROOT_DIR.is_dir():
        raise FileNotFoundError(
            f"Set USCRN_ROOT_DIR correctly. Folder not found:\n{USCRN_ROOT_DIR}"
        )

    sites = pd.read_csv(SITES_CSV)

    for _, site in sites.iterrows():
        site_id = safe_filename_component(site["id"])
        network = safe_filename_component(site["network"])
        station = safe_filename_component(site["station"])
        lon = format_coordinate(site["lon"])
        lat = format_coordinate(site["lat"])

        prediction_csv = PREDICTION_CSV_DIR / (
            f"{site_id}_{network}_{station}_{lon}_{lat}.csv"
        )

        if not prediction_csv.exists():
            print(f"Prediction CSV not found: {prediction_csv.name}")
            continue

        station_dir = find_station_directory(
            USCRN_ROOT_DIR,
            site["station"],
        )

        if station_dir is None:
            print(f"USCRN station folder not found: {site['station']}")
            continue

        print(f"\nProcessing: {site['station']}")
        print(f"  STM folder: {station_dir}")

        prediction_df = pd.read_csv(prediction_csv)

        if not {"date", "y1km"}.issubset(prediction_df.columns):
            print(f"  Skipped: required date/y1km columns missing.")
            continue

        # Allows the script to be run again without creating duplicate columns.
        depth_columns = [
            column for column in prediction_df.columns
            if column.startswith("sm_d")
        ]
        prediction_df = prediction_df.drop(columns=depth_columns)

        prediction_df["date"] = pd.to_datetime(
            prediction_df["date"],
            errors="coerce",
        ).dt.normalize()

        prediction_df = prediction_df.dropna(subset=["date"])

        ismn_daily_df = get_daily_ismn_soil_moisture(station_dir)

        # Retain every predicted-raster row; unmatched ISMN dates become NaN.
        merged_df = prediction_df.merge(
            ismn_daily_df,
            on="date",
            how="left",
        )

        merged_df["date"] = merged_df["date"].dt.strftime("%Y-%m-%d")

        # Keep date, predicted 1-km SM, then depth columns in order.
        sm_columns = sorted(
            [column for column in merged_df.columns if column.startswith("sm_d")],
            key=lambda column: float(
                column.replace("sm_d", "").replace("_", ".")
            ),
        )

        merged_df = merged_df[["date", "y1km", *sm_columns]]

        # Updates the existing per-site prediction CSV.
        merged_df.to_csv(prediction_csv, index=False)

        print(
            f"  Updated: {prediction_csv.name} "
            f"| prediction rows: {len(merged_df)} "
            f"| depths: {', '.join(sm_columns) if sm_columns else 'none'}"
        )

    print("\nFinished adding ISMN/USCRN soil-moisture columns.")


if __name__ == "__main__":
    main()