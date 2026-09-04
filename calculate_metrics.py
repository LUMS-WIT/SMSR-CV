"""Compute 0.05 m soil-moisture validation metrics for the retained USCRN sites."""

from __future__ import annotations

import csv
import math
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = DATA_DIR / "results_0_05.csv"
OBSERVATION_COLUMN = "sm_d0_05"
PREDICTION_COLUMN = "y1km"
MIN_PAIRS = 2

# Retained from the previous screening requested in this task.
EXCLUDED_SITES = {
    "5_USCRN_Avondale-2-N_-75.7861_39.8593.csv",
    "33_USCRN_Durham-2-N_-70.9277_43.1716.csv",
    "62_USCRN_Los-Alamos-13-W_-106.5214_35.8584.csv",
}


def calculate_metrics(pairs: list[tuple[float, float]]) -> dict[str, float]:
    """Return bias, MSE, ubRMSE, and Pearson r for (observation, prediction) pairs."""
    observations = [observation for observation, _ in pairs]
    predictions = [prediction for _, prediction in pairs]
    errors = [prediction - observation for observation, prediction in pairs]

    bias = sum(errors) / len(errors)
    mse = sum(error**2 for error in errors) / len(errors)
    ubrmse = math.sqrt(sum((error - bias) ** 2 for error in errors) / len(errors))

    observation_mean = sum(observations) / len(observations)
    prediction_mean = sum(predictions) / len(predictions)
    numerator = sum(
        (observation - observation_mean) * (prediction - prediction_mean)
        for observation, prediction in pairs
    )
    observation_ss = sum((observation - observation_mean) ** 2 for observation in observations)
    prediction_ss = sum((prediction - prediction_mean) ** 2 for prediction in predictions)
    pearson_r = numerator / math.sqrt(observation_ss * prediction_ss)

    return {
        "bias": bias,
        "mse": mse,
        "ubrmse": ubrmse,
        "pearson_r": pearson_r,
    }


def read_pairs(csv_path: Path) -> list[tuple[float, float]]:
    """Read finite paired y1km and sm_d0_05 values from one site file."""
    pairs: list[tuple[float, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                observation = float(row[OBSERVATION_COLUMN])
                prediction = float(row[PREDICTION_COLUMN])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(observation) and math.isfinite(prediction):
                pairs.append((observation, prediction))
    return pairs


def main() -> None:
    site_results: list[dict[str, object]] = []
    all_pairs: list[tuple[float, float]] = []
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        if csv_path.name == OUTPUT_FILE.name or csv_path.name in EXCLUDED_SITES:
            continue
        pairs = read_pairs(csv_path)
        if len(pairs) < MIN_PAIRS:
            continue
        metrics = calculate_metrics(pairs)
        site_results.append(
            {"site_or_summary": csv_path.name, "n_pairs": len(pairs), "n_sites": 1, **metrics}
        )
        all_pairs.extend(pairs)

    pooled_metrics = calculate_metrics(all_pairs)
    equal_site_metrics = {
        "bias": sum(result["bias"] for result in site_results) / len(site_results),
        "mse": sum(result["mse"] for result in site_results) / len(site_results),
        "ubrmse": sum(result["ubrmse"] for result in site_results) / len(site_results),
        "pearson_r": math.tanh(
            sum(math.atanh(result["pearson_r"]) for result in site_results) / len(site_results)
        ),
    }
    results = [
        *site_results,
        {
            "site_or_summary": "COMBINED_POOLED",
            "n_pairs": len(all_pairs),
            "n_sites": len(site_results),
            **pooled_metrics,
        },
        {
            "site_or_summary": "EQUAL_SITE_AVERAGE",
            "n_pairs": "",
            "n_sites": len(site_results),
            **equal_site_metrics,
        },
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["site_or_summary", "n_pairs", "n_sites", "bias", "mse", "ubrmse", "pearson_r"],
        )
        writer.writeheader()
        writer.writerows(results)

    for result in results:
        print(
            f"{result['site_or_summary']}: n_pairs={result['n_pairs']}, n_sites={result['n_sites']}, "
            f"bias={result['bias']:.6f}, mse={result['mse']:.6f}, "
            f"ubrmse={result['ubrmse']:.6f}, r={result['pearson_r']:.6f}"
        )


if __name__ == "__main__":
    main()
