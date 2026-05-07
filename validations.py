import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
# =============================================================================
# 1. EXTRACTION — Read all CSVs and build paired values
# =============================================================================

def extraction(
    merged_dir="/content/ismn_sub_with_smap",
    target_col="1km",
    require_y1km=False,
    require_triplet=False
):
    """
    Read all merged CSVs from merged_dir (covering all subroi_* subfolders),
    build paired (ismn_sm, target) values.

    Parameters
    ----------
    merged_dir : str
        Root directory with subroi_*/ subfolders containing merged CSVs.
    target_col : str
        Which column to pair against ISMN soil moisture.
        One of: '1km', '9km', 'y1km'.
    require_y1km : bool
        If True AND target_col is '1km' or '9km', further filter to only
        rows where y1km also exists. Ignored when target_col='y1km'.
    require_triplet : bool
        If True, only keep rows where ALL of swc, 1km, 9km, y1km exist.
        Overrides require_y1km when True.

    Returns
    -------
    paired_values : pd.DataFrame
        Only rows where both swc and target_col are non-NaN
        (plus additional filters if require_y1km or require_triplet).
    """
    all_pairs = []

    for subroi_folder in sorted(os.listdir(merged_dir)):
        subroi_path = os.path.join(merged_dir, subroi_folder)
        if not os.path.isdir(subroi_path) or not subroi_folder.startswith("subroi_"):
            continue

        subroi_id = subroi_folder.replace("subroi_", "")

        for csv_file in sorted(os.listdir(subroi_path)):
            if not csv_file.endswith(".csv"):
                continue

            csv_path = os.path.join(subroi_path, csv_file)
            df = pd.read_csv(csv_path)

            # Check required columns exist
            if "swc" not in df.columns or target_col not in df.columns:
                continue

            # Base filter: both swc and target must be non-NaN
            mask = df["swc"].notna() & df[target_col].notna()

            # Triplet filter: require ALL of 1km, 9km, y1km to exist
            if require_triplet:
                for col in ["1km", "9km", "y1km"]:
                    if col in df.columns:
                        mask = mask & df[col].notna()
                    else:
                        # If a required column doesn't exist, no rows pass
                        mask = mask & False
            # Pair filter: require y1km to exist alongside 1km/9km
            elif require_y1km and target_col != "y1km" and "y1km" in df.columns:
                mask = mask & df["y1km"].notna()

            subset = df.loc[mask, ["date", "swc", target_col]].copy()

            if subset.empty:
                continue

            # Parse lat/lon from filename
            name = csv_file.replace(".csv", "")
            parts = name.rsplit("_", 2)
            try:
                lat = float(parts[-2])
                lon = float(parts[-1])
            except (ValueError, IndexError):
                lat, lon = np.nan, np.nan

            # Station identifier (everything before lat_lon)
            station_id = "_".join(name.rsplit("_", 2)[:-2]) if len(parts) >= 3 else name

            subset["network_station_sensor"] = station_id
            subset["subroi"] = subroi_id
            subset["lat"] = lat
            subset["lon"] = lon

            # Also carry y1km if it exists and we might need it for reference
            if "y1km" in df.columns and target_col != "y1km":
                subset["y1km"] = df.loc[mask, "y1km"].values

            all_pairs.append(subset)

    if not all_pairs:
        print("⚠ No paired data found!")
        return pd.DataFrame()

    paired_values = pd.concat(all_pairs, ignore_index=True)

    filter_desc = "triplet (1km & 9km & y1km)" if require_triplet else \
                  ("pair + y1km" if require_y1km else "pair only")
    print(f"Extraction complete:")
    print(f"  Target column: {target_col}")
    print(f"  Filter mode: {filter_desc}")
    print(f"  Total paired rows: {len(paired_values)}")
    print(f"  Unique stations: {paired_values['network_station_sensor'].nunique()}")
    print(f"  Sub-ROIs covered: {sorted(paired_values['subroi'].unique())}")

    return paired_values


# =============================================================================
# 2a. VALIDATION (per-station) — Compute metrics for a SINGLE station
# =============================================================================

def validation_single(paired_values, target_col="1km", confidence=0.95):
    """
    Compute validation metrics for a single station's paired (swc, target_col) values.

    Returns
    -------
    metrics_dict : dict
        Includes: bias, bias_cl, bias_cu, mse, rmse, ubrmsd, ubrmsd_cl, ubrmsd_cu, p_rho, s_rho
    N : int
    """
    x = paired_values["swc"].to_numpy(dtype=float)
    y = paired_values[target_col].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    N = len(x)

    if N < 3:
        empty = {k: np.nan for k in [
            'bias_cl', 'bias', 'bias_cu', 'mse', 'rmse',
            'ubrmsd_cl', 'ubrmsd', 'ubrmsd_cu', 'p_rho', 's_rho'
        ]}
        return empty, N

    # Bias
    diff = y - x
    bias = np.mean(diff)
    bias_std = np.std(diff, ddof=1)
    t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=N - 1)
    bias_ci = t_crit * bias_std / np.sqrt(N)

    # MSE & RMSE
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)

    # ubRMSD
    diff_anom = diff - bias
    ubrmsd = np.sqrt(np.mean(diff_anom ** 2))

    # CI for ubRMSD
    n_dof = N - 1
    chi2_lower = scipy_stats.chi2.ppf((1 - confidence) / 2, df=n_dof)
    chi2_upper = scipy_stats.chi2.ppf((1 + confidence) / 2, df=n_dof)
    ubrmsd_var = ubrmsd ** 2
    if chi2_upper > 0 and chi2_lower > 0:
        ubrmsd_cl = np.sqrt(n_dof * ubrmsd_var / chi2_upper)
        ubrmsd_cu = np.sqrt(n_dof * ubrmsd_var / chi2_lower)
    else:
        ubrmsd_cl = np.nan
        ubrmsd_cu = np.nan

    # Correlations
    p_rho, _ = scipy_stats.pearsonr(x, y)
    s_rho, _ = scipy_stats.spearmanr(x, y)

    metrics_dict = {
        'bias_cl': bias - bias_ci,
        'bias': bias,
        'bias_cu': bias + bias_ci,
        'mse': mse,
        'rmse': rmse,
        'ubrmsd_cl': ubrmsd_cl,
        'ubrmsd': ubrmsd,
        'ubrmsd_cu': ubrmsd_cu,
        'p_rho': p_rho,
        's_rho': s_rho
    }

    return metrics_dict, N


# =============================================================================
# 2b. VALIDATION (overall) — Average of per-station metrics
# =============================================================================

def fisher_z_mean(rho_values):
    """Compute mean correlation via Fisher Z-transform."""
    rho_arr = np.array(rho_values, dtype=float)
    rho_arr = rho_arr[np.isfinite(rho_arr)]
    if len(rho_arr) == 0:
        return np.nan
    rho_arr = np.clip(rho_arr, -0.9999, 0.9999)
    z = np.arctanh(rho_arr)
    z_mean = np.mean(z)
    return np.tanh(z_mean)


def validation(paired_values, target_col="1km", confidence=0.95):
    """
    Compute overall validation metrics as the AVERAGE OF PER-STATION METRICS.
    Correlations are averaged using Fisher-Z transform.

    Returns
    -------
    metrics_dict : dict
    N : int (total observations across all stations)
    stats_results : dict (for plotting, with 'mean' keys)
    """
    station_metrics_list = []
    total_N = 0

    for station_id, grp in paired_values.groupby("network_station_sensor"):
        m, n = validation_single(grp, target_col=target_col, confidence=confidence)
        if n < 3:
            continue
        m["station"] = station_id
        m["N"] = n
        station_metrics_list.append(m)
        total_N += n

        print(f"\n  Station: {station_id} (N = {n}):")
        print(f"    Bias:    {m['bias']:.4f}  [{m['bias_cl']:.4f}, {m['bias_cu']:.4f}]")
        print(f"    MSE:     {m['mse']:.4f}")
        print(f"    RMSE:    {m['rmse']:.4f}")
        print(f"    ubRMSD:  {m['ubrmsd']:.4f}  [{m['ubrmsd_cl']:.4f}, {m['ubrmsd_cu']:.4f}]")
        print(f"    R (Pearson):   {m['p_rho']:.4f}")
        print(f"    ρ (Spearman):  {m['s_rho']:.4f}")

    if not station_metrics_list:
        print("⚠ No valid stations for overall metrics.")
        empty = {k: np.nan for k in [
            'bias_cl', 'bias', 'bias_cu', 'mse', 'rmse',
            'ubrmsd_cl', 'ubrmsd', 'ubrmsd_cu', 'p_rho', 's_rho'
        ]}
        empty_stats = {k: {'mean': np.nan} for k in ['bias', 'mse', 'rmse', 'ubrmsd', 'p_rho', 's_rho']}
        return empty, 0, empty_stats

    metrics_df = pd.DataFrame(station_metrics_list)
    n_stations = len(metrics_df)

    # Average of per-station metrics
    mean_bias    = metrics_df['bias'].mean()
    mean_mse     = metrics_df['mse'].mean()
    mean_rmse    = metrics_df['rmse'].mean()
    mean_ubrmsd  = metrics_df['ubrmsd'].mean()

    # Fisher-Z averaged correlations
    mean_p_rho = fisher_z_mean(metrics_df['p_rho'].values)
    mean_s_rho = fisher_z_mean(metrics_df['s_rho'].values)

    # CI for averaged bias
    if n_stations > 1:
        bias_std = metrics_df['bias'].std(ddof=1)
        t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=n_stations - 1)
        bias_ci = t_crit * bias_std / np.sqrt(n_stations)
    else:
        bias_ci = 0.0

    # CI for averaged ubRMSD
    if n_stations > 1:
        ubrmsd_std = metrics_df['ubrmsd'].std(ddof=1)
        t_crit_ub = scipy_stats.t.ppf((1 + confidence) / 2, df=n_stations - 1)
        ubrmsd_ci = t_crit_ub * ubrmsd_std / np.sqrt(n_stations)
    else:
        ubrmsd_ci = 0.0

    metrics_dict = {
        'bias_cl': mean_bias - bias_ci,
        'bias': mean_bias,
        'bias_cu': mean_bias + bias_ci,
        'mse': mean_mse,
        'rmse': mean_rmse,
        'ubrmsd_cl': mean_ubrmsd - ubrmsd_ci,
        'ubrmsd': mean_ubrmsd,
        'ubrmsd_cu': mean_ubrmsd + ubrmsd_ci,
        'p_rho': mean_p_rho,
        's_rho': mean_s_rho
    }

    stats_results = {
        'bias':   {'mean': mean_bias,   'cl': mean_bias - bias_ci,   'cu': mean_bias + bias_ci},
        'mse':    {'mean': mean_mse},
        'rmse':   {'mean': mean_rmse},
        'ubrmsd': {'mean': mean_ubrmsd, 'cl': mean_ubrmsd - ubrmsd_ci, 'cu': mean_ubrmsd + ubrmsd_ci},
        'p_rho':  {'mean': mean_p_rho},
        's_rho':  {'mean': mean_s_rho}
    }

    print(f"\n{'='*50}")
    print(f"OVERALL METRICS (average of {n_stations} stations, total N = {total_N}):")
    print(f"{'='*50}")
    print(f"  Bias:    {mean_bias:.4f}  [{mean_bias - bias_ci:.4f}, {mean_bias + bias_ci:.4f}]")
    print(f"  MSE:     {mean_mse:.4f}")
    print(f"  RMSE:    {mean_rmse:.4f}")
    print(f"  ubRMSD:  {mean_ubrmsd:.4f}  [{mean_ubrmsd - ubrmsd_ci:.4f}, {mean_ubrmsd + ubrmsd_ci:.4f}]")
    print(f"  R (Pearson, Fisher-Z avg):   {mean_p_rho:.4f}")
    print(f"  ρ (Spearman, Fisher-Z avg):  {mean_s_rho:.4f}")

    return metrics_dict, total_N, stats_results


# =============================================================================
# 3. PLOTTING — Scatter plot with metrics annotation (UNCHANGED)
# =============================================================================

def plot_paired(paired_values, stats_results, N, target_col="1km",
                title=None, xlabel=None, ylabel=None):
    """
    Scatter plot of ISMN soil moisture vs. target column with 1:1 line
    and annotated metrics.
    """
    x = paired_values["swc"].to_numpy(dtype=float)
    y = paired_values[target_col].to_numpy(dtype=float)

    if x.size == 0 or y.size == 0:
        raise ValueError("No paired data available after filtering to plot.")

    min_v = np.nanmin([x.min(), y.min()])
    max_v = np.nanmax([x.max(), y.max()])
    pad = 0.02 * (max_v - min_v) if np.isfinite(max_v - min_v) else 0.02

    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(x, y, s=12, alpha=0.65, edgecolor='none', label='Pairs')
    plt.plot([min_v - pad, max_v + pad], [min_v - pad, max_v + pad],
             linestyle='--', color='gray', linewidth=1, label='1:1')

    if target_col == "y1km":
        ylabel = "Predicted 1km Soil Moisture [m³/m³]"
    
    plt.xlabel(xlabel or "ISMN Soil Moisture [$m^3 m^{-3}$]")
    plt.ylabel(ylabel or f"{target_col} Soil Moisture [$m^3 m^{-3}$]")
    plt.title(title or f"ISMN vs {target_col} (N={N})")
    plt.xlim(min_v - pad, max_v + pad)
    plt.ylim(min_v - pad, max_v + pad)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(loc='lower right')

    mean_bias   = stats_results['bias']['mean']
    mean_mse    = stats_results['mse']['mean']
    mean_rmse   = stats_results['rmse']['mean']
    mean_p_rho  = stats_results['p_rho']['mean']
    mean_s_rho  = stats_results['s_rho']['mean']
    mean_ubrmsd = stats_results['ubrmsd']['mean']

    r2_disp = mean_p_rho ** 2 if mean_p_rho is not None else np.nan

    textstr = (
        f"Bias = {mean_bias:.3f} m³/m³\n"
        # f"RMSE = {mean_rmse:.3f} m³/m³\n"
        f"MSE = {mean_mse:.3f} m³/m³\n"
        f"ubRMSE = {mean_ubrmsd:.3f} m³/m³\n"
        # f"R = {mean_p_rho:.3f}  (R² = {r2_disp:.3f})\n"
        f"R = {mean_p_rho:.3f}"
        # fr"$\rho$ = {mean_s_rho:.3f}"
        f"\nN = {N}"
    )
    plt.gca().text(
        0.02, 0.98, textstr,
        transform=plt.gca().transAxes,
        va='top', ha='left',
        fontsize=9,
        bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9)
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# 4. BOX & WHISKER PLOTS — Two separate plots
# =============================================================================

def plot_boxwhisker(paired_values, target_col="1km", confidence=0.95, title=None):
    """
    Compute metrics per station and show TWO box-and-whisker plots:
      Plot 1: Pearson R and Spearman ρ (y-axis: 0 to 1)
      Plot 2: Bias, MSE, ubRMSD (y-axis: -0.2 to 0.5, no outliers)

    Parameters
    ----------
    paired_values : pd.DataFrame
        Must contain 'swc', target_col, and 'network_station_sensor'.
    target_col : str
        Column to validate against.
    confidence : float
        Confidence level for CIs.
    title : str, optional
        Custom plot title prefix.
    """
    station_metrics = []

    for station_id, grp in paired_values.groupby("network_station_sensor"):
        if len(grp) < 3:
            continue
        m, n = validation_single(
            grp,
            target_col=target_col,
            confidence=confidence
        )
        m["station"] = station_id
        m["N"] = n
        station_metrics.append(m)

    if not station_metrics:
        print("⚠ Not enough per-station data for box plots.")
        return

    metrics_df = pd.DataFrame(station_metrics)
    base_title = title or f"Per-Station Metrics: ISMN vs {target_col}"

    # =========================================================================
    # Plot 1: Correlations — Pearson R and Spearman ρ (shared plot, y: 0–1)
    # =========================================================================
    corr_cols = ['p_rho', 's_rho']
    corr_labels = ['Pearson R', 'Spearman ρ']

    corr_data = [metrics_df[c].dropna().values for c in corr_cols]

    fig1, ax1 = plt.subplots(figsize=(5, 5))
    bp1 = ax1.boxplot(
        corr_data,
        patch_artist=True,
        widths=0.4,
        showfliers=False
    )

    colors_corr = ['#7CB9E8', '#F4A460']
    for patch, color in zip(bp1['boxes'], colors_corr):
        patch.set_facecolor(color)
    for median in bp1['medians']:
        median.set_color('red')
        median.set_linewidth(1.5)

    ax1.set_xticklabels(corr_labels, fontsize=11)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Correlation", fontsize=11)
    # ax1.set_title(f"{base_title}\nCorrelations (n_stations={len(metrics_df)})", fontsize=12)
    ax1.set_title(f"{base_title}", fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # Annotate means
    for i, (col, label) in enumerate(zip(corr_cols, corr_labels)):
        data = metrics_df[col].dropna()
        if len(data) > 0:
            mean_val = data.mean()
            ax1.scatter(i + 1, mean_val, marker='D', color='blue', s=40, zorder=5)
            ax1.text(i + 1.2, mean_val, f"μ={mean_val:.3f}", va='center', fontsize=9, color='blue')

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # Plot 2: Error metrics — Bias, MSE, ubRMSD (no outliers, y: -0.2–0.5)
    # =========================================================================
    err_cols = ['bias', 'mse', 'ubrmsd']
    err_labels = ['Bias', 'MSE', 'ubRMSD']

    err_data = [metrics_df[c].dropna().values for c in err_cols]

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    bp2 = ax2.boxplot(
        err_data,
        patch_artist=True,
        widths=0.4,
        showfliers=False  # no outliers
    )

    colors_err = ['#90EE90', '#FFB6C1', '#DDA0DD']
    for patch, color in zip(bp2['boxes'], colors_err):
        patch.set_facecolor(color)
    for median in bp2['medians']:
        median.set_color('red')
        median.set_linewidth(1.5)

    ax2.set_xticklabels(err_labels, fontsize=11)
    ax2.set_ylim(-0.10, 0.2)
    ax2.set_ylabel("Metrics Value [m³/m³]", fontsize=11)
    # ax2.set_title(f"{base_title}\nError Metrics (n_stations={len(metrics_df)}, no outliers)", fontsize=12)
    ax2.set_title(f"{base_title}", fontsize=12)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)

    # Annotate means
    for i, (col, label) in enumerate(zip(err_cols, err_labels)):
        data = metrics_df[col].dropna()
        if len(data) > 0:
            mean_val = data.mean()
            ax2.scatter(i + 1, mean_val, marker='D', color='blue', s=40, zorder=5)
            ax2.text(i + 1.2, mean_val, f"μ={mean_val:.3f}", va='center', fontsize=9, color='blue')

    plt.tight_layout()
    plt.show()

    # Print summary table
    print(f"\nPer-Station Metrics Summary (n_stations = {len(metrics_df)}):")
    summary_cols = ['station', 'N', 'bias', 'mse', 'rmse', 'ubrmsd', 'p_rho', 's_rho']
    summary = metrics_df[summary_cols].round(4)
    print(summary.to_string(index=False))

    return metrics_df


# =============================================================================
# 5. RUN EVERYTHING — Convenience wrapper
# =============================================================================

def run_validation_pipeline(
    merged_dir="/content/ismn_sub_with_smap",
    target_col="1km",
    require_y1km=False,
    require_triplet=False,
    title=None
):
    """
    Full pipeline: extract -> validate -> scatter plot -> box plots.

    Parameters
    ----------
    merged_dir : str
        Root directory with merged CSVs.
    target_col : str
        '1km', '9km', or 'y1km'.
    require_y1km : bool
        If True and target_col is '1km'/'9km', only keep rows
        where y1km also exists (pair filter).
    require_triplet : bool
        If True, only keep rows where ALL of 1km, 9km, y1km exist
        (triplet filter). Overrides require_y1km.
    title : str, optional
        Custom title for plots.
    """
    filter_mode = "triplet" if require_triplet else ("pair + y1km" if require_y1km else "pair only")
    print(f"{'='*70}")
    print(f"VALIDATION PIPELINE: ISMN vs {target_col}")
    print(f"  Filter mode: {filter_mode}")
    print(f"{'='*70}")

    # Step 1: Extract
    paired = extraction(
        merged_dir=merged_dir,
        target_col=target_col,
        require_y1km=require_y1km,
        require_triplet=require_triplet
    )
    if paired.empty:
        return

    # Step 2: Validate (average of per-station metrics)
    metrics, N, stats_res = validation(paired, target_col=target_col)

    # Step 3: Scatter plot
    suffix = f" ({filter_mode})" if require_y1km or require_triplet else ""
    plot_title = title or f"ISMN vs {target_col}"
    plot_paired(paired, stats_res, N, target_col=target_col, title=plot_title)

    # Step 4: Box & whisker (per station)
    metrics_df = plot_boxwhisker(paired, target_col=target_col, title=plot_title)

    return paired, metrics, N, stats_res, metrics_df

run_validation_pipeline(
    merged_dir="output\\sm_full",
    target_col="y1km",
    require_y1km=True,
    require_triplet=True,
    title=None
)

# =============================================================================
# 6. PER-STATION TIME SERIES PLOTS — Reading count on x-axis
#    (ADD-ON ONLY; does not change any existing code)
# =============================================================================

def plot_station_timeseries_count(
    paired_values,
    target_col="y1km",
    station_col="network_station_sensor",
    date_col="date",
    swc_col="swc",
    sort_by_date=True,
    figsize=(10, 4.5),
    linewidth=1.5,
    marker_size=18,
    alpha=0.85,
    save_dir=None,
    dpi=300,
    show=True
):
    """
    Plot per-station time series of ISMN vs predicted values using reading count on x-axis.

    Parameters
    ----------
    paired_values : pd.DataFrame
        Output dataframe from extraction() / run_validation_pipeline().
        Must contain station_col, swc_col, target_col, and optionally date_col.
    target_col : str
        Column to compare against ISMN (e.g., '1km', '9km', 'y1km').
    station_col : str
        Station identifier column.
    date_col : str
        Date column used only for sorting, if available.
    swc_col : str
        In-situ soil moisture column.
    sort_by_date : bool
        If True, sorts each station by date before assigning reading count.
    figsize : tuple
        Figure size for each station.
    linewidth : float
        Line width.
    marker_size : int
        Scatter marker size.
    alpha : float
        Transparency.
    save_dir : str or None
        If provided, saves each station plot in this directory.
    dpi : int
        Save DPI.
    show : bool
        If True, display plots. If False, only save them.

    Returns
    -------
    station_plot_summary : pd.DataFrame
        Summary table with station names and number of plotted readings.
    """
    required_cols = {station_col, swc_col, target_col}
    missing = required_cols - set(paired_values.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = paired_values.copy()

    # Try parsing date safely if present
    if sort_by_date and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Keep only rows valid for plotting
    df = df[df[swc_col].notna() & df[target_col].notna()].copy()

    if df.empty:
        raise ValueError("No valid paired rows found for time series plotting.")

    # Create save directory if needed
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    summary_rows = []

    for station_id, grp in df.groupby(station_col):
        grp = grp.copy()

        # Sort by date if requested and date exists
        if sort_by_date and date_col in grp.columns:
            grp = grp.sort_values(by=date_col, kind="stable")
        else:
            grp = grp.reset_index(drop=True)

        grp = grp.reset_index(drop=True)
        grp["reading_count"] = np.arange(1, len(grp) + 1)

        x = grp["reading_count"].to_numpy()
        y_obs = grp[swc_col].to_numpy(dtype=float)
        y_pred = grp[target_col].to_numpy(dtype=float)

        # Optional station metrics for annotation
        station_metrics, N_station = validation_single(grp, target_col=target_col)

        fig, ax = plt.subplots(figsize=figsize)

        # ISMN observed
        ax.plot(
            x, y_obs,
            linestyle='-',
            linewidth=linewidth,
            marker='o',
            markersize=np.sqrt(marker_size),
            alpha=alpha,
            label='ISMN in-situ'
        )

        # Predicted / target
        label_map = {
            "1km": "1 km",
            "9km": "9 km",
            "y1km": "Predicted 1 km"
        }
        pred_label = label_map.get(target_col, target_col)

        ax.plot(
            x, y_pred,
            linestyle='-',
            linewidth=linewidth,
            marker='s',
            markersize=np.sqrt(marker_size),
            alpha=alpha,
            label=pred_label
        )

        ax.set_xlabel("Observation count")
        ax.set_ylabel("Soil Moisture [$m^3 m^{-3}$]")
        ax.set_title(f"Station: {station_id}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Annotation box
        if N_station >= 3:
            textstr = (
                f"Bias = {station_metrics['bias']:.3f}\n"
                f"ubRMSD = {station_metrics['ubrmsd']:.3f}\n"
                f"R = {station_metrics['p_rho']:.3f}\n"
                f"N = {N_station}"
                # f"ρ = {station_metrics['s_rho']:.3f}"
            )
            ax.text(
                0.02, 0.98, textstr,
                transform=ax.transAxes,
                va='top', ha='left',
                fontsize=9,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9)
            )

        plt.tight_layout()

        # Save if requested
        if save_dir is not None:
            safe_station = "".join(c if c.isalnum() or c in ('-', '_') else "_" for c in str(station_id))
            out_path = os.path.join(save_dir, f"{safe_station}_{target_col}_timeseries_count.png")
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        summary_rows.append({
            "station": station_id,
            "N": N_station
        })

    station_plot_summary = pd.DataFrame(summary_rows).sort_values("station").reset_index(drop=True)

    print("\nPer-station time series plotting completed.")
    print(f"Total stations plotted: {len(station_plot_summary)}")
    print(station_plot_summary.to_string(index=False))

    return station_plot_summary


# =============================================================================
# 7. OPTIONAL WRAPPER — Run validation pipeline + per-station time series plots
#    (ADD-ON ONLY)
# =============================================================================

def run_validation_pipeline_with_station_timeseries(
    merged_dir="/content/ismn_sub_with_smap",
    target_col="y1km",
    require_y1km=False,
    require_triplet=False,
    title=None,
    station_plot_save_dir=None,
    station_plot_show=True
):
    """
    Existing pipeline + per-station reading-count plots.

    Returns
    -------
    paired, metrics, N, stats_res, metrics_df, station_plot_summary
    """
    results = run_validation_pipeline(
        merged_dir=merged_dir,
        target_col=target_col,
        require_y1km=require_y1km,
        require_triplet=require_triplet,
        title=title
    )

    if results is None:
        return None

    paired, metrics, N, stats_res, metrics_df = results

    station_plot_summary = plot_station_timeseries_count(
        paired_values=paired,
        target_col=target_col,
        save_dir=station_plot_save_dir,
        show=station_plot_show
    )

    return paired, metrics, N, stats_res, metrics_df, station_plot_summary


# Option 2: single wrapper
run_validation_pipeline_with_station_timeseries(
    merged_dir="Output\\sm",
    target_col="y1km",
    require_y1km=True,
    require_triplet=True,
    title=None,
    station_plot_save_dir="Output\\timeseries_plots",
    # station_plot_save_dir=None,
    station_plot_show=True
)