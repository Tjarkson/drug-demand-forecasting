import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

#############################
# Umgang mit fehlenden Werten
#############################

def missing_data_handling(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country, group in df.groupby('REF_AREA'):
        years = group['TIME_PERIOD'].dropna().astype(int)
        if years.empty:
            continue
        first_year = int(years.min())
        last_year = int(years.max())
        full_years = set(range(first_year, last_year + 1))
        observed_years = set(years)
        missing_years = sorted(full_years - observed_years)
        span_len = last_year - first_year + 1
        observed_periods = len(observed_years)
        missing_count = len(missing_years)
        missing_quote = missing_count / span_len if span_len else 0.0
        rows.append(
            {
                "country": country,
                "first_year": first_year,
                "last_year": last_year,
                "series_length": span_len,
                "observed_periods": observed_periods,
                "missing_count": missing_count,
                "missing_quote": missing_quote,
                "missing_years": missing_years,
            }
        )
    summary = (
        pd.DataFrame(rows)
        .query("missing_count > 0")
        .sort_values("missing_quote", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def missing_data_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country, group in df.groupby('REF_AREA'):
        years = group['TIME_PERIOD'].dropna().astype(int)
        if years.empty:
            continue
        first_year = int(years.min())
        last_year = int(years.max())
        full_years = set(range(first_year, last_year + 1))
        observed_years = set(years)
        missing_years = sorted(full_years - observed_years)
        nan_years = sorted(
            group.loc[group["OBS_VALUE"].isna(), "TIME_PERIOD"].dropna().astype(int).unique()
        )
        combined_missing_years = sorted(set(missing_years) | set(nan_years))
        span_len = last_year - first_year + 1
        observed_periods = len(observed_years)
        missing_count = len(combined_missing_years)
        missing_quote = missing_count / span_len if span_len else 0.0
        rows.append(
            {
                "country": country,
                "first_year": first_year,
                "last_year": last_year,
                "series_length": span_len,
                "observed_periods": observed_periods,
                "missing_count": missing_count,
                "missing_quote": missing_quote,
                "missing_years": combined_missing_years,
            }
        )
    summary = (
        pd.DataFrame(rows)
        .query("missing_count > 0")
        .sort_values("missing_quote", ascending=False)
        .reset_index(drop=True)
    )
    summary["missing_years"] = summary["missing_years"].apply(
        lambda ys: ", ".join(str(y) for y in ys)
    )
    return summary

def plot_missing_data(
    df: pd.DataFrame,
    ref_area: str,
    time_col: str = "TIME_PERIOD",
    value_col: str = "OBS_VALUE",
    global_years_range: tuple[int, int] | None = None,
    global_ylim: tuple[float, float] | None = None,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    # Visualisiert eine Zeitreihe und zeigt Luecken als Unterbrechungen.
    data = df[df["REF_AREA"] == ref_area].sort_values(time_col)
    if data.empty:
        raise ValueError(f"Keine Daten fuer {ref_area} in Spalte REF_AREA.")

    if global_years_range is not None:
        min_year, max_year = global_years_range
    else:
        years = df[time_col].dropna().astype(int)
        if years.empty:
            raise ValueError(f"Keine gueltigen Jahre in Spalte {time_col}.")
        min_year = int(years.min())
        max_year = int(years.max())

    if xlim is None:
        xlim = (min_year, max_year)
    if ylim is None:
        if global_ylim is not None:
            ylim = global_ylim
        else:
            all_vals = df[value_col].dropna()
            if not all_vals.empty:
                ylim = (float(all_vals.min()), float(all_vals.max()))

    series = (
        data.set_index(data[time_col].astype(int))[value_col]
        .reindex(range(min_year, max_year + 1))
    )

    fig, ax = plt.subplots(figsize=(6.5, 1))
    ax.plot(list(series.index), series.to_numpy(), color="black", linewidth=1.5)
    present_mask = series.notna()
    if present_mask.any():
        ax.scatter(
            list(series.index[present_mask]),
            series[present_mask].to_numpy(),
            color="black",
            s=10,
            zorder=4,
        )
    font_kwargs = {"fontfamily": "Times New Roman", "fontsize": 12}
    ax.set_title(f"{ref_area} - Zeitreihe mit fehlenden Werten", **font_kwargs)
    ax.set_xlabel("Jahr", **font_kwargs)
    ax.set_ylabel(value_col, **font_kwargs)
    ax.tick_params(axis="both", labelsize=12)
    plt.setp(ax.get_xticklabels(), fontfamily="Times New Roman", fontsize=12)
    plt.setp(ax.get_yticklabels(), fontfamily="Times New Roman", fontsize=12)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.2)
    plt.show()

######################################################################
# Kürzen der Zeitreihen mit Anteil fehlender Werte von mindestens 20 %
######################################################################

def trim_by_missing_ratio(df: pd.DataFrame, threshold: float = 0.2) -> tuple[pd.DataFrame, list[tuple[str, int, int]], pd.DataFrame, dict[str, str]]:
    # Kürzt Zeitreihen von Anfang an bis der Fehlanteil <= Schwellenwert ist.
    original_columns = list(df.columns)
    extra_cols = [c for c in original_columns if c not in ["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]]
    summary = missing_data_handling(df)
    countries_above = summary.loc[summary["missing_quote"] > threshold].reset_index(drop=True)
    countries_above_set = set(countries_above["country"])
    missing_count_map = summary.set_index("country")["missing_count"].to_dict()

    trimmed_groups = []
    cut_events: list[tuple[str, int, int]] = []  # (country, from_year, to_year)
    removed_existing_map: dict[str, str] = {}

    for country, group in df.groupby('REF_AREA'):
        group_sorted = group.sort_values('TIME_PERIOD')
        full_years = range(group_sorted['TIME_PERIOD'].min(), group_sorted['TIME_PERIOD'].max() + 1)
        series = (
            group_sorted
            .set_index('TIME_PERIOD')['OBS_VALUE']
            .reindex(full_years)
        )
        if country in countries_above_set:
            total_len = len(series)
            missing_count = int(missing_count_map.get(country, 0))
            removed_start = None
            removed_end = None
            # Kürzen vom Anfang, solange Missing-Quote über Schwelle
            while total_len > 0 and (missing_count / total_len) > threshold:
                year0 = int(series.index[0])
                removed_start = removed_start or year0
                removed_end = year0
                if pd.isna(series.iloc[0]):
                    missing_count -= 1
                series = series.iloc[1:]
                total_len -= 1
            # Wenn nach Unterschreiten der Schwelle weiterhin führende NaNs vorhanden sind,
            # ebenfalls abschneiden.
            while total_len > 0 and pd.isna(series.iloc[0]):
                year0 = int(series.index[0])
                removed_start = removed_start or year0
                removed_end = year0
                missing_count = max(missing_count - 1, 0)
                series = series.iloc[1:]
                total_len -= 1
            if removed_start is not None:
                cut_events.append((country, removed_start, removed_end))
                existing_years = group_sorted["TIME_PERIOD"].dropna().astype(int)
                removed_existing = sorted(
                    y for y in existing_years if removed_start <= y <= removed_end
                )
                removed_existing_map[country] = ", ".join(str(y) for y in removed_existing)
        if series.empty:
            print()
            print(f"Warnung: {country} nach Schnitt keine Daten mehr vorhanden.")
            continue

        trimmed = pd.DataFrame({
            "TIME_PERIOD": series.index.astype(int),
            "OBS_VALUE": series.to_numpy(),
            "REF_AREA": country,
        })
        if extra_cols:
            original_extra = (
                group_sorted[["TIME_PERIOD"] + extra_cols]
                .drop_duplicates(subset=["TIME_PERIOD"])
            )
            trimmed = trimmed.merge(original_extra, on="TIME_PERIOD", how="left")
        trimmed_groups.append(trimmed)

    df_trimmed = pd.concat(trimmed_groups, ignore_index=True)
    df_trimmed = df_trimmed.sort_values(['REF_AREA', 'TIME_PERIOD']).reset_index(drop=True)

    return df_trimmed, cut_events, countries_above, removed_existing_map

##################################################################################
# Auswahl einer geeigneten Imputationsmethode mithilfe einer Montecarlo-Simulation
##################################################################################

def missing_ratio_isolated(df: pd.DataFrame) -> float:
    # Berechnet den Anteil isolierter fehlender Werte an der Gesamtlänge der Zeitreihen.
    if df.empty:
        return 0.0

    total_span = 0
    isolated_missing = 0

    for country, group in df.groupby("REF_AREA"):
        years = group["TIME_PERIOD"].dropna().astype(int)
        if years.empty:
            continue
        full_years = range(years.min(), years.max() + 1)
        series = (
            group
            .sort_values("TIME_PERIOD")
            .set_index("TIME_PERIOD")["OBS_VALUE"]
            .reindex(full_years)
        )
        total_span += len(series)
        if series.isna().any():
            mask = series.isna()
            run_lengths = mask.groupby(mask.ne(mask.shift()).cumsum()).transform("sum")
            isolated_missing += int((mask & (run_lengths == 1)).sum())

    if total_span == 0:
        return 0.0

    missing_ratio_isolated = float(isolated_missing / total_span)

    return missing_ratio_isolated

def missing_ratio_sequential(df: pd.DataFrame) -> tuple[float, str | None, int | None]:
    # Berechnet den Anteil aufeinanderfolgender fehlender Werte an der Gesamtlänge der Zeitreihen.
    if df.empty:
        return 0.0, None, None

    total_span = 0
    sequential_missing = 0
    longest_sequential_runs: dict[str, int] = {}

    for country, group in df.groupby("REF_AREA"):
        years = group["TIME_PERIOD"].dropna().astype(int)
        if years.empty:
            continue
        full_years = range(years.min(), years.max() + 1)
        series = (
            group
            .sort_values("TIME_PERIOD")
            .set_index("TIME_PERIOD")["OBS_VALUE"]
            .reindex(full_years)
        )
        total_span += len(series)
        if series.isna().any():
            mask = series.isna()
            run_lengths = mask.groupby(mask.ne(mask.shift()).cumsum()).transform("sum")
            seq_mask = mask & (run_lengths > 1)
            if seq_mask.any():
                sequential_missing += int(seq_mask.sum())
                longest_sequential_runs[country] = int(run_lengths[seq_mask].max())

    if total_span == 0:
        return 0.0, None, None

    missing_ratio_sequential = float(sequential_missing / total_span)

    if longest_sequential_runs:
        longest_seq_country = max(longest_sequential_runs, key=longest_sequential_runs.get)
        longest_seq_len = longest_sequential_runs[longest_seq_country]
    else:
        longest_seq_country = None
        longest_seq_len = None

    return missing_ratio_sequential, longest_seq_country, longest_seq_len


def filter_complete_series(
    df: pd.DataFrame, group_col: str = "REF_AREA", value_col: str = "OBS_VALUE"
) -> pd.DataFrame:
    # Filtert den DataFrame auf Gruppen mit vollständigen Werten (keine NaNs).
    if df.empty:
        return df.copy()
    na_counts = df.groupby(group_col)[value_col].apply(lambda s: s.isna().sum())
    complete = na_counts[na_counts == 0].index
    return df[df[group_col].isin(complete)].copy()

def mask_isolated(df: pd.DataFrame, missing_ratio_isolated: float, seed: int):
    # Maskiert isolierte fehlende Werte global über alle vollständigen Reihen (globales Target).
    rng = np.random.default_rng(seed)
    masked_list: list[tuple[str, np.ndarray, np.ndarray]] = []
    original_list: list[tuple[str, np.ndarray, np.ndarray]] = []
    positions: list[tuple[int, int]] = []
    if df.empty:
        return masked_list, positions, original_list

    series_data = []
    for ref_area, group in df.sort_values("TIME_PERIOD").groupby("REF_AREA"):
        time_period = group["TIME_PERIOD"].to_numpy()
        vals = group["OBS_VALUE"].to_numpy().copy()
        original_vals = vals.copy()
        series_data.append((ref_area, time_period, vals, original_vals))

    # Globale Maskierung über vollständige Reihen
    complete_indices = [i for i, (_, _, vals, _) in enumerate(series_data) if not np.isnan(vals).any()]
    if not complete_indices:
        print("Keine vollstaendigen Zeitreihen ohne NaNs fuer Maskierung (isolated).")
        for ref_area, time_period, vals, original_vals in series_data:
            masked_list.append((ref_area, time_period, vals))
            original_list.append((ref_area, time_period, original_vals))
        return masked_list, positions, original_list

    global_candidates: list[tuple[int, int]] = []
    for idx in complete_indices:
        vals = series_data[idx][2]
        valid = [
            j
            for j in range(1, len(vals) - 1)
            if not (np.isnan(vals[j]) or np.isnan(vals[j - 1]) or np.isnan(vals[j + 1]))
        ]
        for j in valid:
            global_candidates.append((idx, j))

    total_valid = len(global_candidates)
    if total_valid == 0:
        print("Keine gueltigen Positionen fuer Maskierung (isolated).")
        for ref_area, time_period, vals, original_vals in series_data:
            masked_list.append((ref_area, time_period, vals))
            original_list.append((ref_area, time_period, original_vals))
        return masked_list, positions, original_list

    target = max(1, int(round(total_valid * missing_ratio_isolated)))
    target = min(target, total_valid)

    blocked_map: dict[int, set[int]] = {}
    chosen_count = 0
    for idx, pos in rng.permutation(global_candidates):
        if chosen_count >= target:
            break
        blocked = blocked_map.setdefault(idx, set())
        if pos in blocked:
            continue
        blocked.update({pos - 1, pos, pos + 1})
        positions.append((idx, pos))
        chosen_count += 1

    for idx, (ref_area, time_period, vals, original_vals) in enumerate(series_data):
        masked_vals = vals.copy()
        for s_idx, pos in positions:
            if s_idx == idx and pos < len(masked_vals):
                masked_vals[pos] = np.nan
        masked_list.append((ref_area, time_period, masked_vals))
        original_list.append((ref_area, time_period, original_vals))

    return masked_list, positions, original_list

def mask_sequential(df: pd.DataFrame, missing_ratio_sequential: float, seed: int) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], list[tuple[int, int]], list[tuple[str, np.ndarray, np.ndarray]]]:
    # Maskiert sequenzielle fehlende Werte global über alle vollständigen Reihen (globales Target).
    rng = np.random.default_rng(seed)
    masked_list: list[tuple[str, np.ndarray, np.ndarray]] = []
    original_list: list[tuple[str, np.ndarray, np.ndarray]] = []
    positions: list[tuple[int, int]] = []
    if df.empty:
        return masked_list, positions, original_list
    series_data = []
    for ref_area, group in df.sort_values("TIME_PERIOD").groupby("REF_AREA"):
        time_period = group["TIME_PERIOD"].to_numpy()
        vals = group["OBS_VALUE"].to_numpy().copy()
        original_vals = vals.copy()
        series_data.append((ref_area, time_period, vals, original_vals))

    # Globale Maskierung über vollständige Reihen
    complete_indices = [i for i, (_, _, vals, _) in enumerate(series_data) if not np.isnan(vals).any()]
    if not complete_indices:
        print("Keine vollstaendigen Zeitreihen ohne NaNs fuer Maskierung (sequential).")
        for ref_area, time_period, vals, original_vals in series_data:
            masked_list.append((ref_area, time_period, vals))
            original_list.append((ref_area, time_period, original_vals))
        return masked_list, positions, original_list

    global_starts: list[tuple[int, int]] = []
    for idx in complete_indices:
        vals = series_data[idx][2]
        # Stelle sicher, dass nach dem Maskieren (3 Punkte) noch mindestens order+1 Stützpunkte übrig bleiben.
        if len(vals) - 3 < (3 + 1):  # order=3 -> mind. 4 verbleibende Punkte
            continue
        valid_starts = [
            j
            for j in range(len(vals))
            if j > 0
            and (j + 2) < (len(vals) - 1)  # keine Maskierung am Anfang/Ende
            and not (np.isnan(vals[j]) or np.isnan(vals[j + 1]) or np.isnan(vals[j + 2]))
        ]
        for j in valid_starts:
            global_starts.append((idx, j))

    total_starts = len(global_starts)
    if total_starts == 0:
        print("Keine gueltigen Startpunkte fuer Maskierung (sequential).")
        for ref_area, time_period, vals, original_vals in series_data:
            masked_list.append((ref_area, time_period, vals))
            original_list.append((ref_area, time_period, original_vals))
        return masked_list, positions, original_list

    target = max(1, int(round(total_starts * missing_ratio_sequential)))
    target = min(target, total_starts)

    blocked_map: dict[int, set[int]] = {}
    chosen = 0
    for idx, start in rng.permutation(global_starts):
        if chosen >= target:
            break
        blocked = blocked_map.setdefault(idx, set())
        if start in blocked or (start + 1) in blocked or (start + 2) in blocked:
            continue
        blocked.update({start, start + 1, start + 2})
        positions.append((idx, start))
        chosen += 1

    for idx, (ref_area, time_period, vals, original_vals) in enumerate(series_data):
        masked_vals = vals.copy()
        for s_idx, start in positions:
            if s_idx == idx:
                if start < len(masked_vals):
                    masked_vals[start] = np.nan
                if start + 1 < len(masked_vals):
                    masked_vals[start + 1] = np.nan
                if start + 2 < len(masked_vals):
                    masked_vals[start + 2] = np.nan
        masked_list.append((ref_area, time_period, masked_vals))
        original_list.append((ref_area, time_period, original_vals))

    return masked_list, positions, original_list


def interpolate_masked(masked_list, method: str, order: int = 3) -> list[np.ndarray]:
    filled = []
    for _, years, vals in masked_list:
        ser = pd.Series(vals, index=years).sort_index()
        kwargs = {"method": method, "limit_direction": "both"}
        if method == "spline":
            non_na = ser.notna().sum()
            eff_order = min(order, max(1, non_na - 1))
            kwargs["order"] = eff_order
        try:
            filled.append(ser.interpolate(**kwargs).to_numpy())
        except Exception:
            # Fallback auf lineare Interpolation, falls Spline scheitert
            filled.append(ser.interpolate(method="linear", limit_direction="both").to_numpy())
    return filled


def evaluate_masking(original_list, filled_list, positions):
    # Berechnet MAE und RMSE zwischen Originalwerten und interpolierten Werten an maskierten Positionen.
    orig = []
    pred = []
    for idx, pos in positions:
        if idx >= len(original_list) or idx >= len(filled_list):
            continue
        if pos >= len(original_list[idx][2]) or pos >= len(filled_list[idx]):
            continue
        o = original_list[idx][2][pos]
        p = filled_list[idx][pos]
        if np.isnan(o) or np.isnan(p):
            continue
        orig.append(o)
        pred.append(p)
    if not orig:
        return {"mae": float("nan"), "rmse": float("nan"), "count": 0}
    orig_arr = np.array(orig)
    pred_arr = np.array(pred)
    abs_err = np.abs(orig_arr - pred_arr)
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(abs_err ** 2))),
        "count": len(abs_err),
    }

def interpolate_masked_linear(masked_list):
    # Füllt maskierte Werte mit linearer Interpolation.
    return interpolate_masked(masked_list, method="linear")


def interpolate_masked_spline(masked_list, order: int = 3):
    # Füllt maskierte Werte mit Spline-Interpolation.
    return interpolate_masked(masked_list, method="spline", order=order)


def compare_interpolation_methods(
    df: pd.DataFrame,
    missing_ratio_isolated: float,
    missing_ratio_sequential: float,
    trials: int = 100,
    base_seed: int = 42,
) -> dict[str, float]:
    # Vergleicht die Leistung von linearer und Spline-Interpolation.
    mae_lin_total = 0.0
    mse_lin_total = 0.0
    count_lin_total = 0

    mae_spl_total = 0.0
    mse_spl_total = 0.0
    count_spl_total = 0
    runs = 0

    def _mean_safe(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    def _fmt(x: float) -> str:
        return f"{x:.2f}" if not np.isnan(x) else "nan"

    base_rng = np.random.default_rng(base_seed)

    for t in range(trials):
        seed = int(base_rng.integers(0, 2**31 - 1))

        # Isolierte Lücken
        masked_iso, pos_iso, orig_iso = mask_isolated(df, missing_ratio_isolated, seed)
        filled_iso_lin = interpolate_masked_linear(masked_iso)
        filled_iso_spl = interpolate_masked_spline(masked_iso, order=3)
        eval_iso_lin = evaluate_masking(orig_iso, filled_iso_lin, pos_iso)
        eval_iso_spl = evaluate_masking(orig_iso, filled_iso_spl, pos_iso)

        # Sequenzielle Lücken
        masked_seq, pos_seq, orig_seq = mask_sequential(df, missing_ratio_sequential, seed)
        filled_seq_lin = interpolate_masked_linear(masked_seq)
        filled_seq_spl = interpolate_masked_spline(masked_seq, order=3)
        eval_seq_lin = evaluate_masking(orig_seq, filled_seq_lin, pos_seq)
        eval_seq_spl = evaluate_masking(orig_seq, filled_seq_spl, pos_seq)

        # Per-Trial-Ausgabe
        mae_linear_trial = _mean_safe([
            ev["mae"] for ev in (eval_iso_lin, eval_seq_lin) if ev["count"] > 0 and not np.isnan(ev["mae"])
        ])
        rmse_linear_trial = _mean_safe([
            ev["rmse"] for ev in (eval_iso_lin, eval_seq_lin) if ev["count"] > 0 and not np.isnan(ev["rmse"])
        ])
        mae_spline_trial = _mean_safe([
            ev["mae"] for ev in (eval_iso_spl, eval_seq_spl) if ev["count"] > 0 and not np.isnan(ev["mae"])
        ])
        rmse_spline_trial = _mean_safe([
            ev["rmse"] for ev in (eval_iso_spl, eval_seq_spl) if ev["count"] > 0 and not np.isnan(ev["rmse"])
        ])
        runs += 1

        line = (
            f"Durchgang {t+1}: "
            f"mae_linear: {_fmt(mae_linear_trial)} | "
            f"rmse_linear: {_fmt(rmse_linear_trial)} | "
            f"mae_spline: {_fmt(mae_spline_trial)} | "
            f"rmse_spline: {_fmt(rmse_spline_trial)}"
        )
        print(line)

        # Aggregation (gewichteter Mittelwert über alle Fehlstellen)
        for ev in (eval_iso_lin, eval_seq_lin):
            if ev["count"] > 0 and not np.isnan(ev["mae"]):
                mae_lin_total += ev["mae"] * ev["count"]
                mse_lin_total += (ev["rmse"] ** 2) * ev["count"]
                count_lin_total += ev["count"]

        for ev in (eval_iso_spl, eval_seq_spl):
            if ev["count"] > 0 and not np.isnan(ev["mae"]):
                mae_spl_total += ev["mae"] * ev["count"]
                mse_spl_total += (ev["rmse"] ** 2) * ev["count"]
                count_spl_total += ev["count"]

    mae_linear = float(mae_lin_total / count_lin_total) if count_lin_total else float("nan")
    rmse_linear = float(np.sqrt(mse_lin_total / count_lin_total)) if count_lin_total else float("nan")
    mae_spline = float(mae_spl_total / count_spl_total) if count_spl_total else float("nan")
    rmse_spline = float(np.sqrt(mse_spl_total / count_spl_total)) if count_spl_total else float("nan")

    def _fmt(x: float) -> float:
        return float("nan") if np.isnan(x) else round(x, 2)

    return {
        "mae_linear": _fmt(mae_linear),
        "rmse_linear": _fmt(rmse_linear),
        "mae_spline": _fmt(mae_spline),
        "rmse_spline": _fmt(rmse_spline),
        "runs": runs,
    }

def fill_remaining_linear(df: pd.DataFrame) -> pd.DataFrame:
    # Interpoliert alle NaN-Einträge je Land per linearer Interpolation.
    filled_groups: list[pd.DataFrame] = []
    filled_points: list[tuple[str, int, float]] = []
    for country, group in df.groupby("REF_AREA"):
        group_sorted = group.sort_values("TIME_PERIOD").copy()
        series = group_sorted.set_index("TIME_PERIOD")["OBS_VALUE"]
        if series.isna().any():
            interp = series.interpolate(method="linear", limit_direction="both")
            interp[series.notna()] = series[series.notna()]
            newly_filled = interp[series.isna()].dropna()
            if not newly_filled.empty:
                filled_points.extend((country, int(yr), float(val)) for yr, val in newly_filled.items())
            series = interp
        filled_group = group_sorted.copy()
        filled_group["OBS_VALUE"] = series.reindex(group_sorted["TIME_PERIOD"]).to_numpy()
        filled_group = filled_group.dropna(subset=["OBS_VALUE"])
        filled_groups.append(filled_group)

    df_imputed = pd.concat(filled_groups, ignore_index=True)
    df_imputed = df_imputed.sort_values(["REF_AREA", "TIME_PERIOD"]).reset_index(drop=True)
    if filled_points:
        print(f"{len(filled_points)} fehlende Werte mit linearer Interpolation gefüllt.")

    return df_imputed


def fill_remaining_spline(df: pd.DataFrame, order: int = 3) -> pd.DataFrame:
    # Interpoliert alle NaN-Einträge je Land per Spline-Interpolation.
    filled_groups: list[pd.DataFrame] = []
    filled_points: list[tuple[str, int, float]] = []
    for country, group in df.groupby("REF_AREA"):
        group_sorted = group.sort_values("TIME_PERIOD").copy()
        series = group_sorted.set_index("TIME_PERIOD")["OBS_VALUE"]
        if series.isna().any():
            non_na = series.notna().sum()
            if non_na >= 2:
                eff_order = min(order, max(1, non_na - 1))
                try:
                    interp = series.interpolate(method="spline", order=eff_order, limit_direction="both")
                except Exception:
                    interp = series.interpolate(method="linear", limit_direction="both")
            else:
                interp = series
            interp[series.notna()] = series[series.notna()]
            newly_filled = interp[series.isna()].dropna()
            if not newly_filled.empty:
                filled_points.extend((country, int(yr), float(val)) for yr, val in newly_filled.items())
            series = interp
        filled_group = group_sorted.copy()
        filled_group["OBS_VALUE"] = series.reindex(group_sorted["TIME_PERIOD"]).to_numpy()
        filled_group = filled_group.dropna(subset=["OBS_VALUE"])
        filled_groups.append(filled_group)

    df_imputed = pd.concat(filled_groups, ignore_index=True)
    df_imputed = df_imputed.sort_values(["REF_AREA", "TIME_PERIOD"]).reset_index(drop=True)
    if filled_points:
        print(f"{len(filled_points)} fehlende Werte mit Spline-Interpolation gefüllt.")
        
    return df_imputed
