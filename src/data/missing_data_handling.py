import pandas as pd
import numpy as np
import numpy as np

SEP = "=" * 40


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def analyse_lengths(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby('REF_AREA_LABEL')['year']
        .nunique()
        .reset_index(name='num_years')
        .sort_values('num_years', ascending=False)
    )


def find_gaps(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(['REF_AREA_LABEL', 'year'])
    gaps = []
    for country, group in df_sorted.groupby('REF_AREA_LABEL'):
        years = group['year'].dropna().astype(int).sort_values()
        expected = range(years.min(), years.max() + 1)
        missing = sorted(set(expected).difference(years))
        if missing:
            gaps.append({
                'country': country,
                'first_year': int(years.min()),
                'last_year': int(years.max()),
                'missing_years': missing,
            })
    gap_df = pd.DataFrame(gaps)
    print("================================================")
    print("Länder mit Lücken in den Zeitreihen:")
    print("================================================")
    print(gap_df)
    return gap_df


def trim_by_missing_ratio(df: pd.DataFrame, threshold: float = 0.2) -> tuple[pd.DataFrame, list[tuple[str, int, int]], int, int]:
    """Kürzt Reihen vom Anfang, bis der Fehlanteil <= threshold ist, und entfernt verbleibende NaNs."""
    trimmed_groups = []
    cut_events: list[tuple[str, int, int]] = []  # (country, from_year, to_year)
    total_before = 0
    total_after = 0
    removed_na_after_cut = 0

    for country, group in df.groupby('REF_AREA_LABEL'):
        group_sorted = group.sort_values('year')
        full_years = range(group_sorted['year'].min(), group_sorted['year'].max() + 1)
        series = (
            group_sorted
            .set_index('year')['OBS_VALUE']
            .reindex(full_years)
        )
        total_before += len(series)

        removed_years = []
        # Kürzen vom Anfang, solange Quote über Schwelle oder erste/letzte Werte fehlen
        while len(series) > 0:
            total_len = len(series)
            missing_ratio = series.isna().sum() / total_len if total_len else 0
            first_missing = pd.isna(series.iloc[0])
            last_missing = pd.isna(series.iloc[-1])
            if missing_ratio <= threshold and not first_missing and not last_missing:
                break
            # entferne das erste Jahr
            removed_years.append(int(series.index[0]))
            series = series.iloc[1:]
        # Sicherstellen, dass keine trailing NaNs verbleiben
        while len(series) > 0 and pd.isna(series.iloc[-1]):
            removed_years.append(int(series.index[-1]))
            series = series.iloc[:-1]
        if removed_years:
            removed_years = sorted(removed_years)
            start = removed_years[0]
            prev = removed_years[0]
            for yr in removed_years[1:]:
                if yr == prev + 1:
                    prev = yr
                else:
                    cut_events.append((country, start, prev))
                    start = prev = yr
            cut_events.append((country, start, prev))
        if series.empty:
            print()
            print(f"Warnung: {country} nach Schnitt keine Daten mehr vorhanden.")
            continue

        # Restliche NaNs belassen (werden später interpoliert); nur trailing NaNs wurden oben entfernt
        # Zähle ggf. entfernte trailing NaNs
        removed_na_after_cut += 0

        total_after += len(series)
        trimmed = series.reset_index()
        trimmed['REF_AREA_LABEL'] = country
        trimmed_groups.append(trimmed)

    df_trimmed = (
        pd.concat(trimmed_groups, ignore_index=True)
        [['REF_AREA_LABEL', 'year', 'OBS_VALUE']]
        .sort_values(['REF_AREA_LABEL', 'year'])
        .reset_index(drop=True)
    )
    print()
    print("================================================")
    print("Kürzung nach Missing-Anteil:")
    print("================================================")
    print(f"Anzahl Zeilen vor Kürzung: {total_before}")
    print(f"Anzahl Zeilen nach Kürzung: {total_after}")
    print(f"Entfernte Zeilen durch Kürzung: {total_before - total_after}")
    # removed_na_after_cut nicht mehr genutzt (trailing NaNs fließen in removed_years ein)
    if cut_events:
        print()
        print("================================================")
        print("Abgeschnittene Reihen (>20 % fehlend/ Zeitreihe):")
        print("================================================")
        # Intervalle pro Land zusammenfassen (auch angrenzend)
        per_country = {}
        for country, start, end in cut_events:
            per_country.setdefault(country, []).append((start, end))
        for country, intervals in per_country.items():
            merged = []
            for s, e in sorted(intervals):
                if not merged or s > merged[-1][1] + 1:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)
            for s, e in merged:
                if s == e:
                    print(f"- {country}: Jahr {s} entfernt")
                else:
                    print(f"- {country}: von Jahr {s} bis {e} entfernt")
    return df_trimmed, cut_events, total_before, total_after


def fill_isolated_linear(df: pd.DataFrame) -> pd.DataFrame:
    filled_groups = []
    filled_points: list[tuple[str, int, float]] = []
    for country, group in df.groupby('REF_AREA_LABEL'):
        series = (
            group
            .sort_values('year')
            .set_index('year')['OBS_VALUE']
        )
        missing_mask = series.isna()
        if missing_mask.any():
            run_lengths = missing_mask.groupby(missing_mask.ne(missing_mask.shift()).cumsum()).transform('sum')
            max_run = run_lengths[missing_mask].max()
            if max_run == 1:
                series_interp = series.interpolate(method='linear', limit_area='inside')
                newly_filled = series_interp.loc[series_interp.index[missing_mask]].dropna()
                for yr, val in newly_filled.items():
                    filled_points.append((country, int(yr), float(val)))
                series = series_interp

        filled = series.reset_index()
        filled['REF_AREA_LABEL'] = country
        filled_groups.append(filled)

    df_filled = (
        pd.concat(filled_groups, ignore_index=True)
        [['REF_AREA_LABEL', 'year', 'OBS_VALUE']]
        .sort_values(['REF_AREA_LABEL', 'year'])
        .reset_index(drop=True)
    )
    print()
    print("================================================")
    print("Gefüllte Zeitreihen (isolierte Lücken):")
    print("================================================")
    if filled_points:
        for country, yr, val in filled_points:
            print(f"- {country} {yr}: {val:.3f}")
    print()
    return df_filled


def fill_remaining_spline(df_filled: pd.DataFrame, order: int = 3) -> pd.DataFrame:
    """Füllt nach der linearen Behandlung verbliebene längere Lücken per Spline (nur innenliegend)."""
    filled_groups = []
    filled_points: list[tuple[str, int, float]] = []
    for country, group in df_filled.groupby('REF_AREA_LABEL'):
        series = (
            group
            .sort_values('year')
            .set_index('year')['OBS_VALUE']
        )

        if series.isna().any() and series.notna().sum() >= order + 1:
            missing_mask = series.isna()
            run_lengths = missing_mask.groupby(missing_mask.ne(missing_mask.shift()).cumsum()).transform('sum')
            target_mask = missing_mask & (run_lengths >= 2)
            if target_mask.any():
                try:
                    series_interp = series.interpolate(
                        method='spline',
                        order=order,
                        limit_direction='both',
                        limit_area=None,
                    )
                except Exception:
                    series_interp = series.interpolate(method='linear', limit_direction='both')
                series_interp[series.notna()] = series[series.notna()]
                newly_filled = series_interp[target_mask].dropna()
                for yr, val in newly_filled.items():
                    filled_points.append((country, int(yr), float(val)))
                # Nur die Ziel-Lücken übernehmen, Rest unverändert lassen
                series[target_mask] = series_interp[target_mask]
        # Verbleibende NaNs (nicht gefüllt) entfernen
        series = series.dropna()
        filled = series.reset_index()
        filled['REF_AREA_LABEL'] = country
        filled_groups.append(filled)

    df_filled_spline = (
        pd.concat(filled_groups, ignore_index=True)
        [['REF_AREA_LABEL', 'year', 'OBS_VALUE']]
        .sort_values(['REF_AREA_LABEL', 'year'])
        .reset_index(drop=True)
    )
    print("================================================")
    print("Gefüllte Zeitreihen (Spline für längere Lücken):")
    print("================================================")
    print()
    if filled_points:
        for country, yr, val in filled_points:
            print(f"- {country} {yr}: {val:.3f}")
    return df_filled_spline


def fill_remaining_linear(df_filled: pd.DataFrame) -> pd.DataFrame:
    """Füllt längere Lücken per linearer Interpolation (nur Runs >=2)."""
    filled_groups = []
    filled_points: list[tuple[str, int, float]] = []
    for country, group in df_filled.groupby('REF_AREA_LABEL'):
        series = (
            group
            .sort_values('year')
            .set_index('year')['OBS_VALUE']
        )

        if series.isna().any():
            missing_mask = series.isna()
            run_lengths = missing_mask.groupby(missing_mask.ne(missing_mask.shift()).cumsum()).transform('sum')
            target_mask = missing_mask & (run_lengths >= 2)
            if target_mask.any():
                series_interp = series.interpolate(method='linear', limit_direction='both')
                series_interp[series.notna()] = series[series.notna()]
                newly_filled = series_interp[target_mask].dropna()
                for yr, val in newly_filled.items():
                    filled_points.append((country, int(yr), float(val)))
                series[target_mask] = series_interp[target_mask]
        series = series.dropna()
        filled = series.reset_index()
        filled['REF_AREA_LABEL'] = country
        filled_groups.append(filled)

    df_filled_linear = (
        pd.concat(filled_groups, ignore_index=True)
        [['REF_AREA_LABEL', 'year', 'OBS_VALUE']]
        .sort_values(['REF_AREA_LABEL', 'year'])
        .reset_index(drop=True)
    )
    if filled_points:
        print()
        print("================================================")
        print("Gefüllte Zeitreihen (sequenzielle Lücken):")
        print("================================================")
        for country, yr, val in filled_points:
            print(f"- {country} {yr}: {val:.3f}")
    return df_filled_linear


# Vergleich linear vs. Spline für 2er-Sequenzen
def _load_complete_series(path: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    df = pd.read_csv(path)
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for country, group in df.groupby("REF_AREA_LABEL"):
        years = group["year"].astype(int)
        full_years = np.arange(years.min(), years.max() + 1)
        series = group.set_index("year")["OBS_VALUE"].reindex(full_years)
        if series.isna().any() or len(series) < 4:
            continue
        out.append((country, full_years, series.to_numpy()))
    return out


def _mask_blocks(series_list, fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    masked_list = []
    positions = []
    for idx, (country, years, values) in enumerate(series_list):
        vals = values.copy()
        valid_starts = []
        for j in range(1, len(vals) - 2):
            if (
                not np.isnan(vals[j - 1])
                and not np.isnan(vals[j])
                and not np.isnan(vals[j + 1])
                and not np.isnan(vals[j + 2])
            ):
                valid_starts.append(j)
        if not valid_starts:
            masked_list.append((country, years, vals))
            continue
        target = max(1, int(len(valid_starts) * fraction))
        available = set(valid_starts)
        chosen = []
        while available and len(chosen) < target:
            s = rng.choice(sorted(available))
            chosen.append(s)
            for x in (s - 1, s, s + 1, s + 2):
                available.discard(x)
        for s in chosen:
            for c in (s, s + 1):
                positions.append((idx, c))
                vals[c] = np.nan
        masked_list.append((country, years, vals))
    return masked_list, positions


def _interp(series_list, method: str, order: int = 3):
    out = []
    for _, years, vals in series_list:
        ser = pd.Series(vals, index=years).sort_index()
        kwargs = {"method": method, "limit_direction": "both"}
        if method == "spline":
            # Effektive Ordnung nur, wenn genügend Stützpunkte vorhanden sind
            non_na = ser.notna().sum()
            eff_order = min(order, max(1, non_na - 1))
            kwargs["order"] = eff_order
        try:
            out.append(ser.interpolate(**kwargs).to_numpy())
        except Exception:
            # Fallback auf linear, falls Spline scheitert
            out.append(ser.interpolate(method="linear", limit_direction="both").to_numpy())
    return out


def _eval(series_list, filled_list, positions):
    orig = []
    pred = []
    for idx, pos in positions:
        o = series_list[idx][2][pos]
        p = filled_list[idx][pos]
        if np.isnan(o) or np.isnan(p):
            continue
        orig.append(o)
        pred.append(p)
    if not orig:
        return {"mae": float("nan"), "rmse": float("nan")}
    orig_arr = np.array(orig)
    pred_arr = np.array(pred)
    abs_err = np.abs(orig_arr - pred_arr)
    return {"mae": float(abs_err.mean()), "rmse": float(np.sqrt(np.mean(abs_err ** 2)))}


def compare_sequence_methods(data_path: str, trials: int = 50, base_seed: int = 42, fraction: float = 0.05):
    series_list = _load_complete_series(data_path)
    mae_lin = []
    mae_spl = []
    rmse_lin = []
    rmse_spl = []
    for t in range(trials):
        seed = base_seed + t
        masked, pos = _mask_blocks(series_list, fraction=fraction, seed=seed)
        filled_lin = _interp(masked, method="linear")
        filled_spl = _interp(masked, method="spline", order=3)
        m_lin = _eval(series_list, filled_lin, pos)
        m_spl = _eval(series_list, filled_spl, pos)
        mae_lin.append(m_lin["mae"])
        mae_spl.append(m_spl["mae"])
        rmse_lin.append(m_lin["rmse"])
        rmse_spl.append(m_spl["rmse"])
    return (
        float(np.nanmean(mae_lin)),
        float(np.nanmean(mae_spl)),
        float(np.nanmean(rmse_lin)),
        float(np.nanmean(rmse_spl)),
    )


# Vergleich linear vs. Spline auf isolierten Masken
def _load_complete_series(path: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    df = pd.read_csv(path)
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for country, group in df.groupby("REF_AREA_LABEL"):
        years = group["year"].astype(int)
        full_years = np.arange(years.min(), years.max() + 1)
        series = group.set_index("year")["OBS_VALUE"].reindex(full_years)
        if series.isna().any() or len(series) < 4:
            continue
        out.append((country, full_years, series.to_numpy()))
    return out


def _mask_isolated(series_list, fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    masked_list = []
    positions = []
    for idx, (country, years, values) in enumerate(series_list):
        vals = values.copy()
        valid = []
        for j in range(1, len(vals) - 1):
            if not np.isnan(vals[j]) and not np.isnan(vals[j - 1]) and not np.isnan(vals[j + 1]):
                valid.append(j)
        if len(valid) < 3:
            masked_list.append((country, years, vals))
            continue
        target = max(1, int(len(valid) * fraction))
        available = set(valid)
        chosen = []
        while available and len(chosen) < target:
            pos = rng.choice(sorted(available))
            chosen.append(pos)
            available.discard(pos)
            available.discard(pos - 1)
            available.discard(pos + 1)
        for c in chosen:
            positions.append((idx, c))
            vals[c] = np.nan
        masked_list.append((country, years, vals))
    return masked_list, positions


def _interpolate(series_list, method: str, order: int | None = None):
    filled = []
    for _, years, vals in series_list:
        ser = pd.Series(vals, index=years).sort_index()
        kwargs = {"method": method, "limit_area": "inside"}
        if method == "spline":
            kwargs["order"] = order or 3
        filled.append(ser.interpolate(**kwargs).to_numpy())
    return filled


def _evaluate(series_list, filled_list, positions):
    orig = []
    pred = []
    for idx, pos in positions:
        o = series_list[idx][2][pos]
        p = filled_list[idx][pos]
        if np.isnan(o) or np.isnan(p):
            continue
        orig.append(o)
        pred.append(p)
    if not orig:
        return {"mae": float("nan"), "rmse": float("nan")}
    orig_arr = np.array(orig)
    pred_arr = np.array(pred)
    abs_err = np.abs(orig_arr - pred_arr)
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(abs_err ** 2))),
    }


def compare_isolated_methods(data_path: str, trials: int = 50, base_seed: int = 42, fraction: float = 0.05):
    series_list = _load_complete_series(data_path)
    mae_lin, mae_spl, rmse_lin, rmse_spl = [], [], [], []
    for t in range(trials):
        trial_seed = base_seed + t
        masked, pos = _mask_isolated(series_list, fraction=fraction, seed=trial_seed)
        filled_lin = _interpolate(masked, method="linear")
        filled_spl = _interpolate(masked, method="spline", order=3)
        m_lin = _evaluate(series_list, filled_lin, pos)
        m_spl = _evaluate(series_list, filled_spl, pos)
        mae_lin.append(m_lin["mae"])
        mae_spl.append(m_spl["mae"])
        rmse_lin.append(m_lin["rmse"])
        rmse_spl.append(m_spl["rmse"])
    return (
        float(np.nanmean(mae_lin)),
        float(np.nanmean(mae_spl)),
        float(np.nanmean(rmse_lin)),
        float(np.nanmean(rmse_spl)),
    )


def fill_isolated_spline(df: pd.DataFrame, order: int = 3) -> pd.DataFrame:
    filled_groups = []
    filled_points: list[tuple[str, int, float]] = []
    for country, group in df.groupby('REF_AREA_LABEL'):
        series = (
            group
            .sort_values('year')
            .set_index('year')['OBS_VALUE']
        )
        missing_mask = series.isna()
        if missing_mask.any():
            run_lengths = missing_mask.groupby(missing_mask.ne(missing_mask.shift()).cumsum()).transform('sum')
            max_run = run_lengths[missing_mask].max()
            if max_run == 1:
                try:
                    series_interp = series.interpolate(method='spline', order=order, limit_area='inside')
                except Exception:
                    series_interp = series.interpolate(method='linear', limit_area='inside')
                newly = series_interp.loc[series_interp.index[missing_mask]].dropna()
                for yr, val in newly.items():
                    filled_points.append((country, int(yr), float(val)))
                series = series_interp
        filled = series.reset_index()
        filled['REF_AREA_LABEL'] = country
        filled_groups.append(filled)

    df_filled = (
        pd.concat(filled_groups, ignore_index=True)
        [['REF_AREA_LABEL', 'year', 'OBS_VALUE']]
        .sort_values(['REF_AREA_LABEL', 'year'])
        .reset_index(drop=True)
    )
    if filled_points:
        print("Aufgefüllte Punkte (Spline, isoliert):")
        for country, yr, val in filled_points:
            print(f"- {country} {yr}: {val:.3f}")
    return df_filled


def save_clean(df_filled: pd.DataFrame, path: str, decimals: int | None = None) -> None:
    output = df_filled[['REF_AREA_LABEL', 'year', 'OBS_VALUE']].copy()
    if decimals is not None:
        output['OBS_VALUE'] = output['OBS_VALUE'].round(decimals)
    output.to_csv(path, index=False)
    print()
    print("================================================")
    print(f"CSV-Datei abgespeichert unter: {path}")
    print("================================================")



def _max_nan_run(series: pd.Series) -> float:
    mask = series.isna()
    if not mask.any():
        return float("nan")
    run_lengths = mask.groupby(mask.ne(mask.shift()).cumsum()).transform('sum')
    return float(run_lengths[mask].max())


def main():
    df = load_data("data/pharma_consumption.csv")
    _ = find_gaps(df)

    df_trimmed, _, _, _ = trim_by_missing_ratio(df, threshold=0.2)

    # Prüfen, ob isolierte Lücken vorhanden sind
    has_isolated = False
    for _, group in df_trimmed.groupby('REF_AREA_LABEL'):
        series = (
            group
            .sort_values('year')
            .set_index('year')['OBS_VALUE']
        )
        missing_mask = series.isna()
        if missing_mask.any():
            run_lengths = missing_mask.groupby(missing_mask.ne(missing_mask.shift()).cumsum()).transform('sum')
            if (run_lengths[missing_mask] == 1).any():
                has_isolated = True
                break

    if has_isolated:
        avg_mae_lin, avg_mae_spl, avg_rmse_lin, avg_rmse_spl = compare_isolated_methods(
            "data/pharma_consumption.csv", trials=50, base_seed=42, fraction=0.05
        )
        print()
        print("================================================")
        print("Vergleich von linearer und Spline Interpolation für isolierte Lücken (MAE/RMSE, Durchschnitt):")
        print("================================================")
        print(f"Linear  MAE={avg_mae_lin:.4f}, RMSE={avg_rmse_lin:.4f}")
        print(f"Spline  MAE={avg_mae_spl:.4f}, RMSE={avg_rmse_spl:.4f}")

        if avg_mae_lin <= avg_mae_spl:
            print()
            print("-> Nutze lineare Interpolation für isolierte Lücken.")
            df_filled_isolated = fill_isolated_linear(df_trimmed)
        else:
            print()
            print("-> Nutze Spline-Interpolation für isolierte Lücken.")
            df_filled_isolated = fill_isolated_spline(df_trimmed)
    else:
        df_filled_isolated = df_trimmed

    # Prüfen, ob längere Sequenzen (Runs >=2) vorhanden sind
    has_sequences = False
    for _, group in df_filled_isolated.groupby('REF_AREA_LABEL'):
        series = (
            group
            .sort_values('year')
            .set_index('year')['OBS_VALUE']
        )
        missing_mask = series.isna()
        if missing_mask.any():
            run_lengths = missing_mask.groupby(missing_mask.ne(missing_mask.shift()).cumsum()).transform('sum')
            if (run_lengths[missing_mask] >= 2).any():
                has_sequences = True
                break
    
    if has_sequences:
        avg_seq_lin, avg_seq_spl, avg_seq_rmse_lin, avg_seq_rmse_spl = compare_sequence_methods(
            "data/pharma_consumption.csv", trials=50, base_seed=42, fraction=0.05
        )
        print()
        print("================================================")
        print("Vergleich von linearer und Spline Interpolation für sequenzielle Lücken (MAE/RMSE, Durchschnitt):")
        print("================================================")
        print(f"Linear  MAE={avg_seq_lin:.4f}, RMSE={avg_seq_rmse_lin:.4f}")
        print(f"Spline  MAE={avg_seq_spl:.4f}, RMSE={avg_seq_rmse_spl:.4f}")

        if avg_seq_lin <= avg_seq_spl:
            print()
            print("-> Nutze lineare Interpolation für längere Lücken.")
            df_final = fill_remaining_linear(df_filled_isolated)
        else:
            print()
            print("-> Nutze Spline-Interpolation für längere Lücken.")
            df_final = fill_remaining_spline(df_filled_isolated)
    else:
        df_final = df_filled_isolated

    save_clean(df_final, "data/pharma_consumption_imputed.csv", decimals=1) #Datei Pfad später im Abgabeordner noch anpassen


if __name__ == "__main__":
    main()
