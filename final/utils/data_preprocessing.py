import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

##########################################
# Initiales Entfernen irrelevanter Spalten
##########################################

def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Entfernt irrelevante Spalten aus dem DataFrame
    columns_to_drop = [
        'REF_AREA_LABEL',
        'MEASURE',
        'MEASURE_LABEL',
        'UNIT_MEASURE',
        'UNIT_MEASURE_LABEL',
        'MARKET_TYPE',
        'MARKET_TYPE_LABEL',
        'PHARMACEUTICAL_LABEL',
        'UNIT_MULT',
        'UNIT_MULT_LABEL',
        'OBS_STATUS_LABEL',
        'OBS_STATUS2_LABEL',
        'OBS_STATUS3_LABEL',
    ]
    df_cleaned = df.drop(columns=columns_to_drop, errors="raise").copy()
    return columns_to_drop, df_cleaned

#########################################################
# Überprüfung der Datenhomogenität anhand von Statusflags
#########################################################

def compute_status_report(df: pd.DataFrame):
    # Überprüfe auf 'D' in den Statusspalten
    status_cols = ['OBS_STATUS', 'OBS_STATUS2', 'OBS_STATUS3']
    existing = [c for c in status_cols if c in df.columns]
    if existing:
        # Strips/normalisiert Werte, damit auch Varianten wie " d " gezählt werden
        status_norm = df[existing].apply(lambda s: s.astype(str).str.strip().str.upper())
        mask_d = status_norm.eq('D').any(axis=1)
    else:
        mask_d = pd.Series(False, index=df.index)
    total_rows = len(df)
    total_d = mask_d.sum()
    share_total = (total_d / total_rows * 100) if total_rows else 0
    per_code_total = df.groupby('PHARMACEUTICAL').size()
    per_code_d = df[mask_d].groupby('PHARMACEUTICAL').size()

    report = (
        pd.DataFrame({
            'rows_with_D': per_code_d,
            'rows_total': per_code_total,
        })
        .fillna(0)
        .assign(share_D=lambda x: x['rows_with_D'] / x['rows_total'].replace(0, pd.NA))
        .fillna({'share_D': 0})
        .sort_values('share_D', ascending=True)
    )

    return report, total_d, total_rows, share_total


def plot_share_d(report: pd.DataFrame):
    # Visualisierung der Anteile
    top10 = (
        report.reset_index()
        .rename(columns={'index': 'PHARMACEUTICAL'})
        .sort_values('share_D')
        .head(10)
    )
    # Einheitlich kräftiges Grün für alle Balken
    base_color = (0/255, 130/255, 70/255, 1.0)
    colors = [base_color] * len(top10)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))  # etwas kompakter in der Höhe
    font_kwargs = {"fontfamily": "Times New Roman", "fontsize": 12}
    sns.barplot(
        data=top10,
        y='PHARMACEUTICAL',
        x='share_D',
        palette=colors,
        ax=ax,
    )
    ax.set_xlabel('Anteil D-Flags (%)', **font_kwargs)
    ax.set_ylabel('ATC-Code', **font_kwargs)
    ax.set_title('D-Flags je ATC-Code (Top 10 mit niedrigstem Anteil)', pad=10, **font_kwargs)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(axis='both', labelsize=12)
    plt.setp(ax.get_xticklabels(), fontfamily="Times New Roman", fontsize=12)
    plt.setp(ax.get_yticklabels(), fontfamily="Times New Roman", fontsize=12)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    plt.show()

#####################################################
# Zu heterogene ATC-Codes aus dem Datensatz entfernen
#####################################################

def pick_lowest_atc_code(report: pd.DataFrame) -> str:
    # Wähle den ATC-Code der Ebene 2 mit dem niedrigsten Anteil an D-Flags
    if report.empty or 'share_D' not in report.columns:
        raise ValueError("Report ist leer oder enthält keine Spalte 'share_D'")
    for code in report.sort_values('share_D').index:
        atc_code = str(code).strip()
        if len(atc_code) >= 2:
            return atc_code
    raise ValueError("Kein ATC-Code mit mindestens 2 Zeichen gefunden.")


def filter_to_j01(df: pd.DataFrame) -> pd.DataFrame:
    # Filtere den Datensatz auf den ATC-Code J01 (Antibiotika)
    mask = df['PHARMACEUTICAL'].astype(str).str.upper().str.startswith('J01')
    return df.loc[mask].copy()

################################################
# Analyse des Datensatzes für homogenen ATC Code
################################################

def analyze_j01_dataset(df_j01: pd.DataFrame):
    # Anzahl Länder und Zeitabdeckung je Land ausgeben
    num_countries = df_j01['REF_AREA'].nunique()
    coverage = (
        df_j01.groupby('REF_AREA')['TIME_PERIOD']
        .agg(min_year='min', max_year='max', n_periods='count')
        .sort_values('n_periods', ascending=False)
    )

    # Definition differs (D-Flags) je Land
    status_cols = ['OBS_STATUS', 'OBS_STATUS2', 'OBS_STATUS3']
    existing_status = [c for c in status_cols if c in df_j01.columns]
    if existing_status:
        status_norm = df_j01[existing_status]
        mask_d = status_norm.eq('D').any(axis=1)
        defdiff_per_country = df_j01[mask_d].groupby('REF_AREA').size()
    else:
        defdiff_per_country = pd.Series(dtype=int)
    defdiff_per_country = defdiff_per_country.reindex(coverage.index, fill_value=0)

    # Brüche (Status 'B') je Land in den Statusspalten
    if existing_status:
        mask_b = status_norm.eq('B').any(axis=1)
        time_breaks_per_country = df_j01[mask_b].groupby('REF_AREA').size()
    else:
        time_breaks_per_country = pd.Series(dtype=int)
    time_breaks_per_country = time_breaks_per_country.reindex(coverage.index, fill_value=0)

    # Fehlende Jahre zwischen min_year und max_year je Land ermitteln
    missing_years_per_country = {}
    missing_counts = {}
    for country, grp in df_j01.groupby('REF_AREA'):
        years = grp['TIME_PERIOD'].dropna().astype(int).unique()
        if years.size == 0:
            missing_years_per_country[country] = []
            missing_counts[country] = 0
            continue
        min_y, max_y = years.min(), years.max()
        full_span = set(range(min_y, max_y + 1))
        missing = sorted(full_span - set(years))
        missing_years_per_country[country] = missing
        missing_counts[country] = len(missing)
    missing_counts = pd.Series(missing_counts).reindex(coverage.index, fill_value=0).astype(int)

    # Zusatzinfos in Coverage integrieren
    coverage = coverage.assign(
        time_break_rows=time_breaks_per_country,
        defdiff_rows=defdiff_per_country,
        missing_years=missing_counts,
    )
    return num_countries, coverage, defdiff_per_country, time_breaks_per_country, missing_counts, missing_years_per_country

######################################
# Umgang mit Brüchen in den Zeitreihen
######################################

def handle_time_series_breaks(df_j01: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, dict[str, list[int]]]:
    # Schneidet Zeitreihen an den Time-Series-Breaks (Status 'B') ab
    status_cols = [c for c in ['OBS_STATUS', 'OBS_STATUS2', 'OBS_STATUS3'] if c in df_j01.columns]

    mask_b = df_j01[status_cols].eq('B').any(axis=1)
    b_rows = df_j01[mask_b].copy()

    if b_rows.empty:
        print("Keine Brüche gefunden.")
        return df_j01.copy()

    summary = (
        b_rows.groupby('REF_AREA')['TIME_PERIOD']
        .agg(last_break='max', years_with_breaks=lambda s: ", ".join(str(int(y)) for y in sorted(set(s))))
        .sort_index()
    )
    break_years_by_area = {
        area: [int(y.strip()) for y in str(years).split(",") if y.strip()]
        for area, years in summary["years_with_breaks"].items()
    }
    # Anzahl der Zeilen, die pro Land bis inkl. letztem Break-Jahr abgeschnitten werden
    cut_counts = {}
    for country, year in summary['last_break'].items():
        cut_mask = (df_j01['REF_AREA'] == country) & (df_j01['TIME_PERIOD'] <= year)
        cut_counts[country] = int(cut_mask.sum())
    summary['rows_cut'] = summary.index.map(lambda c: cut_counts.get(c, 0))

    # Schneide pro Land ab: alles bis inkl. letztem Break-Jahr wird entfernt
    df_j01_filtered = df_j01.copy()
    for country, year in summary['last_break'].items():
        df_j01_filtered = df_j01_filtered[~((df_j01_filtered['REF_AREA'] == country) & (df_j01_filtered['TIME_PERIOD'] <= year))]

    total_cut = int(summary['rows_cut'].sum())
    
    return summary, df_j01_filtered, total_cut, break_years_by_area


def plot_series_with_breaks(
    df: pd.DataFrame,
    ref_area: str,
    break_years: list[int] | None = None,
    summary: pd.DataFrame | None = None,
    break_years_by_area: dict[str, list[int]] | None = None,
    time_col: str = "TIME_PERIOD",
    value_col: str = "OBS_VALUE",
    global_years_range: tuple[int, int] | None = None,
    global_ylim: tuple[float, float] | None = None,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    # Visualisiert die Zeitreihe eines Landes mit Markierung der Break-Jahre.
    # Fehlende Jahre werden als Luecken ohne Verbindungslinie dargestellt.
    if break_years is None:
        if break_years_by_area is not None and ref_area in break_years_by_area:
            break_years = break_years_by_area[ref_area]
        elif summary is not None and ref_area in summary.index:
            years_str = str(summary.loc[ref_area, "years_with_breaks"])
            break_years = [int(y.strip()) for y in years_str.split(",") if y.strip()]
        else:
            raise ValueError("break_years fehlt und keine Break-Info für REF_AREA gefunden.")

    data = df[df["REF_AREA"] == ref_area].sort_values(time_col)
    if data.empty:
        raise ValueError(f"Keine Daten für {ref_area} in Spalte REF_AREA.")
    if global_years_range is not None:
        min_year, max_year = global_years_range
    else:
        all_years = df[time_col].dropna().astype(int)
        if all_years.empty:
            raise ValueError(f"Keine gueltigen Jahre in Spalte {time_col}.")
        min_year = int(all_years.min())
        max_year = int(all_years.max())
    full_years = range(min_year, max_year + 1)
    series = (
        data.set_index(data[time_col].astype(int))[value_col]
        .reindex(full_years)
    )
    if xlim is None:
        xlim = (min_year, max_year)
    if ylim is None:
        if global_ylim is not None:
            ylim = global_ylim
        else:
            all_vals = df[value_col].dropna()
            if not all_vals.empty:
                ylim = (float(all_vals.min()), float(all_vals.max()))
    fig, ax = plt.subplots(figsize=(6.5, 1))
    ax.plot(list(full_years), series.to_numpy(), color="black", linewidth=1.5)
    present_mask = series.notna()
    if present_mask.any():
        ax.scatter(
            list(series.index[present_mask]),
            series[present_mask].to_numpy(),
            color="black",
            s=10,
            zorder=4,
        )
    for year in break_years:
        ax.axvline(year, color="red", linestyle="--", alpha=0.8, linewidth=1)
    font_kwargs = {"fontfamily": "Times New Roman", "fontsize": 12}
    ax.set_title(f"{ref_area} - Zeitreihe mit Brüchen", **font_kwargs)
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

############################################################
# Kürzen der Zeitreihen mit fehlenden Anteilen von über 20 %
############################################################

def trim_by_missing_ratio(df: pd.DataFrame, threshold: float = 0.2) -> tuple[pd.DataFrame, list[tuple[str, int, int]], pd.DataFrame]:
    # Kürzt Zeitreihen vom Anfangan solange, bis der Fehlanteil kleiner gleich dem threshold ist.
    trimmed_groups = []
    cut_events: list[tuple[str, int, int]] = []  # (country, from_year, to_year)
    removed_na_after_cut = 0

    for country, group in df.groupby('REF_AREA'):
        group_sorted = group.sort_values('TIME_PERIOD')
        full_years = range(group_sorted['TIME_PERIOD'].min(), group_sorted['TIME_PERIOD'].max() + 1)
        series = (
            group_sorted
            .set_index('TIME_PERIOD')['OBS_VALUE']
            .reindex(full_years)
        )
        removed_years = []
        # Kürzen von Anfang an, solange Missing-Quote über Schwelle
        while len(series) > 0:
            total_len = len(series)
            missing_ratio = series.isna().sum() / total_len if total_len else 0
            if missing_ratio <= threshold:
                break
            # entferne das erste Jahr
            removed_years.append(int(series.index[0]))
            series = series.iloc[1:]
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

        # Restliche NaNs belassen (werden später interpoliert)
        removed_na_after_cut += 0

        trimmed = series.reset_index()
        trimmed['REF_AREA'] = country
        trimmed_groups.append(trimmed)

    df_trimmed = (
        pd.concat(trimmed_groups, ignore_index=True)
        [['REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']]
        .sort_values(['REF_AREA', 'TIME_PERIOD'])
        .reset_index(drop=True)
    )

    # removed_na_after_cut nicht mehr genutzt (trailing NaNs fließen in removed_years ein)
    if cut_events:
        print()
        print("Abgeschnittene Reihen (>20 % fehlend/ Zeitreihe):")
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
    summary = missing_data_handling(df)
    countries_above = summary.loc[summary["missing_quote"] > threshold].reset_index(drop=True)

    return df_trimmed, cut_events, countries_above
