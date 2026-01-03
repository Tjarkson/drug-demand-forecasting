import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

##################
# Feature-Pipeline
##################

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wendet alle Feature-Engineering-Schritte sequentiell an und gibt einen erweiterten DataFrame zurück.
    Jeder Schritt arbeitet immutabel (kopiert intern) und fügt neue Spalten hinzu.
    """
    df_feat = (
        df
        .pipe(add_d_flag_column, flag_col="D-Flag")
        .pipe(add_status_flag_column, flag_col="Status-Flag")
        .pipe(add_lag_features, lags=(1, 2))
        .pipe(add_rolling_features, window=3)
        # Weitere Feature-Schritte hier per .pipe(...) anhängen.
    )
    return df_feat

########################
# Status-Flag hinzufügen
########################

def add_status_flag_column(
    df: pd.DataFrame,
    flag_col: str = "Status-Flag",
    trigger_values: tuple[str, ...] = ("P", "E", "D"),
) -> pd.DataFrame:
    # Fügt eine Spalte mit 1/0 hinzu, wenn eine der Status-Spalten einen der Trigger-Werte enthält.

    status_cols = ["OBS_STATUS", "OBS_STATUS2", "OBS_STATUS3"]
    df_status_flag = df.copy()
    existing = [c for c in status_cols if c in df_status_flag.columns]

    if not existing:
        df_status_flag[flag_col] = 0
        return df_status_flag
    
    status_stripped = df_status_flag[existing].apply(lambda s: s.astype(str).str.strip())
    mask_trigger = status_stripped.apply(lambda s: s.isin(trigger_values)).any(axis=1)
    df_status_flag[flag_col] = mask_trigger.astype(int)
    print(f"Spalte '{flag_col}' hinzugefügt: {mask_trigger.sum()} von {len(df_status_flag)} Zeilen markiert.")

    return df_status_flag


def summarize_status_entries(
    df: pd.DataFrame,
    trigger_values: tuple[str, ...] = ("P", "E", "D"),
    status_cols: tuple[str, ...] = ("OBS_STATUS", "OBS_STATUS2", "OBS_STATUS3"),
) -> dict[str, pd.DataFrame | int]:
    # Zählt, wie viele Zeilen die Trigger-Werte enthalten, und wie viele Zeilen mehrere Trigger gleichzeitig haben.
   
    existing = [c for c in status_cols if c in df.columns]
    if not existing:
        summary = {
            "counts": pd.DataFrame(columns=["value", "count"]),
            "rows_with_any": 0,
            "rows_with_multiple": 0,
        }
        _print_status_summary(summary)
        return summary
    status_stripped = df[existing].apply(lambda s: s.astype(str).str.strip())
    counts_list = []
    for val in trigger_values:
        counts_list.append({"value": val, "count": int(status_stripped.eq(val).any(axis=1).sum())})
    counts_df = pd.DataFrame(counts_list)
    mask_per_col = status_stripped.apply(lambda s: s.isin(trigger_values))
    mask_any = mask_per_col.any(axis=1)
    mask_multiple = mask_per_col.sum(axis=1) > 1
    summary = {
        "counts": counts_df,
        "rows_with_any": int(mask_any.sum()),
        "rows_with_multiple": int(mask_multiple.sum()),
    }

    # Ausgabe direkt hier
    counts = summary["counts"]
    print("\nStatus-Counts (Zeilen mit mindestens einem Vorkommen):")
    if not counts.empty:
        for _, row in counts.iterrows():
            print(f"  {row['value']}: {int(row['count'])}")
    else:
        print("  Keine Status-Spalten vorhanden.")
    print(f"\nZeilen mit mindestens einem Trigger: {summary['rows_with_any']}")
    print(f"\nZeilen mit mehreren Trigger-Spalten: {summary['rows_with_multiple']}")
    return summary

########################
# Corona-Flag hinzufügen
########################

def plot_average_trend_over_years(
    df: pd.DataFrame,
    time_col: str = "TIME_PERIOD",
    value_col: str = "OBS_VALUE",
    corona_years: tuple[int, int] | None = (2020, 2021),
    title: str | None = None,
) -> pd.DataFrame:
    # Zeigt den mittleren Verlauf des Verbrauchsmengen über die Jahre hinweg an.

    df_year = df[[time_col, value_col]].dropna(subset=[time_col]).copy()
    if df_year.empty:
        raise ValueError("Keine Daten für die Trend-Visualisierung vorhanden.")

    df_year[time_col] = df_year[time_col].astype(int)
    agg = (
        df_year.groupby(time_col)[value_col]
        .agg(mean="mean", count="count")
        .sort_index()
        .reset_index()
    )
    print()
    # Breite analog zu den anderen Plots halten
    plt.figure(figsize=(6.5, 3))
    plt.plot(agg[time_col], agg["mean"], color="black", linewidth=1.8, label="Mittelwert")
    if corona_years is not None and len(corona_years) == 2:
        start, end = corona_years
        plt.axvspan(start - 0.5, end + 0.5, color="red", alpha=0.1, label="Corona-Jahre")
    plt.xlabel("Jahr")
    plt.ylabel(value_col)
    plt.title(title or "Mittlerer Verlauf je Jahr (alle Länder)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return agg

def add_corona_flag_column(
    df: pd.DataFrame,
    flag_col: str = "Corona-Flag",
    time_col: str = "TIME_PERIOD",
    corona_years: tuple[int, int] = (2020, 2021),
) -> pd.DataFrame:
    # Fügt eine Spalte mit 1/0 hinzu, wenn das Jahr in den Corona-Jahren liegt.

    df_corona_flag = df.copy()
    start_year, end_year = corona_years
    df_corona_flag[time_col] = df_corona_flag[time_col].astype(int)
    mask_corona = df_corona_flag[time_col].between(start_year, end_year)
    df_corona_flag[flag_col] = mask_corona.astype(int)
    print(f"Spalte '{flag_col}' hinzugefügt: {mask_corona.sum()} von {len(df_corona_flag)} Zeilen markiert.")

    return df_corona_flag

##########################
# Zeit-Features hinzufügen
##########################

def add_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2),
    group_col: str = "REF_AREA",
    value_col: str = "OBS_VALUE",
    time_col: str = "TIME_PERIOD",
) -> pd.DataFrame:
    # Fügt Lag-Features pro Land hinzu (lag_1, lag_2).

    df_lag = df.copy()
    df_lag = df_lag.sort_values([group_col, time_col])
    for lag in lags:
        df_lag[f"lag_{lag}"] = df_lag.groupby(group_col)[value_col].shift(lag)

    print(f"Lag-Features hinzugefügt: {len(df_lag)} Zeilen betroffen.")

    return df_lag


def add_rolling_features(
    df: pd.DataFrame,
    window: int = 3,
    group_col: str = "REF_AREA",
    value_col: str = "OBS_VALUE",
    time_col: str = "TIME_PERIOD",
) -> pd.DataFrame:
    # Fügt einen gleitenden Mittelwert pro Land hinzu (roll_mean_<window>).

    df_roll = df.copy()
    df_roll = df_roll.sort_values([group_col, time_col])
    df_roll[f"roll_mean_{window}"] = (
        df_roll.groupby(group_col)[value_col]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )
    print(f"\nGleitende Mittelwerte hinzugefügt: {len(df_roll)} Zeilen betroffen.")

    return df_roll

#######################################
# Gesundheitssektor-Features hinzufügen
#######################################






