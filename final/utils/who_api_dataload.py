
from pathlib import Path

import pandas as pd
import requests

# Basis: Projektverzeichnis final/
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
GHO_BASE = "https://ghoapi.azureedge.net/api"
PHARMA_PATH = DATA_DIR / "pharma_consumption_preprocessed.csv"
OUTPUT_PATH = DATA_DIR / "features_who.csv"

###############################################################################################
# Manuell Eingabe der gewünschten WHO-Indikatoren (Code in der API, Spaltenname in der Ausgabe)
###############################################################################################

INDICATORS = [
    {"code": "HWF_0001", "col": "Ärzte_pro_10000"},
    {"code": "HWF_0002", "col": "Ärzte"},
    {"code": "HWF_0004", "col": "Arzthelfer_pro_10000"},
    {"code": "WHS6_102", "col": "Krankenhausbetten_pro_10000"},
    {
        "code": "GLASSAMC_AWARE",
        "col": "Access_Anteil",
        "dim1_filter": "GLASSAMCAWARE_ANTIBIOTICSGROUP_GLASSAMC_AWARE_A",
    },
]

########################################
# Hilffunktionen zum Laden der WHO-Daten
########################################

def load_country_years(path: Path = PHARMA_PATH):
    pharma = pd.read_csv(path)
    # Erwartet Spalten REF_AREA und TIME_PERIOD in der Eingabedatei.
    pharma = pharma.dropna(subset=["REF_AREA", "TIME_PERIOD"])
    pharma["TIME_PERIOD"] = pd.to_numeric(pharma["TIME_PERIOD"], errors="coerce").astype("Int64")
    pharma = pharma.dropna(subset=["TIME_PERIOD"])
    pharma["TIME_PERIOD"] = pharma["TIME_PERIOD"].astype(int)
    countries = sorted(pharma["REF_AREA"].dropna().unique().tolist())
    country_years = {
        c: sorted(pharma.loc[pharma["REF_AREA"] == c, "TIME_PERIOD"].unique().astype(int).tolist())
        for c in countries
    }
    return countries, country_years


def normalize_indicator(df: pd.DataFrame) -> pd.DataFrame:
    # Normalisiert die Rohdaten eines Indikators in ein tidy DataFrame mit Spalten REF_AREA, TIME_PERIOD, numericValue.
    df = df.rename(columns={"SpatialDim": "REF_AREA", "TimeDim": "TIME_PERIOD"})

    value_col = None
    for cand in ("numericValue", "NumericValue", "value"):
        if cand in df.columns:
            value_col = cand
            break
    if value_col is None:
        raise ValueError("Keine Value-Spalte in Indikator-Daten gefunden")

    tidy = (
        df[["REF_AREA", "TIME_PERIOD", value_col]]
        .rename(columns={value_col: "numericValue"})
        .drop_duplicates(subset=["REF_AREA", "TIME_PERIOD"])
    )

    return tidy


def fetch_indicator_for_country(indicator_code: str, country_code: str, years: list[int], dim1_filter: str | None = None, timeout: int = 60) -> pd.DataFrame:
    # Ruft Daten für einen Indikator und ein Land von der WHO-API ab und filtert nach Jahren und Dim1 (optional).
    if not years:
        return pd.DataFrame()
    start, end = min(years), max(years)
    url = (
        f"{GHO_BASE}/{indicator_code}"
        f"?$filter=SpatialDim%20eq%20'{country_code}'%20and%20TimeDim%20ge%20{start}%20and%20TimeDim%20le%20{end}"
    )
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARN] Überspringe {indicator_code} für {country_code}: {exc}")
        return pd.DataFrame()
    payload = resp.json()
    df = pd.DataFrame(payload.get("value", []))
    if df.empty:
        return df
    if dim1_filter is not None:
        if "Dim1" not in df.columns:
            print(f"[WARN] Dim1-Spalte fehlt in {indicator_code} für {country_code}, Filter nicht anwendbar")
            return pd.DataFrame()
        df = df[df["Dim1"] == dim1_filter]
        if df.empty:
            return df
    df = df.drop_duplicates(subset=["SpatialDim", "TimeDim"]) if {"SpatialDim", "TimeDim"} <= set(df.columns) else df
    df_norm = normalize_indicator(df)
    df_norm = df_norm[df_norm["TIME_PERIOD"].isin(years)]
    return df_norm

###################################################
# Main-Funktion zum Laden der Daten aus der WHO-API
###################################################

def load_who_data(timeout: int = 60, overwrite: bool = False) -> pd.DataFrame:

    if not PHARMA_PATH.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {PHARMA_PATH}")
    
    countries, country_years = load_country_years()

    base = (
        pd.DataFrame(
            [(c, y) for c in countries for y in country_years.get(c, [])],
            columns=["REF_AREA", "TIME_PERIOD"],
        )
        .drop_duplicates()
        .sort_values(by=["REF_AREA", "TIME_PERIOD"])
        .reset_index(drop=True)
    )

    df_health_care_who = base.copy()

    for item in INDICATORS:
        code = item["code"]
        col = item["col"]
        dim1_filter = item.get("dim1_filter") if code == "GLASSAMC_AWARE" else None
        print(f'Lade Daten für Indikator-Code {code}')
        ind_frames = []
        for country_code in countries:
            years = country_years.get(country_code) or []
            df_norm = fetch_indicator_for_country(code, country_code, years, dim1_filter=dim1_filter, timeout=timeout)
            if df_norm.empty:
                continue
            df_norm = df_norm.rename(columns={"numericValue": col})
            ind_frames.append(df_norm[["REF_AREA", "TIME_PERIOD", col]])

        if not ind_frames:
            print(f"[WARN] Keine Daten für Indikator {code}.")
            continue

        ind_df = (
            pd.concat(ind_frames, ignore_index=True)
            .drop_duplicates(subset=["REF_AREA", "TIME_PERIOD"])
        )
        df_health_care_who = df_health_care_who.merge(ind_df, on=["REF_AREA", "TIME_PERIOD"], how="left")
        print(f"    -> gemerged: {len(ind_df)} Zeilen")

    if df_health_care_who.empty:
        print("[WARN] Keine WHO-Indikator-Daten gesammelt.")
        return df_health_care_who

    df_health_care_who = df_health_care_who.sort_values(by=["REF_AREA", "TIME_PERIOD"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_health_care_who.to_csv(OUTPUT_PATH, index=False)

    indicator_cols = [item["col"] for item in INDICATORS if item.get("col") and item["col"] in df_health_care_who.columns]
    total_entries = int(df_health_care_who[indicator_cols].notna().sum().sum()) if indicator_cols else 0
    print(f"CSV exportiert nach {OUTPUT_PATH}")

    return df_health_care_who
