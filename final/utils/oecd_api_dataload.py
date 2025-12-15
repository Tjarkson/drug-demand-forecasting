from __future__ import annotations
from typing import Any, Dict, Iterator, List, Sequence

import pandas as pd
import requests
import time

##################################
# Konstanten und globale Variablen
##################################

BASE_URL = "https://sdmx.oecd.org/public/rest"
DATAFLOW_ID = "HEALTH_PHMC@DF_PHMC_CONSUM"
DSD_AGENCY_ID = "OECD.ELS.HD"
DSD_ID = "DSD_HEALTH_PHMC"
DSD_VERSION = "1.1"
MEASURE_CODE = "PH_CON"
UNIT_MEASURE_CODE = "DDD_10P3HB"  # OECD label for DDD per 1 000 inhabitants per day #deutsch
MARKET_TYPE_CODE = "_Z"  # Only “Not applicable” is disseminated for this dataset
DEFAULT_PHARMACEUTICAL_CODES: List[str] = []  # wird aus der Codelist befüllt
DEFAULT_LAST_N_OBSERVATIONS: int | None = 5000  # holt vollständige Historie pro Serie

DIMENSION_AT_OBSERVATION = "TIME_PERIOD"
REQUEST_HEADERS = {
    "Accept": "application/vnd.sdmx.data+json;version=1.0",
    "User-Agent": "drug-demand-forecasting/0.1 (+codex-cli)", #entfernen im Nachhinein!
}
MAX_CODES_PER_CHUNK = 1  # ein ATC-Code pro Request, um Kürzungen zu vermeiden
MAX_RETRIES = 6
BACKOFF_INITIAL_SECONDS = 1.0
REQUEST_COOLDOWN_SECONDS = 12.0  # Pause nach jedem Request, um Rate Limits zu vermeiden
MAX_BACKOFF_SECONDS = 90.0
RATE_LIMIT_RESET_SECONDS = 120.0  # längere Cooldown-Phase nach wiederholten 429er
INTER_CHUNK_SLEEP_SECONDS = 20.0  # zusätzliche Pause zwischen Codes
MIN_RETRY_AFTER_SECONDS = 60.0  # falls die API kein Retry-After schickt

_SESSION = requests.Session()

#######################################################
# Hilfsfunktionen für OECD SDMX API Zugriff und Parsing
#######################################################

def _chunked(seq: Sequence[str], chunk_size: int) -> Iterator[List[str]]:
    # Zerlegt eine Sequenz in Chunks fester Größe
    for idx in range(0, len(seq), max(1, chunk_size)):
        yield list(seq[idx : idx + chunk_size])


def _extract_label(value_meta: Dict[str, Any]) -> str | None:
    # Holt eine menschenlesbare Beschriftung aus einem SDMX-Werteobjekt
    label = value_meta.get("name")
    if isinstance(label, str) and label.strip():
        return label
    names = value_meta.get("names") or {}
    for key in ("en", "en-US"):
        text = names.get(key)
        if isinstance(text, str) and text.strip():
            return text
    for text in names.values():
        if isinstance(text, str) and text.strip():
            return text
    return None


def _resolve_attr_values(
    indexes: Sequence[Any] | None, attr_defs: Sequence[Dict[str, Any]] | None
) -> Dict[str, Any]:
    # Mapped Attribut-Indizes auf IDs/Labels gemäß SDMX-Metadaten
    if not indexes or not attr_defs:
        return {}
    resolved: Dict[str, Any] = {}
    for attr_def, raw_value in zip(attr_defs, indexes):
        if raw_value is None:
            continue
        attr_id = attr_def["id"]
        possible_values = attr_def.get("values") or []
        if isinstance(raw_value, int) and possible_values and 0 <= raw_value < len(possible_values):
            value_meta = possible_values[raw_value]
            resolved[attr_id] = value_meta["id"]
            label = _extract_label(value_meta)
            if label:
                resolved[f"{attr_id}_LABEL"] = label
        else:
            resolved[attr_id] = raw_value
    return resolved


def _fetch_pharmaceutical_codes() -> List[str]:
    # Holt alle ATC-Codes aus der CL_PHARMACEUTICAL Codelist
    url = f"{BASE_URL}/datastructure/{DSD_AGENCY_ID}/{DSD_ID}/{DSD_VERSION}"
    params = {
        "references": "all",
        "detail": "referencepartial",
        "format": "sdmx-json",
    }
    headers = dict(REQUEST_HEADERS)
    headers["Accept"] = "application/vnd.sdmx.structure+json;version=1.0"
    try:
        resp = _SESSION.get(url, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        print(f"Warnung: Codelist konnte nicht geladen werden ({exc}); breche ab.")
        return []

    payload = resp.json() or {}
    data_block = payload.get("data") or payload
    codelists = data_block.get("codelists") or data_block.get("Codelists") or []

    codes: List[str] = []
    for codelist in codelists:
        if codelist.get("id") != "CL_PHARMACEUTICAL":
            continue
        for item in (
            codelist.get("codes")
            or codelist.get("items")
            or codelist.get("Items")
            or []
        ):
            code = item.get("id") or item.get("Id")
            if isinstance(code, str) and code.strip():
                codes.append(code.strip())

    if not codes:
        print("Warnung: Keine Codes in der Codelist gefunden.")
        return []

    # Default-Liste zur Transparenz aktualisieren
    DEFAULT_PHARMACEUTICAL_CODES.clear()
    DEFAULT_PHARMACEUTICAL_CODES.extend(sorted(set(codes)))
    return sorted(set(codes))


def _request_chunk(
    pharma_codes: Sequence[str],
    last_n_observations: int | None,
) -> Dict[str, Any]:
    # Ruft einen ATC-Code-Chunk bei der OECD-SDMX-API ab und handhabt Rate-Limits/Backoff
    params: Dict[str, Any] = {"dimensionAtObservation": DIMENSION_AT_OBSERVATION}
    if last_n_observations is not None:
        params["lastNObservations"] = last_n_observations

    # Dimension order laut API-Struktur: REF_AREA.MEASURE.UNIT_MEASURE.MARKET_TYPE.PHARMACEUTICAL
    pharma_part = "+".join(pharma_codes)
    key = f".{MEASURE_CODE}.{UNIT_MEASURE_CODE}.{MARKET_TYPE_CODE}.{pharma_part}"
    url = f"{BASE_URL}/data/{DATAFLOW_ID}/{key}"

    backoff = BACKOFF_INITIAL_SECONDS
    retries = 0
    while True:
        resp = _SESSION.get(url, params=params, headers=REQUEST_HEADERS, timeout=120)
        if resp.status_code == 429:
            print("Hinweis: OECD-API Rate Limit (429) – warte und versuche erneut.")
            retry_after = resp.headers.get("Retry-After")
            try:
                retry_after_seconds = float(retry_after)
            except (TypeError, ValueError):
                retry_after_seconds = 0.0
            sleep_time = retry_after_seconds if retry_after_seconds > 0 else max(backoff, MIN_RETRY_AFTER_SECONDS)
            time.sleep(sleep_time)
            retries += 1
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            if retries >= MAX_RETRIES:
                # Längere Pause nach wiederholten 429er
                print("Warnung: Zu viele 429-Antworten – längere Wartezeit vor dem nächsten Versuch.")
                time.sleep(RATE_LIMIT_RESET_SECONDS)
                retries = 0
                backoff = BACKOFF_INITIAL_SECONDS
            continue

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            if resp.status_code == 404:
                return {"dataSets": [], "structure": {}}
            if resp.status_code >= 500 and retries < MAX_RETRIES:
                print(f"Warnung: OECD-API Serverfehler ({resp.status_code}) – versuche erneut.")
                retries += 1
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                continue
            raise RuntimeError(
                f"OECD request failed ({resp.status_code}): {resp.text[:200]}"
            ) from exc

        if REQUEST_COOLDOWN_SECONDS:
            time.sleep(REQUEST_COOLDOWN_SECONDS)
        data = resp.json().get("data")
        return data or {"dataSets": [], "structure": {}}


def _parse_sdmx(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Parst die SDMX-JSON-Struktur in eine Liste flacher Dict-Zeilen
    datasets = message.get("dataSets") or []
    if not datasets:
        return []

    structure = message.get("structure") or {}
    series_dims = (structure.get("dimensions") or {}).get("series") or []
    obs_dim_meta = ((structure.get("dimensions") or {}).get("observation") or [None])[0]
    if not obs_dim_meta:
        return []

    obs_dim_id = obs_dim_meta["id"]
    obs_values_meta = obs_dim_meta.get("values") or []
    series_attr_defs = (structure.get("attributes") or {}).get("series") or []
    obs_attr_defs = (structure.get("attributes") or {}).get("observation") or []

    rows: List[Dict[str, Any]] = []
    for series_key, series_payload in datasets[0].get("series", {}).items():
        indexes = [int(token) for token in series_key.split(":")]
        if len(indexes) != len(series_dims):
            continue

        base_row: Dict[str, Any] = {}
        for dim_meta, idx in zip(series_dims, indexes):
            values = dim_meta.get("values") or []
            if idx >= len(values):
                continue
            value_meta = values[idx]
            base_row[dim_meta["id"]] = value_meta["id"]
            label = _extract_label(value_meta)
            if label:
                base_row[f"{dim_meta['id']}_LABEL"] = label

        base_row.update(_resolve_attr_values(series_payload.get("attributes"), series_attr_defs))

        for obs_idx_str, obs_payload in (series_payload.get("observations") or {}).items():
            obs_idx = int(obs_idx_str)
            if obs_idx >= len(obs_values_meta):
                continue
            obs_meta = obs_values_meta[obs_idx]
            row = dict(base_row)
            row[obs_dim_id] = obs_meta["id"]
            row["OBS_VALUE"] = obs_payload[0]
            row.update(_resolve_attr_values(obs_payload[1:], obs_attr_defs))
            rows.append(row)

    return rows

########################################
# Main-Funktion zum Laden der OECD-Daten
########################################

def load_pharmaceutical_consumption(
    pharmaceutical_codes: Sequence[str] | None = None,
    last_n_observations: int | None = DEFAULT_LAST_N_OBSERVATIONS,
) -> pd.DataFrame:
    # Lädt OECD-Verbrauchsdaten (DDD) und liefert sie als DataFrame zurück
    all_codes_from_list: List[str] | None = None
    
    if pharmaceutical_codes is None:
        codes = _fetch_pharmaceutical_codes()
        if codes:
            print("Gefundene ATC-Codes aus CL_PHARMACEUTICAL:")
            print(", ".join(codes))
    else:
        codes = list(pharmaceutical_codes)
        single_letter_prefixes = [c for c in codes if isinstance(c, str) and len(c) == 1]
        if single_letter_prefixes:
            all_codes_from_list = _fetch_pharmaceutical_codes()
            expanded: List[str] = []
            for prefix in single_letter_prefixes:
                expanded.extend([c for c in (all_codes_from_list or []) if c.startswith(prefix)])
            codes = [c for c in codes if not (isinstance(c, str) and len(c) == 1)]
            if expanded:
                codes.extend(sorted(set(expanded)))

    if not codes:
        return pd.DataFrame()

    if codes and all_codes_from_list is not None and not set(codes).issubset(set(all_codes_from_list)):
        print("Warnung: Prefix-Expansion ergab keine gültigen Codes; breche ab.")
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for idx, chunk in enumerate(_chunked(codes, MAX_CODES_PER_CHUNK), start=1):
        print(f"({idx}/{len(codes)}) ATC-Code(s): {', '.join(chunk)}")
        payload = _request_chunk(chunk, last_n_observations)
        rows = _parse_sdmx(payload)
        if rows:
            frames.append(pd.DataFrame(rows))
            print(f"    -> erhalten: {len(rows)} Zeilen")
        else:
            print("    -> keine Daten erhalten")
        if INTER_CHUNK_SLEEP_SECONDS:
            time.sleep(INTER_CHUNK_SLEEP_SECONDS)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    # Roh-Spalten beibehalten; Typen und Sortierung anwenden
    df["TIME_PERIOD"] = pd.to_numeric(df["TIME_PERIOD"], errors="coerce").astype("Int64")
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df.sort_values([ "PHARMACEUTICAL", "REF_AREA", "TIME_PERIOD"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def main() -> None:
    # Führt den vollständigen OECD-Datenabruf und CSV-Export aus
    df = load_pharmaceutical_consumption()
    output_path = "data/raw/pharma_consumption.csv"
    df.to_csv(output_path, index=False)
    print(f"CSV exportiert nach {output_path}")

    country_col = "REF_AREA" if "REF_AREA" in df.columns else None
    pharma_col = "PHARMACEUTICAL" if "PHARMACEUTICAL" in df.columns else None
    rows = len(df)
    pharma_n = df[pharma_col].nunique() if pharma_col else 0
    country_n = df[country_col].nunique() if country_col else 0
    print(
        f"Loaded {rows:,} rows for {pharma_n} ATC class "
        f"and {country_n} countries."
    )
    print(df.head(10))



