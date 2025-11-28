from __future__ import annotations

"""
Load pharmaceutical consumption time series from the OECD SDMX REST API.

The API exposes DDD per 1 000 inhabitants per day under the code DDD_10P3HB in
the CL_UNIT_MEASURE codelist. This is the successor of the legacy DDD_TP_POP
label mentioned in the requirements.
"""

import time
from typing import Any, Dict, Iterator, List, Sequence

import pandas as pd
import requests

BASE_URL = "https://sdmx.oecd.org/public/rest"
DATAFLOW_ID = "HEALTH_PHMC@DF_PHMC_CONSUM"
MEASURE_CODE = "PH_CON"
UNIT_MEASURE_CODE = "DDD_10P3HB"  # OECD label for DDD per 1 000 inhabitants per day
MARKET_TYPE_CODE = "_Z"  # Only “Not applicable” is disseminated for this dataset
DEFAULT_PHARMACEUTICAL_CODES = [
    "J01",
]

DIMENSION_AT_OBSERVATION = "TIME_PERIOD"
REQUEST_HEADERS = {
    "Accept": "application/vnd.sdmx.data+json;version=1.0",
    "User-Agent": "drug-demand-forecasting/0.1 (+codex-cli)", #entfernen im Nachhinein!
}
MAX_CODES_PER_CHUNK = 4
MAX_RETRIES = 6
BACKOFF_INITIAL_SECONDS = 1.0
REQUEST_COOLDOWN_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0
RATE_LIMIT_RESET_SECONDS = 15.0

_SESSION = requests.Session()


def load_pharmaceutical_consumption(
    pharmaceutical_codes: Sequence[str] | None = None,
    start_period: str | None = None,
    end_period: str | None = None,
) -> pd.DataFrame:
    """Return a tidy DataFrame with ATC consumption per country/year."""
    codes = list(pharmaceutical_codes or DEFAULT_PHARMACEUTICAL_CODES)
    if not codes:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for chunk in _chunked(codes, MAX_CODES_PER_CHUNK):
        payload = _request_chunk(chunk, start_period, end_period)
        rows = _parse_sdmx(payload)
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    rename_map = {
        "REF_AREA": "country_code",
        "PHARMACEUTICAL": "pharmaceutical_code",
        "TIME_PERIOD": "year",
        "UNIT_MEASURE": "unit_measure_code",
        "UNIT_MEASURE_LABEL": "unit_measure_label",
        "MEASURE": "measure_code",
        "MARKET_TYPE": "market_type_code",
        "MARKET_TYPE_LABEL": "market_type_label",
        "OBS_STATUS": "observation_status",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "consumption_ddd_per_1000_inhabitants_per_day" in df.columns:
        df["consumption_ddd_per_1000_inhabitants_per_day"] = pd.to_numeric(
            df["consumption_ddd_per_1000_inhabitants_per_day"], errors="coerce"
        )
    sort_cols = [col for col in ("country_code", "pharmaceutical_code", "year") if col in df.columns]
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def _request_chunk(
    pharma_codes: Sequence[str], start_period: str | None, end_period: str | None
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"dimensionAtObservation": DIMENSION_AT_OBSERVATION}
    if start_period:
        params["startPeriod"] = start_period
    if end_period:
        params["endPeriod"] = end_period

    key = f".{MEASURE_CODE}.{UNIT_MEASURE_CODE}.{MARKET_TYPE_CODE}." + "+".join(pharma_codes)
    url = f"{BASE_URL}/data/{DATAFLOW_ID}/{key}"

    backoff = BACKOFF_INITIAL_SECONDS
    retries = 0
    while True:
        resp = _SESSION.get(url, params=params, headers=REQUEST_HEADERS, timeout=120)
        if resp.status_code == 429:
            retries += 1
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            if retries >= MAX_RETRIES:
                # Cool down once the short exponential backoff has been exhausted.
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


def _resolve_attr_values(
    indexes: Sequence[Any] | None, attr_defs: Sequence[Dict[str, Any]] | None
) -> Dict[str, Any]:
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


def _extract_label(value_meta: Dict[str, Any]) -> str | None:
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


def _chunked(seq: Sequence[str], chunk_size: int) -> Iterator[List[str]]:
    for idx in range(0, len(seq), max(1, chunk_size)):
        yield list(seq[idx : idx + chunk_size])


def main() -> None:
    df = load_pharmaceutical_consumption()
    output_path = "data/raw/pharma_consumption.csv"
    df.to_csv(output_path, index=False)
    print(f"CSV exportiert nach {output_path}")

    print(
        f"Loaded {len(df):,} rows for {df['pharmaceutical_code'].nunique()} ATC class "
        f"and {df['country_code'].nunique()} countries."
    )
    print(df.head(10))


if __name__ == "__main__":
    main()
