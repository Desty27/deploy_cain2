from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.services import data_loader
from app.services.firebase_client import delete as fb_delete
from app.services.firebase_client import get as fb_get
from app.services.firebase_client import put as fb_put

SUPPORTED_KINDS = {"matches", "deliveries", "players", "wellness", "candidates"}


def _detect_kind(df: pd.DataFrame) -> Optional[str]:
    cols = {str(c).lower() for c in df.columns}
    if {"match_id", "team_a", "team_b"}.issubset(cols):
        return "matches"
    if {"match_id", "innings", "over", "ball"}.issubset(cols):
        return "deliveries"
    wellness_signals = {
        "readiness",
        "readiness_score",
        "risk_score",
        "wellness_score",
        "acute_load",
        "chronic_load",
        "acute_chronic_ratio",
        "soreness",
        "sleep_hours",
        "recovery_index",
    }
    if {"player_id", "team"}.issubset(cols) and cols.intersection(wellness_signals):
        return "wellness"
    if {"player_id"}.issubset(cols) and ("full_name" in cols or "name" in cols) and "team" in cols:
        return "players"
    if {"final_score", "player_id"}.issubset(cols) or {"role", "league_level"}.issubset(cols):
        return "candidates"
    return None


def _canonical_row(row: Dict[str, Any]) -> str:
    """Stable string key for deduping rows irrespective of ordering."""
    try:
        return json.dumps(row, sort_keys=True, default=str)
    except TypeError:
        return str(row)


def _merge_records(existing: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    merged: List[Dict[str, Any]] = []
    for row in existing:
        key = _canonical_row(row)
        if key not in seen:
            seen.add(key)
            merged.append(row)
    for row in new_rows:
        key = _canonical_row(row)
        if key not in seen:
            seen.add(key)
            merged.append(row)
    return merged


def _normalize(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "deliveries":
        numeric_cols = ["runs_off_bat", "extras", "runs_total", "dismissed_batter_id", "over", "ball", "innings", "match_id"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "runs_total" not in df.columns and {"runs_off_bat", "extras"}.issubset(df.columns):
            df["runs_total"] = df["runs_off_bat"].fillna(0) + df["extras"].fillna(0)
    if kind == "matches" and "match_id" in df.columns:
        df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")
    if kind == "players" and "player_id" in df.columns:
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    if kind == "candidates" and "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(str)
    return df


def _parse_upload(content: bytes, filename: str) -> pd.DataFrame:
    lower = filename.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(io.BytesIO(content))
    raise ValueError("Unsupported file type; upload CSV or Excel")


def save_file(content: bytes, filename: str, kind: Optional[str] = None) -> Tuple[str, int]:
    df = _parse_upload(content, filename)
    detected = kind or _detect_kind(df)
    if detected is None:
        raise ValueError("Could not detect dataset type; specify kind explicitly")
    if detected not in SUPPORTED_KINDS:
        raise ValueError(f"Dataset kind '{detected}' not supported")

    df = _normalize(df, detected)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    records_list = df.where(pd.notnull(df), None).to_dict(orient="records")

    # Append to existing Firebase dataset while ignoring identical rows
    existing_raw = fb_get(f"datasets/{detected}")
    existing_records: List[Dict[str, Any]] = []
    if isinstance(existing_raw, dict):
        existing_records = [val for _, val in sorted(existing_raw.items(), key=lambda kv: kv[0])]
    merged_records = _merge_records(existing_records, records_list)

    # Firebase prefers objects over top-level arrays; key by index
    keyed_records: Dict[str, Any] = {str(i): row for i, row in enumerate(merged_records)}

    ok, status, detail = fb_put(f"datasets/{detected}", keyed_records)
    if not ok:
        raise RuntimeError(f"Failed to persist to Firebase (status {status}): {detail}")

    # Clear cached CSV frames so next request reloads from Firebase
    data_loader.reset_cache()
    return detected, len(merged_records)


def reset(kind: Optional[str] = None) -> Dict[str, int]:
    """Clear one dataset or all datasets in Firebase."""
    targets: List[str]
    if kind is None or kind == "all":
        targets = sorted(SUPPORTED_KINDS)
    elif kind not in SUPPORTED_KINDS:
        raise ValueError(f"Dataset kind '{kind}' not supported")
    else:
        targets = [kind]

    for target in targets:
        ok, status, detail = fb_delete(f"datasets/{target}")
        if not ok and status not in (200, 204, 404):
            raise RuntimeError(f"Failed to reset '{target}' (status {status}): {detail}")

    data_loader.reset_cache()
    return {target: 0 for target in targets}
