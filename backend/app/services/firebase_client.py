from __future__ import annotations

import json
import math
import os
from typing import Any, Optional, Tuple

import requests

# Hardcoded Realtime Database base URL (free tier)
FIREBASE_BASE = "https://cain-a08c1-default-rtdb.asia-southeast1.firebasedatabase.app"
# Auth is optional; public DBs don't need it. Set to True and provide a token to enable.
FIREBASE_USE_AUTH = False
# If FIREBASE_USE_AUTH is True, this token is used; leave empty to fall back to env vars.
FIREBASE_AUTH_TOKEN = ""


def _make_url(path: str) -> str:
    cleaned = path.strip("/")
    base = f"{FIREBASE_BASE}/{cleaned}.json"
    if FIREBASE_USE_AUTH:
        auth = FIREBASE_AUTH_TOKEN or os.getenv("FIREBASE_DATABASE_SECRET") or os.getenv("FIREBASE_AUTH")
        if auth:
            return f"{base}?auth={auth}"
    return base


def _sanitize_for_json(value: Any) -> Any:
    """Recursively replace NaN/inf with None so json.dumps rejects nothing."""
    if value is None:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def get(path: str, timeout: int = 10) -> Optional[Any]:
    try:
        res = requests.get(_make_url(path), timeout=timeout)
        if res.status_code == 200:
            return res.json()
        return None
    except Exception:
        return None


def put(path: str, data: Any, timeout: int = 10) -> Tuple[bool, int, str]:
    """Write data to Firebase; returns (ok, status, detail)."""
    try:
        payload = json.dumps(_sanitize_for_json(data), allow_nan=False)
        res = requests.put(
            _make_url(path),
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        return (200 <= res.status_code < 300, res.status_code, res.text)
    except ValueError as exc:  # JSON encoding issues (NaN/inf)
        return False, 0, f"JSON encoding error: {exc}"
    except Exception as exc:  # pragma: no cover - network failures
        return False, 0, str(exc)


def delete(path: str, timeout: int = 10) -> Tuple[bool, int, str]:
    """Delete data at path; returns (ok, status, detail)."""
    try:
        res = requests.delete(_make_url(path), timeout=timeout)
        return (200 <= res.status_code < 300, res.status_code, res.text)
    except Exception as exc:  # pragma: no cover - network failures
        return False, 0, str(exc)
