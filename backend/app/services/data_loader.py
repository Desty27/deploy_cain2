from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import sys

import pandas as pd

from app.core.config import get_settings
from app.services.firebase_client import get as fb_get

# Ensure repository root (contains src/) is on sys.path so imports work when running inside deploy_cain/backend
repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from src.analyzer import load_deliveries_csv, load_matches_csv, load_players_csv


class DatasetMissingError(FileNotFoundError):
    """Raised when expected CSV assets are missing."""


def _prepare_frames(deliveries: pd.DataFrame, matches: pd.DataFrame, players: Optional[pd.DataFrame | Dict[str, Any]]):
    if not deliveries.empty:
        deliveries["runs_off_bat"] = deliveries.get("runs_off_bat", 0).fillna(0)
        deliveries["extras"] = deliveries.get("extras", 0).fillna(0)
        deliveries["runs_total"] = deliveries["runs_off_bat"] + deliveries["extras"]
        deliveries["wicket"] = deliveries.get("dismissed_batter_id", pd.Series([])).notna()
    if not matches.empty and "match_id" in matches.columns:
        matches["match_id"] = matches["match_id"].apply(lambda v: int(v) if pd.notna(v) else None)

    players_map: Dict[int, str] = {}
    if players is not None:
        if isinstance(players, pd.DataFrame):
            for _, row in players.iterrows():
                pid = row.get("player_id")
                name = row.get("name") or row.get("full_name")
                try:
                    players_map[int(pid)] = str(name)
                except Exception:
                    continue
        elif isinstance(players, dict):
            for pid, name in players.items():
                try:
                    players_map[int(pid)] = str(name)
                except Exception:
                    continue
    return deliveries, matches, players_map


def _load_from_firebase() -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]]:
    deliveries_json = fb_get("datasets/deliveries")
    matches_json = fb_get("datasets/matches")
    players_json = fb_get("datasets/players")

    if deliveries_json is not None and matches_json is not None:
        deliveries_df = pd.DataFrame.from_dict(deliveries_json, orient="index") if isinstance(deliveries_json, dict) else pd.DataFrame(deliveries_json)
        matches_df = pd.DataFrame.from_dict(matches_json, orient="index") if isinstance(matches_json, dict) else pd.DataFrame(matches_json)
        players_df = None
        if players_json:
            players_df = pd.DataFrame.from_dict(players_json, orient="index") if isinstance(players_json, dict) else pd.DataFrame(players_json)
        return _prepare_frames(deliveries_df, matches_df, players_df)
    return None


@lru_cache(maxsize=1)
def _load_raw_frames() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    # First try Firebase (preferred)
    fb_frames = _load_from_firebase()
    if fb_frames:
        return fb_frames

    settings = get_settings()
    base = Path(settings.data_dir)
    deliveries_path = base / "deliveries.csv"
    matches_path = base / "matches.csv"
    players_path = base / "players.csv"

    if not deliveries_path.exists() or not matches_path.exists():
        raise DatasetMissingError("deliveries.csv or matches.csv not found in data_dir")

    deliveries = pd.DataFrame(load_deliveries_csv(deliveries_path))
    matches = pd.DataFrame(load_matches_csv(matches_path))
    players = load_players_csv(players_path)

    return _prepare_frames(deliveries, matches, players)


def load_frames() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    return _load_raw_frames()


def reset_cache() -> None:
    _load_raw_frames.cache_clear()
