"""Saving and loading the trained model bundle and JSON artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_reference_stats(X: pd.DataFrame) -> Dict[str, Any]:
    """Typical values used as a neutral baseline in the app explanations."""
    stats: Dict[str, Any] = {}
    for col in NUMERIC_FEATURES:
        stats[col] = float(pd.to_numeric(X[col], errors="coerce").median())
    for col in CATEGORICAL_FEATURES:
        mode = X[col].astype("string").mode(dropna=True)
        stats[col] = str(mode.iloc[0]) if len(mode) else "Mixed"
    return stats


def save_bundle(bundle: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {path}\nRun `python train.py` first."
        )
    return joblib.load(path)


def save_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=default)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
