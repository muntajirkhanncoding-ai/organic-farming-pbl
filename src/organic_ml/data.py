"""Data loading, validation and quality auditing.

The goal of this module is to fail loudly and early when the input file is
wrong, instead of silently training a model on garbage (a common problem in
student projects).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .config import (
    CATEGORICAL_FEATURES,
    ID_COL,
    KEY_FEATURES,
    NUMERIC_FEATURES,
    SCORE_COL,
    TARGET_COL,
)

# Columns that must be present for the pipeline to run.
REQUIRED_COLUMNS = KEY_FEATURES + [TARGET_COL, SCORE_COL]

# Physically plausible ranges used for a soft validation check.
EXPECTED_RANGES = {
    "pH": (3.0, 11.0),
    "Organic_Carbon_pct": (0.0, 6.0),
    "Available_N_kg_ha": (0.0, 1500.0),
    "Available_P_kg_ha": (0.0, 500.0),
    "Available_K_kg_ha": (0.0, 2000.0),
    "Available_S_mg_kg": (0.0, 200.0),
    "Years_Chemical_Input": (0, 60),
}

# Allowed category values (case-insensitive, stripped).
ALLOWED_CATEGORIES = {
    "Previous_Fertilizer_Usage": {"chemical", "mixed", "organic"},
    "Previous_Pesticide_Usage": {"high", "moderate", "low"},
}

TARGET_MAP = {
    "suitable": 1,
    "not suitable": 0,
    "not_suitable": 0,
    "unsuitable": 0,
    "yes": 1,
    "no": 0,
    "true": 1,
    "false": 0,
    "1": 1,
    "0": 0,
}


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------
def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel dataset and normalise column names.

    Parameters
    ----------
    path : str or Path
        Location of the dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset with stripped column names.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Put the CSV inside the data/ folder or pass --data <path>."
        )

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    df.columns = [str(c).strip() for c in df.columns]

    # Strip whitespace from text columns (common Excel export problem).
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()

    return df


# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------
def parse_target(series: pd.Series) -> pd.Series:
    """Convert the target column to 0/1 with strict validation.

    Unlike the original script, unknown labels raise an error instead of
    being silently mapped to a class.  Silent mapping can hide data-entry
    mistakes and inflate accuracy.
    """
    if series.isna().any():
        raise ValueError(
            f"Target column '{series.name}' contains {int(series.isna().sum())} "
            "missing value(s)."
        )

    normalised = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", " ", regex=True)
    )
    # "not suitable" is kept as a two-word phrase because it is meaningful.
    mapped = normalised.map(TARGET_MAP)

    if mapped.isna().any():
        unknown = sorted(normalised[mapped.isna()].unique().tolist())
        raise ValueError(
            "Unknown target label(s): "
            + ", ".join(repr(u) for u in unknown)
            + ". Expected one of: Suitable / Not Suitable."
        )
    return mapped.astype(int).rename("target")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """Run a set of quality checks and return a structured report.

    Raises
    ------
    ValueError
        If required columns are missing or the target is unparseable.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Dataset is missing required column(s): " + ", ".join(missing_cols)
        )

    y = parse_target(df[TARGET_COL])
    checks: Dict[str, object] = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "class_balance_pct": {
            "not_suitable": round(float((y == 0).mean() * 100), 2),
            "suitable": round(float((y == 1).mean() * 100), 2),
        },
        "warnings": [],
    }

    warnings: List[str] = checks["warnings"]  # type: ignore[assignment]

    # Category values.
    for col, allowed in ALLOWED_CATEGORIES.items():
        if col in df.columns:
            values = set(df[col].astype("string").str.lower().unique())
            unexpected = values - allowed - {pd.NA}
            if unexpected:
                warnings.append(f"{col} has unexpected value(s): {sorted(unexpected)}")

    # Numeric ranges.
    for col, (low, high) in EXPECTED_RANGES.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            bad = int(((s < low) | (s > high)).sum())
            if bad:
                warnings.append(f"{col} has {bad} value(s) outside [{low}, {high}]")

    # Label leakage check: the original dataset derives the label from the
    # suitability score.  We never use that score as a feature.
    if SCORE_COL in df.columns:
        score = pd.to_numeric(df[SCORE_COL], errors="coerce")
        inferred = (score >= 55).astype(int)
        agreement = float((inferred == y).mean())
        checks["score_label_agreement"] = round(agreement, 4)
        if agreement > 0.98:
            warnings.append(
                f"{SCORE_COL} reproduces the label with {agreement:.1%} agreement. "
                "It is excluded from the features to prevent leakage."
            )

    # Very strong single-feature correlations with the target.
    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    high_corr = []
    for col in numeric_cols:
        if col in {SCORE_COL, ID_COL}:
            continue
        corr = df[col].corr(y)
        if pd.notna(corr) and abs(corr) > 0.90:
            high_corr.append(f"{col} (r={corr:.2f})")
    if high_corr:
        warnings.append("Suspiciously high target correlation: " + ", ".join(high_corr))

    return checks


# --------------------------------------------------------------------------
# Dataset audit table
# --------------------------------------------------------------------------
def audit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column quality table for the report."""
    rows = []
    for col in df.columns:
        s = df[col]
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
        }
        if is_numeric_dtype(s):
            row.update(
                {
                    "min": round(float(s.min()), 3),
                    "median": round(float(s.median()), 3),
                    "max": round(float(s.max()), 3),
                    "mean": round(float(s.mean()), 3),
                }
            )
        else:
            top = s.astype("string").value_counts(dropna=True)
            row["most_common"] = str(top.index[0]) if len(top) else ""
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Feature matrix
# --------------------------------------------------------------------------
def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return the raw feature matrix and the binary target.

    The ID and the Suitability_Score are deliberately dropped:
      * ID is a row identifier with no predictive meaning.
      * Score is the rubric from which the label was generated (leakage).
    """
    X = df[KEY_FEATURES].copy()
    y = parse_target(df[TARGET_COL])

    # Coerce numeric columns and report bad values instead of ignoring them.
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isna().any():
            raise ValueError(f"Column '{col}' contains non-numeric values.")
    return X, y


def dataset_summary(df: pd.DataFrame) -> str:
    """Short human-readable summary printed by train.py."""
    y = parse_target(df[TARGET_COL])
    lines = [
        f"Rows: {len(df):,} | Columns: {df.shape[1]}",
        f"Target: {TARGET_COL}",
        f"  Not Suitable: {(y == 0).sum():,} ({(y == 0).mean():.1%})",
        f"  Suitable    : {(y == 1).sum():,} ({(y == 1).mean():.1%})",
        f"Missing cells: {int(df.isna().sum().sum())}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
    ]
    return "\n".join(lines)


def numeric_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for the original numeric soil features."""
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    stats = df[cols].describe().T
    stats["skew"] = df[cols].skew()
    return stats.round(3)


def remove_exact_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows and return the number removed."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)
