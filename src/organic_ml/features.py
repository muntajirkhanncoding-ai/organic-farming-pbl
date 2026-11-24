"""Feature engineering and preprocessing.

The dataset's label is produced by a *banded* agronomic rubric: pH has an
optimal range, nutrients have sufficiency thresholds and past chemical use
matters.  A plain linear model cannot represent those thresholds, so we add
explicit indicator features.  This is the single biggest reason v2 beats v1.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)

# Agronomic thresholds used to build the indicator features.
PH_OPTIMAL_LOW = 6.0
PH_OPTIMAL_HIGH = 7.8
PH_IDEAL = 6.9
PH_TOO_HIGH = 8.2

OC_SUFFICIENT = 0.75
OC_HIGH = 1.15

N_SUFFICIENT = 280.0
N_HIGH = 375.0
P_SUFFICIENT = 25.0
K_SUFFICIENT = 250.0
K_HIGH = 435.0
S_SUFFICIENT = 15.0

FERTILIZER_SCORE = {"chemical": 0.0, "mixed": 1.0, "organic": 2.0}
PESTICIDE_SCORE = {"high": 0.0, "moderate": 1.0, "low": 2.0}


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered feature set from raw input columns.

    This function is row-wise (it never looks at other rows or at the
    target), so applying it before a train/test split cannot leak
    information.
    """
    X = X.copy()

    # If engineered columns already exist (e.g. re-scoring a saved file),
    # remove them first so the function is idempotent.
    already = [c for c in ENGINEERED_NUMERIC if c in X.columns]
    if already:
        X = X.drop(columns=already)

    fert = X["Previous_Fertilizer_Usage"].astype("string").str.strip().str.lower()
    pest = X["Previous_Pesticide_Usage"].astype("string").str.strip().str.lower()

    X["chem_score"] = fert.map(FERTILIZER_SCORE).astype(float).fillna(1.0)
    X["pest_score"] = pest.map(PESTICIDE_SCORE).astype(float).fillna(1.0)

    # pH: optimum band + distance + penalties.
    X["ph_in_optimal"] = ((X["pH"] >= PH_OPTIMAL_LOW) & (X["pH"] <= PH_OPTIMAL_HIGH)).astype(int)
    X["ph_distance"] = (X["pH"] - PH_IDEAL).abs()
    X["ph_low"] = (X["pH"] < PH_OPTIMAL_LOW).astype(int)
    X["ph_high"] = (X["pH"] > PH_OPTIMAL_HIGH).astype(int)
    X["ph_too_high"] = (X["pH"] > PH_TOO_HIGH).astype(int)

    # Organic carbon / macro-nutrient sufficiency flags.
    X["oc_sufficient"] = (X["Organic_Carbon_pct"] >= OC_SUFFICIENT).astype(int)
    X["oc_high"] = (X["Organic_Carbon_pct"] >= OC_HIGH).astype(int)
    X["n_sufficient"] = (X["Available_N_kg_ha"] >= N_SUFFICIENT).astype(int)
    X["n_high"] = (X["Available_N_kg_ha"] >= N_HIGH).astype(int)
    X["p_sufficient"] = (X["Available_P_kg_ha"] >= P_SUFFICIENT).astype(int)
    X["k_sufficient"] = (X["Available_K_kg_ha"] >= K_SUFFICIENT).astype(int)
    X["k_high"] = (X["Available_K_kg_ha"] >= K_HIGH).astype(int)
    X["s_sufficient"] = (X["Available_S_mg_kg"] >= S_SUFFICIENT).astype(int)
    X["nutrient_sufficiency_index"] = X[
        ["oc_sufficient", "n_sufficient", "p_sufficient", "k_sufficient", "s_sufficient"]
    ].sum(axis=1)

    # Chemical history: high burden = many years x chemical fertilizer x
    # high pesticide use.  Lower burden is better for organic transition.
    X["chemical_burden"] = (
        X["Years_Chemical_Input"] * (3 - X["chem_score"]) * (3 - X["pest_score"])
    )
    X["stewardship"] = X["chem_score"] + X["pest_score"] - X["Years_Chemical_Input"] / 10.0
    X["ph_x_chemical"] = X["ph_in_optimal"] * (3 - X["chem_score"]) * (3 - X["pest_score"])
    X["organic_history"] = (X["chem_score"] == 2).astype(int)

    # Guarantee a stable column order for the preprocessor.
    return X[FEATURE_COLUMNS]


class OrganicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible wrapper around :func:`engineer_features`."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return engineer_features(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(FEATURE_COLUMNS, dtype=object)


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Column transformer for imputation, scaling and one-hot encoding.

    Everything happens *inside* the pipeline, so the test set is never used
    to fit the imputer or the scaler.
    """
    numeric_cols = NUMERIC_FEATURES + ENGINEERED_NUMERIC

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, scale_numeric: bool = False) -> Pipeline:
    """Assemble the full feature-engineering + preprocessing + model pipeline."""
    return Pipeline(
        [
            ("features", OrganicFeatureEngineer()),
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def clean_feature_names(names: List[str]) -> List[str]:
    """Turn 'num__pH' / 'cat__Previous_Fertilizer_Usage_Organic' into labels."""
    cleaned = []
    for name in names:
        label = name.split("__", 1)[-1]
        label = label.replace("_", " ").strip()
        cleaned.append(label)
    return cleaned
