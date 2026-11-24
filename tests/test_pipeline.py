"""Test suite for the organic farming suitability pipeline.

Run with:
    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from organic_ml.config import KEY_FEATURES  # noqa: E402
from organic_ml.data import (  # noqa: E402
    parse_target,
    prepare_xy,
    validate_dataset,
)
from organic_ml.features import (  # noqa: E402
    ENGINEERED_NUMERIC,
    build_pipeline,
    engineer_features,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "organic_farming_suitability.csv"
    if not path.exists():
        pytest.skip("Dataset not available.")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def small_xy(dataset):
    X, y = prepare_xy(dataset)
    return train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)


def test_target_parser_accepts_known_labels():
    y = parse_target(pd.Series(["Suitable", "Not Suitable", "suitable", "not_suitable"]))
    assert y.tolist() == [1, 0, 1, 0]


def test_target_parser_rejects_unknown_labels():
    with pytest.raises(ValueError, match="Unknown target"):
        parse_target(pd.Series(["Suitable", "Maybe"]))


def test_validation_requires_columns(dataset):
    broken = dataset.drop(columns=["pH"])
    with pytest.raises(ValueError, match="missing required column"):
        validate_dataset(broken)


def test_validation_flags_score_leakage(dataset):
    checks = validate_dataset(dataset)
    assert checks["score_label_agreement"] > 0.98


def test_feature_engineering_is_complete_and_numeric(dataset):
    X, _ = prepare_xy(dataset)
    engineered = engineer_features(X)
    assert engineered.shape[0] == len(X)
    assert not engineered.isna().any().any()
    for col in ENGINEERED_NUMERIC:
        assert col in engineered.columns
    assert engineered["nutrient_sufficiency_index"].between(0, 5).all()
    assert engineered["chem_score"].between(0, 2).all()


def test_feature_engineering_is_idempotent(dataset):
    X, _ = prepare_xy(dataset)
    once = engineer_features(X)
    twice = engineer_features(once)
    pd.testing.assert_frame_equal(once, twice)


def test_pipeline_end_to_end(small_xy):
    X_train, X_test, y_train, y_test = small_xy
    model = build_pipeline(
        LogisticRegression(max_iter=2000, random_state=42), scale_numeric=True
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    assert probabilities.shape == (len(X_test),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_pipeline_handles_unseen_category(small_xy):
    X_train, X_test, y_train, _ = small_xy
    model = build_pipeline(
        LogisticRegression(max_iter=2000, random_state=42), scale_numeric=True
    )
    model.fit(X_train, y_train)
    row = X_test.iloc[[0]].copy()
    row["Previous_Fertilizer_Usage"] = "Unknown"
    probability = model.predict_proba(row)[0, 1]
    assert 0.0 <= probability <= 1.0


def test_saved_bundle_if_available():
    bundle_path = PROJECT_ROOT / "outputs" / "models" / "best_model.joblib"
    if not bundle_path.exists():
        pytest.skip("Model bundle not trained yet.")
    from organic_ml.io import load_bundle

    bundle = load_bundle(bundle_path)
    assert "model" in bundle
    assert "threshold" in bundle
    assert 0.0 < bundle["threshold"] < 1.0
    assert set(bundle["feature_columns"]) == set(KEY_FEATURES)
