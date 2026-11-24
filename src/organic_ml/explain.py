"""Model explainability and agronomic recommendations.

Two complementary views are produced:

1. **Global** - permutation importance (model-agnostic) and logistic
   coefficients (when the chosen model is linear).
2. **Local** - an occlusion explanation for a single field: how much does
   each input push the predicted probability up or down compared with a
   typical field?
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import FEATURE_LABELS, KEY_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def permutation_importance_table(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 20,
) -> pd.DataFrame:
    """Permutation importance on the *original* input columns."""
    result = permutation_importance(
        estimator,
        X,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return table.reset_index(drop=True)


def _as_frame(row: Any) -> pd.DataFrame:
    if isinstance(row, pd.DataFrame):
        return row.copy()
    if isinstance(row, pd.Series):
        return row.to_frame().T
    if isinstance(row, dict):
        return pd.DataFrame([row])
    raise TypeError("row must be a dict, pandas Series or one-row DataFrame")


def local_explanations(
    estimator: Any,
    row: Any,
    reference: Dict[str, Any],
    top_n: int = 6,
) -> pd.DataFrame:
    """Occlusion-based explanation for one field.

    For every feature we replace its value with a typical value from the
    training set and measure how much the probability of "Suitable" changes.

    contribution > 0  -> this feature is *helping* the field
    contribution < 0  -> this feature is *hurting* the field
    """
    frame = _as_frame(row)
    base_prob = float(estimator.predict_proba(frame)[0, 1])

    rows: List[Dict[str, Any]] = []
    for feature in KEY_FEATURES:
        if feature not in frame.columns:
            continue
        neutral = frame.copy()
        neutral[feature] = reference[feature]  # pandas broadcasts the scalar
        neutral_prob = float(estimator.predict_proba(neutral)[0, 1])
        contribution = base_prob - neutral_prob
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "value": frame.iloc[0][feature],
                "reference": reference[feature],
                "contribution": contribution,
            }
        )

    table = pd.DataFrame(rows)
    table["abs_contribution"] = table["contribution"].abs()
    table = table.sort_values("abs_contribution", ascending=False).head(top_n)
    return table.reset_index(drop=True)


def confidence_label(probability: float) -> str:
    """Human-friendly confidence band."""
    distance = abs(probability - 0.5) * 2
    if distance >= 0.65:
        return "High confidence"
    if distance >= 0.35:
        return "Moderate confidence"
    return "Borderline / uncertain"


def recommendations(
    row: Any,
    probability: float,
    top_n: int = 5,
) -> List[Dict[str, str]]:
    """Rule-based, agronomically sensible suggestions for one field.

    These rules are intentionally simple and transparent.  They are
    educational advice, not a substitute for a soil-test lab report.
    """
    frame = _as_frame(row)
    r = frame.iloc[0]
    items: List[Dict[str, str]] = []

    def add(priority: str, text: str) -> None:
        items.append({"priority": priority, "text": text})

    ph = float(r.get("pH", 7.0))
    oc = float(r.get("Organic_Carbon_pct", 0.7))
    n = float(r.get("Available_N_kg_ha", 250))
    p = float(r.get("Available_P_kg_ha", 25))
    k = float(r.get("Available_K_kg_ha", 250))
    s = float(r.get("Available_S_mg_kg", 15))
    years = float(r.get("Years_Chemical_Input", 0))
    fert = str(r.get("Previous_Fertilizer_Usage", "Mixed")).lower()
    pest = str(r.get("Previous_Pesticide_Usage", "Moderate")).lower()

    if ph < 6.0:
        add("High", "Soil is acidic. Apply agricultural lime or wood ash and re-test pH before the next season.")
    elif ph > 7.8:
        add("High", "Soil is alkaline. Add compost/gypsum and consider acidifying organic matter; choose alkaline-tolerant crops.")
    elif ph < 6.5:
        add("Medium", "Soil is slightly acidic. A small lime dose and more organic matter will help nutrient availability.")

    if oc < 0.75:
        add("High", "Organic carbon is low. Add 5-10 t/ha of compost or farmyard manure and grow a green-manure crop.")
    elif oc < 1.15:
        add("Medium", "Organic carbon is moderate. Keep adding compost and crop residues to build soil carbon.")

    if n < 280:
        add("Medium", "Available nitrogen is low. Include legumes (pulses, groundnut) in rotation and use organic N sources.")
    if p < 25:
        add("Medium", "Available phosphorus is low. Use rock phosphate, bone meal or compost-enriched manure.")
    if k < 250:
        add("Medium", "Available potassium is low. Apply wood ash, compost or sulphate of potash (organic-approved).")
    if s < 15:
        add("Low", "Sulphur is low. Gypsum or compost will usually correct this over a season.")

    if years >= 5 and fert in {"chemical", "mixed"}:
        add("High", "Long chemical-input history. Plan a 2-3 year conversion period with reduced synthetic inputs.")
    if pest == "high":
        add("Medium", "High past pesticide use. Adopt integrated pest management and encourage natural predators.")

    if probability >= 0.65 and not items:
        add("Low", "Field looks well suited. Maintain organic matter, crop rotation and regular soil testing.")

    if probability < 0.40:
        add("Info", "Current field is a weak candidate. Focus on the high-priority corrections above before conversion.")

    return items[:top_n]
