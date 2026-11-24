"""Batch prediction for new fields.

Usage
-----
    python predict.py --input data/sample_fields.csv
    python predict.py --input my_fields.csv --output my_results.csv
    python predict.py --input my_fields.csv --threshold 0.60

The input file must contain the nine raw columns:
    pH, Organic_Carbon_pct, Available_N_kg_ha, Available_P_kg_ha,
    Available_K_kg_ha, Available_S_mg_kg, Previous_Fertilizer_Usage,
    Previous_Pesticide_Usage, Years_Chemical_Input
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from organic_ml.config import BUNDLE_PATH, KEY_FEATURES  # noqa: E402
from organic_ml.explain import confidence_label, local_explanations  # noqa: E402
from organic_ml.io import load_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict organic-farming suitability.")
    parser.add_argument("--input", type=Path, required=True, help="CSV/Excel file with fields.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save predictions (default: <input>_predictions.csv).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=BUNDLE_PATH,
        help="Path to the trained model bundle.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the tuned decision threshold.",
    )
    parser.add_argument(
        "--no-explanations",
        action="store_true",
        help="Skip the per-row factor explanations (faster).",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.model)
    model = bundle["model"]
    threshold = args.threshold if args.threshold is not None else bundle["threshold"]
    reference = bundle["reference_stats"]

    df = read_table(args.input)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in KEY_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit("Missing required column(s): " + ", ".join(missing))

    X = df[KEY_FEATURES].copy()
    for col in KEY_FEATURES:
        if col not in {"Previous_Fertilizer_Usage", "Previous_Pesticide_Usage"}:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            if X[col].isna().any():
                raise SystemExit(f"Column '{col}' contains non-numeric values.")

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    result = df.copy()
    result["probability_suitable"] = probabilities.round(4)
    result["predicted_label"] = [
        "Suitable" if p == 1 else "Not Suitable" for p in predictions
    ]
    result["confidence"] = [confidence_label(float(p)) for p in probabilities]
    result["threshold_used"] = round(float(threshold), 3)

    if not args.no_explanations:
        positives, negatives = [], []
        for _, row in X.iterrows():
            table = local_explanations(model, row, reference, top_n=6)
            pos = table[table["contribution"] > 0].head(1)
            neg = table[table["contribution"] < 0].head(1)
            positives.append(pos["label"].iloc[0] if len(pos) else "-")
            negatives.append(neg["label"].iloc[0] if len(neg) else "-")
        result["top_helping_factor"] = positives
        result["top_hurting_factor"] = negatives

    output = args.output or args.input.with_name(args.input.stem + "_predictions.csv")
    result.to_csv(output, index=False)

    print(f"Model      : {bundle['model_name']}")
    print(f"Threshold  : {threshold:.3f}")
    print(f"Rows scored: {len(result):,}")
    print("\nPrediction summary:")
    print(result["predicted_label"].value_counts().to_string())
    print("\nPreview:")
    columns = [
        "probability_suitable",
        "predicted_label",
        "confidence",
        "top_helping_factor",
        "top_hurting_factor",
    ]
    print(result[[c for c in columns if c in result.columns]].head(10).to_string(index=False))
    print(f"\nSaved -> {output}")


if __name__ == "__main__":
    main()
