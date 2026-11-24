"""Train, evaluate and explain the organic-farming suitability models.

Usage
-----
    python train.py
    python train.py --fast
    python train.py --data data/organic_farming_suitability.csv --n-iter 30
    python train.py --no-optional          # skip XGBoost / LightGBM

Everything is written to the ``outputs/`` folder:

    outputs/figures/   PNG charts used in the report
    outputs/tables/    CSV result tables
    outputs/models/    best_model.joblib (model + threshold + metadata)
    outputs/results_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Make "src" importable when the script is run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from organic_ml.config import (  # noqa: E402
    BUNDLE_PATH,
    CV_FOLDS,
    CV_REPEATS,
    DEFAULT_DATA_PATH,
    FIGURE_DIR,
    KEY_FEATURES,
    MODEL_DIR,
    N_BOOTSTRAP,
    RANDOM_STATE,
    RESULTS_JSON,
    TABLE_DIR,
    TEST_SIZE,
)
from organic_ml.data import (  # noqa: E402
    audit_data,
    dataset_summary,
    load_dataset,
    numeric_feature_stats,
    prepare_xy,
    remove_exact_duplicates,
    validate_dataset,
)
from organic_ml.explain import (  # noqa: E402
    local_explanations,
    permutation_importance_table,
)
from organic_ml.features import build_pipeline  # noqa: E402
from organic_ml.io import build_reference_stats, save_bundle, save_json  # noqa: E402
from organic_ml.modeling import (  # noqa: E402
    bootstrap_ci,
    compare_models,
    compute_metrics,
    evaluate_holdout,
    get_final_estimator,
    model_coefficients,
    out_of_fold_probabilities,
    select_best_model,
    tune_all_models,
    tune_threshold,
)
from organic_ml.plots import (  # noqa: E402
    plot_calibration,
    plot_class_balance,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_features_by_class,
    plot_learning_curve,
    plot_logistic_coefficients,
    plot_model_comparison,
    plot_partial_dependence,
    plot_permutation_importance,
    plot_roc_curve,
    plot_suitability_rates,
    plot_threshold_analysis,
)
from sklearn.model_selection import train_test_split  # noqa: E402


def log(message: str = "") -> None:
    print(message, flush=True)


def banner(title: str) -> None:
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the organic farming suitability model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the CSV/Excel dataset.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=25,
        help="Randomised-search iterations per model (default: 25).",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=CV_FOLDS,
        help="Cross-validation folds during tuning (default: 5).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=CV_REPEATS,
        help="Repeats for the final CV comparison (default: 3).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick run: fewer tuning iterations and no optional boosters.",
    )
    parser.add_argument(
        "--no-optional",
        action="store_true",
        help="Do not use XGBoost / LightGBM even if installed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    if args.fast:
        args.n_iter = 8
        args.cv = 3
        args.repeats = 2
        args.no_optional = True

    for folder in (FIGURE_DIR, TABLE_DIR, MODEL_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    banner("1. LOADING AND VALIDATING DATA")
    df = load_dataset(args.data)
    log(dataset_summary(df))

    checks = validate_dataset(df)
    log(f"\nValidation checks: {json.dumps(checks, indent=2)}")

    df, removed = remove_exact_duplicates(df)
    if removed:
        log(f"\nRemoved {removed} exact duplicate row(s).")

    audit = audit_data(df)
    audit.to_csv(TABLE_DIR / "data_audit.csv", index=False)
    numeric_feature_stats(df).to_csv(TABLE_DIR / "numeric_summary.csv")
    log(f"Saved data audit -> {TABLE_DIR / 'data_audit.csv'}")

    banner("2. EXPLORATORY DATA ANALYSIS")
    X, y = prepare_xy(df)

    figures = {}
    figures["class_balance"] = plot_class_balance(y)
    figures["feature_distributions"] = plot_feature_distributions(df)
    figures["features_by_class"] = plot_features_by_class(df, y)
    figures["correlation"] = plot_correlation_heatmap(df, y)
    figures["suitability_rates"] = plot_suitability_rates(df, y)
    log("Saved 5 EDA figures.")

    banner("3. TRAIN / TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=args.seed,
        stratify=y,
    )
    log(f"Training rows: {len(X_train):,}")
    log(f"Test rows    : {len(X_test):,}")
    log(f"Features     : {len(KEY_FEATURES)} raw inputs -> 29 after engineering")

    banner("4. HYPER-PARAMETER TUNING (training set only)")
    tuned_models = tune_all_models(
        X_train,
        y_train,
        n_iter=args.n_iter,
        cv=args.cv,
        include_optional=not args.no_optional,
        verbose=True,
    )

    tuning_rows = [
        {
            "model": m.name,
            "cv_auc_mean": m.cv_auc_mean,
            "cv_auc_std": m.cv_auc_std,
            "best_params": json.dumps(m.best_params, default=str),
        }
        for m in tuned_models
    ]
    pd.DataFrame(tuning_rows).to_csv(TABLE_DIR / "tuning_results.csv", index=False)

    banner("5. MODEL COMPARISON (repeated cross-validation)")
    comparison = compare_models(
        tuned_models, X_train, y_train, repeats=args.repeats, verbose=True
    )
    comparison_to_save = comparison.copy()
    comparison_to_save["best_params"] = comparison_to_save["best_params"].apply(
        lambda d: json.dumps(d, default=str)
    )
    comparison_to_save.to_csv(TABLE_DIR / "model_comparison.csv", index=False)

    plot_model_comparison(comparison)
    log(f"Saved model comparison -> {TABLE_DIR / 'model_comparison.csv'}")

    banner("6. SELECTING THE BEST MODEL")
    best = select_best_model(tuned_models, comparison)
    log(f"Selected model : {best.name}")
    log(f"CV ROC-AUC     : {best.cv_metrics['roc_auc']:.4f}")
    log(f"CV F1          : {best.cv_metrics['f1']:.4f}")
    log(f"CV Accuracy    : {best.cv_metrics['accuracy']:.4f}")
    log(f"Best params    : {json.dumps(best.best_params, default=str)}")

    banner("7. THRESHOLD TUNING (out-of-fold predictions)")
    oof_probabilities = out_of_fold_probabilities(best.estimator, X_train, y_train)
    best_threshold, threshold_table = tune_threshold(y_train, oof_probabilities)
    threshold_table.to_csv(TABLE_DIR / "threshold_analysis.csv", index=False)
    log(f"Best threshold : {best_threshold:.3f} (default was 0.50)")
    plot_threshold_analysis(threshold_table, best_threshold)

    banner("8. FINAL EVALUATION ON THE HELD-OUT TEST SET")
    final_model = get_final_estimator(best)
    final_model.fit(X_train, y_train)

    metrics_50, prob_50, _ = evaluate_holdout(final_model, X_test, y_test, 0.5)
    metrics_tuned, probabilities, predictions = evaluate_holdout(
        final_model, X_test, y_test, best_threshold
    )

    ci = {}
    for metric in ("accuracy", "f1", "roc_auc"):
        low, high = bootstrap_ci(
            y_test,
            probabilities,
            best_threshold,
            metric,
            n_boot=N_BOOTSTRAP,
        )
        ci[metric] = {"low": low, "high": high}

    metrics_rows = []
    for label, values in [
        ("default threshold (0.50)", metrics_50),
        (f"tuned threshold ({best_threshold:.2f})", metrics_tuned),
    ]:
        row = {"setting": label}
        row.update(values)
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).round(4).to_csv(
        TABLE_DIR / "metrics_summary.csv", index=False
    )

    log("\nTest metrics at the tuned threshold:")
    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
        "brier",
    ):
        value = metrics_tuned[key]
        extra = ""
        if key in ci:
            extra = f"  95% CI [{ci[key]['low']:.4f}, {ci[key]['high']:.4f}]"
        log(f"  {key:<18s} {value:.4f}{extra}")

    y_pred = (probabilities >= best_threshold).astype(int)
    predictions.to_csv(TABLE_DIR / "test_predictions.csv", index=False)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Not Suitable", "Suitable"],
        output_dict=True,
    )
    flat_report = {}
    for label, values in report.items():
        if isinstance(values, dict):
            for metric, value in values.items():
                flat_report[f"{label}_{metric}"] = value
        else:
            flat_report[label] = values
    pd.DataFrame([flat_report]).round(4).to_csv(
        TABLE_DIR / "classification_report.csv", index=False
    )

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, probabilities, best.name)
    plot_calibration(y_test, probabilities)

    banner("9. EXPLAINABILITY")
    importance = permutation_importance_table(
        final_model, X_test, y_test, n_repeats=20
    )
    importance.round(5).to_csv(TABLE_DIR / "feature_importance.csv", index=False)
    plot_permutation_importance(importance, best.name)
    log("\nTop drivers (permutation importance):")
    for _, row in importance.head(6).iterrows():
        log(f"  {row['feature']:<28s} {row['importance_mean']:+.4f}")

    coefficients = model_coefficients(final_model)
    if coefficients is not None:
        coefficients.round(5).to_csv(
            TABLE_DIR / "logistic_coefficients.csv", index=False
        )
        plot_logistic_coefficients(coefficients)
        log("Saved logistic-regression coefficients.")

    top_features = [f for f in importance["feature"].head(4) if f in X.columns]
    plot_partial_dependence(final_model, X_test, top_features, best.name)
    if not args.fast:
        plot_learning_curve(final_model, X_train, y_train, best.name)

    banner("10. SAVING MODEL AND ARTEFACTS")
    reference_stats = build_reference_stats(X_train)
    bundle = {
        "model": final_model,
        "model_name": best.name,
        "threshold": best_threshold,
        "default_threshold": 0.5,
        "feature_columns": KEY_FEATURES,
        "reference_stats": reference_stats,
        "metrics": metrics_tuned,
        "metrics_default_threshold": metrics_50,
        "confidence_intervals": ci,
        "cv_metrics": best.cv_metrics,
        "best_params": best.best_params,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "version": "2.0.0",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "dataset": str(args.data),
    }
    save_bundle(bundle, BUNDLE_PATH)

    summary = {
        "generated_at": bundle["trained_at"],
        "dataset": str(args.data),
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_balance": checks["class_balance_pct"],
        "selected_model": best.name,
        "selected_model_notes": best.notes,
        "best_params": best.best_params,
        "cv_metrics": best.cv_metrics,
        "threshold": best_threshold,
        "test_metrics_default": metrics_50,
        "test_metrics_tuned": metrics_tuned,
        "confidence_intervals": ci,
        "score_label_agreement": checks.get("score_label_agreement"),
        "features_used": KEY_FEATURES,
        "figures": {k: str(v) for k, v in figures.items()},
        "tuning": tuning_rows,
    }
    save_json(summary, RESULTS_JSON)

    log(f"\nModel bundle  -> {BUNDLE_PATH}")
    log(f"Results JSON  -> {RESULTS_JSON}")
    log(f"Figures       -> {FIGURE_DIR}")
    log(f"Tables        -> {TABLE_DIR}")
    log(f"\nTotal time: {time.time() - started:.1f} seconds")
    log("\nDone. Next steps:")
    log("  python predict.py --input data/sample_fields.csv")
    log("  streamlit run app.py")
    log("  python generate_report.py")


if __name__ == "__main__":
    main()
