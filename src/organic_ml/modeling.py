"""Model definitions, hyper-parameter tuning and evaluation."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import CV_FOLDS, CV_REPEATS, RANDOM_STATE
from .features import build_pipeline

# Optional boosting libraries.  The project works without them, but if they
# are installed they are included and usually perform very well.
try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # pragma: no cover
    HAS_XGBOOST = False

try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover
    HAS_LIGHTGBM = False


# --------------------------------------------------------------------------
# Model catalogue
# --------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    pipeline: Any
    param_distributions: Dict[str, Any]
    complexity: int = 0          # lower = simpler / more explainable
    notes: str = ""


def build_model_zoo(include_optional: bool = True) -> List[ModelSpec]:
    """Return every candidate model with its tuning search space."""
    zoo: List[ModelSpec] = []

    zoo.append(
        ModelSpec(
            name="Logistic Regression",
            pipeline=build_pipeline(
                LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
                scale_numeric=True,
            ),
            param_distributions={
                "model__C": loguniform(0.01, 10.0),
                "model__class_weight": [None, "balanced"],
            },
            complexity=0,
            notes="Linear, fully interpretable coefficients.",
        )
    )

    zoo.append(
        ModelSpec(
            name="Decision Tree",
            pipeline=build_pipeline(
                DecisionTreeClassifier(random_state=RANDOM_STATE),
                scale_numeric=False,
            ),
            param_distributions={
                "model__max_depth": randint(3, 13),
                "model__min_samples_leaf": randint(2, 31),
                "model__criterion": ["gini", "entropy"],
                "model__class_weight": [None, "balanced"],
            },
            complexity=1,
            notes="Rule-based, easy to draw on the board.",
        )
    )

    zoo.append(
        ModelSpec(
            name="K-Nearest Neighbours",
            pipeline=build_pipeline(
                KNeighborsClassifier(), scale_numeric=True
            ),
            param_distributions={
                "model__n_neighbors": randint(5, 42),
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            complexity=2,
            notes="Instance-based baseline.",
        )
    )

    zoo.append(
        ModelSpec(
            name="Random Forest",
            pipeline=build_pipeline(
                RandomForestClassifier(
                    n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1
                ),
                scale_numeric=False,
            ),
            param_distributions={
                "model__n_estimators": randint(250, 800),
                "model__max_depth": randint(4, 21),
                "model__min_samples_leaf": randint(1, 16),
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [None, "balanced_subsample"],
            },
            complexity=3,
            notes="Bagged trees, robust to outliers.",
        )
    )

    zoo.append(
        ModelSpec(
            name="Extra Trees",
            pipeline=build_pipeline(
                ExtraTreesClassifier(
                    n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1
                ),
                scale_numeric=False,
            ),
            param_distributions={
                "model__n_estimators": randint(250, 800),
                "model__max_depth": randint(4, 21),
                "model__min_samples_leaf": randint(1, 16),
                "model__max_features": ["sqrt", "log2", None],
            },
            complexity=4,
            notes="Extremely randomised trees.",
        )
    )

    zoo.append(
        ModelSpec(
            name="Gradient Boosting",
            pipeline=build_pipeline(
                GradientBoostingClassifier(random_state=RANDOM_STATE),
                scale_numeric=False,
            ),
            param_distributions={
                "model__n_estimators": randint(150, 650),
                "model__learning_rate": uniform(0.01, 0.14),
                "model__max_depth": randint(2, 5),
                "model__subsample": uniform(0.65, 0.35),
                "model__min_samples_leaf": randint(3, 31),
            },
            complexity=5,
            notes="Boosted trees, strong on tabular data.",
        )
    )

    zoo.append(
        ModelSpec(
            name="Hist Gradient Boosting",
            pipeline=build_pipeline(
                HistGradientBoostingClassifier(random_state=RANDOM_STATE),
                scale_numeric=False,
            ),
            param_distributions={
                "model__max_iter": randint(200, 750),
                "model__learning_rate": uniform(0.01, 0.14),
                "model__max_leaf_nodes": randint(7, 32),
                "model__min_samples_leaf": randint(5, 41),
                "model__l2_regularization": uniform(0.0, 5.0),
            },
            complexity=6,
            notes="Fast histogram-based boosting.",
        )
    )

    if include_optional and HAS_XGBOOST:
        zoo.append(
            ModelSpec(
                name="XGBoost",
                pipeline=build_pipeline(
                    XGBClassifier(
                        n_estimators=400,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        eval_metric="logloss",
                    ),
                    scale_numeric=False,
                ),
                param_distributions={
                    "model__n_estimators": randint(250, 900),
                    "model__learning_rate": uniform(0.01, 0.11),
                    "model__max_depth": randint(2, 6),
                    "model__subsample": uniform(0.6, 0.4),
                    "model__colsample_bytree": uniform(0.5, 0.45),
                    "model__reg_lambda": uniform(0.5, 5.0),
                    "model__min_child_weight": randint(1, 9),
                },
                complexity=7,
                notes="Regularised gradient boosting.",
            )
        )

    if include_optional and HAS_LIGHTGBM:
        zoo.append(
            ModelSpec(
                name="LightGBM",
                pipeline=build_pipeline(
                    LGBMClassifier(
                        n_estimators=400,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                    scale_numeric=False,
                ),
                param_distributions={
                    "model__n_estimators": randint(250, 900),
                    "model__learning_rate": uniform(0.01, 0.11),
                    "model__num_leaves": randint(7, 32),
                    "model__min_child_samples": randint(5, 41),
                    "model__subsample": uniform(0.6, 0.4),
                    "model__colsample_bytree": uniform(0.5, 0.45),
                    "model__reg_lambda": uniform(0.5, 5.0),
                },
                complexity=8,
                notes="Leaf-wise gradient boosting, very fast.",
            )
        )

    return zoo


# --------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------
@dataclass
class TunedModel:
    name: str
    estimator: Any
    best_params: Dict[str, Any]
    cv_auc_mean: float
    cv_auc_std: float
    complexity: int
    notes: str = ""
    cv_metrics: Dict[str, float] = field(default_factory=dict)


def tune_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 25,
    cv: int = CV_FOLDS,
    verbose: bool = True,
) -> TunedModel:
    """Tune one model with randomised search on training data only."""
    search = RandomizedSearchCV(
        estimator=spec.pipeline,
        param_distributions=spec.param_distributions,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0,
        error_score="raise",
    )
    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        search.fit(X_train, y_train)
    elapsed = time.time() - start

    best_index = search.best_index_
    cv_std = float(search.cv_results_["std_test_score"][best_index])

    if verbose:
        print(
            f"  {spec.name:<26s} CV AUC = {search.best_score_:.4f} "
            f"+/- {cv_std:.4f}   ({elapsed:.0f}s)",
            flush=True,
        )

    return TunedModel(
        name=spec.name,
        estimator=search.best_estimator_,
        best_params={k: v for k, v in search.best_params_.items()},
        cv_auc_mean=float(search.best_score_),
        cv_auc_std=cv_std,
        complexity=spec.complexity,
        notes=spec.notes,
    )


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 25,
    cv: int = CV_FOLDS,
    include_optional: bool = True,
    verbose: bool = True,
) -> List[TunedModel]:
    """Tune every model in the catalogue."""
    specs = build_model_zoo(include_optional=include_optional)
    if verbose:
        print(f"\nTuning {len(specs)} models (n_iter={n_iter}, cv={cv})...")
    return [
        tune_model(spec, X_train, y_train, n_iter=n_iter, cv=cv, verbose=verbose)
        for spec in specs
    ]


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def compare_models(
    models: List[TunedModel],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    repeats: int = CV_REPEATS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Repeated stratified CV comparison of the tuned models."""
    cv = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=repeats, random_state=RANDOM_STATE
    )
    rows = []
    if verbose:
        print(f"\nRepeated {CV_FOLDS}x{repeats} cross-validation on the training set...")

    for item in models:
        result = cross_validate(
            item.estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            error_score="raise",
        )
        row: Dict[str, Any] = {"model": item.name}
        for metric in SCORING:
            values = result[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        row["complexity"] = item.complexity
        row["notes"] = item.notes
        row["best_params"] = item.best_params
        rows.append(row)

        item.cv_metrics = {
            "roc_auc": row["roc_auc_mean"],
            "f1": row["f1_mean"],
            "accuracy": row["accuracy_mean"],
        }

        if verbose:
            print(
                f"  {item.name:<26s} AUC {row['roc_auc_mean']:.4f} | "
                f"F1 {row['f1_mean']:.4f} | Acc {row['accuracy_mean']:.4f}"
            )

    comparison = pd.DataFrame(rows)
    return comparison


def select_best_model(
    models: List[TunedModel],
    comparison: pd.DataFrame,
    tie_tolerance: float = 0.002,
) -> TunedModel:
    """Pick the best model by CV ROC-AUC, preferring simpler models on ties.

    A 0.002 AUC difference is smaller than the fold-to-fold noise, so in that
    band we deliberately choose the simpler / more explainable model.  This
    is the "parsimony" tie-break rule.
    """
    by_name = {m.name: m for m in models}
    ranked = comparison.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    best_auc = float(ranked.loc[0, "roc_auc_mean"])
    eligible = ranked[ranked["roc_auc_mean"] >= best_auc - tie_tolerance]
    chosen_row = eligible.sort_values(["complexity", "roc_auc_mean"], ascending=[True, False]).iloc[0]
    chosen = by_name[str(chosen_row["model"])]
    chosen.cv_metrics = {
        "roc_auc": float(chosen_row["roc_auc_mean"]),
        "f1": float(chosen_row["f1_mean"]),
        "accuracy": float(chosen_row["accuracy_mean"]),
    }
    return chosen


# --------------------------------------------------------------------------
# Threshold and hold-out evaluation
# --------------------------------------------------------------------------
def out_of_fold_probabilities(
    estimator: Any, X_train: pd.DataFrame, y_train: pd.Series
) -> np.ndarray:
    """Cross-validated probabilities for every training row."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]


def tune_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    min_gain: float = 0.02,
    default_threshold: float = 0.5,
) -> Tuple[float, pd.DataFrame]:
    """Choose the probability cut-off that maximises F1 on OOF predictions.

    The default 0.5 is arbitrary.  For a screening tool, shifting the cut-off
    changes the precision/recall trade-off.  To avoid over-fitting the
    threshold to a single cross-validation split, the tuned value is only
    adopted when it improves OOF F1 by at least ``min_gain`` over 0.5.
    """
    thresholds = np.round(np.arange(0.10, 0.901, 0.005), 3)
    rows = []
    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy_score(y_true, pred),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
            }
        )
    table = pd.DataFrame(rows)
    best_index = table["f1"].idxmax()
    best_threshold = float(table.loc[best_index, "threshold"])
    best_f1 = float(table.loc[best_index, "f1"])

    default_row = table.iloc[(table["threshold"] - default_threshold).abs().argsort()[:1]]
    default_f1 = float(default_row["f1"].iloc[0])

    if best_f1 - default_f1 < min_gain:
        return float(default_threshold), table
    return best_threshold, table


def compute_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """All classification metrics reported in the project."""
    pred = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "brier": float(brier_score_loss(y_true, probabilities)),
    }


def bootstrap_ci(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    metric: str,
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for one metric."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    p = np.asarray(probabilities)
    values = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b, p_b = y[idx], p[idx]
        if len(np.unique(y_b)) < 2:
            continue
        if metric == "roc_auc":
            values.append(roc_auc_score(y_b, p_b))
        else:
            pred = (p_b >= threshold).astype(int)
            if metric == "accuracy":
                values.append(accuracy_score(y_b, pred))
            elif metric == "f1":
                values.append(f1_score(y_b, pred, zero_division=0))
            else:
                raise ValueError(f"Unsupported metric: {metric}")
    if not values:
        return (float("nan"), float("nan"))
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def evaluate_holdout(
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> Tuple[Dict[str, float], np.ndarray, pd.DataFrame]:
    """Final, one-time evaluation on the untouched test set."""
    probabilities = estimator.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, probabilities, threshold)
    predictions = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "probability_suitable": probabilities,
            "predicted": (probabilities >= threshold).astype(int),
        }
    )
    return metrics, probabilities, predictions


def reliability_table(
    y_true: pd.Series, probabilities: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Binned calibration table (predicted vs observed frequency)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    labels = pd.cut(probabilities, bins=bins, include_lowest=True)
    table = (
        pd.DataFrame({"prob": probabilities, "actual": np.asarray(y_true), "bin": labels})
        .groupby("bin", observed=True)
        .agg(count=("actual", "size"), mean_predicted=("prob", "mean"), observed=("actual", "mean"))
        .reset_index(drop=True)
    )
    return table


def model_coefficients(estimator: Any) -> Optional[pd.DataFrame]:
    """Extract logistic-regression coefficients with readable names."""
    try:
        model = estimator.named_steps["model"]
        pre = estimator.named_steps["preprocess"]
    except AttributeError:
        return None
    if not hasattr(model, "coef_"):
        return None

    names = pre.get_feature_names_out()
    coefs = np.ravel(model.coef_)
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    table["abs_coefficient"] = table["coefficient"].abs()
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def get_final_estimator(model: TunedModel) -> Any:
    """Return a fresh clone of the tuned pipeline (safe to refit)."""
    return clone(model.estimator)
