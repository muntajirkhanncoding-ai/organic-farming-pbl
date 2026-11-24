"""All figures for EDA, evaluation and explainability.

Every function saves a PNG into ``outputs/figures`` and returns the path so
the report generator can embed it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve

from .config import (
    COLORS,
    CV_FOLDS,
    FEATURE_LABELS,
    FIGURE_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)
from .features import clean_feature_names

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 170,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#c9d3c4",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _save(fig: plt.Figure, filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------
def plot_class_balance(y: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    counts = y.map({1: "Suitable", 0: "Not Suitable"}).value_counts()
    counts = counts.reindex(["Not Suitable", "Suitable"])
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[COLORS["not_suitable"], COLORS["suitable"]],
        width=0.55,
    )
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(counts.values) * 0.02,
            f"{value}\n({value / len(y):.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylim(0, max(counts.values) * 1.22)
    ax.set_ylabel("Number of fields")
    ax.set_title("Target class distribution")
    ax.grid(axis="y", alpha=0.35)
    return _save(fig, "01_class_balance.png")


def plot_feature_distributions(df: pd.DataFrame) -> Path:
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        sns.histplot(df[col], kde=True, ax=ax, color=COLORS["accent"], alpha=0.7)
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("")
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Distribution of the key soil features", y=1.01, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "02_feature_distributions.png")


def plot_features_by_class(df: pd.DataFrame, y: pd.Series) -> Path:
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    tmp = df[cols].copy()
    tmp["Class"] = np.where(y == 1, "Suitable", "Not Suitable")

    for ax, col in zip(axes, cols):
        sns.boxplot(
            data=tmp,
            x="Class",
            y=col,
            hue="Class",
            ax=ax,
            palette=[COLORS["not_suitable"], COLORS["suitable"]],
            width=0.5,
            showfliers=False,
            legend=False,
        )
        ax.set_title(col.replace("_", " "))
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle("Key features by suitability class", y=1.01, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "03_features_by_class.png")


def plot_correlation_heatmap(df: pd.DataFrame, y: pd.Series) -> Path:
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    corr = df[cols].copy()
    corr["Suitable"] = y.values
    matrix = corr.corr()

    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation between key features and the target")
    return _save(fig, "04_correlation_heatmap.png")


def plot_suitability_rates(df: pd.DataFrame, y: pd.Series) -> Path:
    """Suitability rate for the categorical context columns."""
    candidates = [
        "Previous_Fertilizer_Usage",
        "Previous_Pesticide_Usage",
        "Soil_Type",
        "Crop",
    ]
    available = [c for c in candidates if c in df.columns]
    ncols = 2
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, available):
        rate = (
            pd.DataFrame({"group": df[col].astype("string"), "target": y.values})
            .groupby("group")["target"]
            .agg(["mean", "count"])
            .sort_values("mean")
        )
        colors = [
            COLORS["suitable"] if v >= 0.5 else COLORS["not_suitable"]
            for v in rate["mean"]
        ]
        bars = ax.barh(rate.index.astype(str), rate["mean"], color=colors, alpha=0.85)
        for bar, (_, row) in zip(bars, rate.iterrows()):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{row['mean']:.0%} (n={int(row['count'])})",
                va="center",
                fontsize=8,
            )
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Share suitable")
        ax.set_title(f"Suitability rate by {col.replace('_', ' ')}")
        ax.grid(axis="x", alpha=0.3)

    for ax in axes[len(available):]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, "05_suitability_rates.png")


# --------------------------------------------------------------------------
# Model comparison
# --------------------------------------------------------------------------
def plot_model_comparison(comparison: pd.DataFrame) -> Path:
    metrics = [
        ("roc_auc_mean", "roc_auc_std", "ROC-AUC"),
        ("f1_mean", "f1_std", "F1 score"),
        ("accuracy_mean", "accuracy_std", "Accuracy"),
    ]
    ordered = comparison.sort_values("roc_auc_mean", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), sharey=True)
    palette = sns.color_palette("crest", n_colors=len(ordered))

    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        y_pos = np.arange(len(ordered))
        ax.barh(
            y_pos,
            ordered[mean_col],
            xerr=ordered[std_col],
            color=palette,
            alpha=0.9,
            capsize=3,
            error_kw={"elinewidth": 1, "ecolor": "#555"},
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ordered["model"])
        ax.set_xlim(0.55, 1.16)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        for y, value, std in zip(y_pos, ordered[mean_col], ordered[std_col]):
            ax.text(
                min(value + std + 0.008, 1.15),
                y,
                f"{value:.3f}",
                va="center",
                fontsize=8.5,
            )

    fig.suptitle("Model comparison - repeated 5-fold cross-validation", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "06_model_comparison.png")


# --------------------------------------------------------------------------
# Final evaluation
# --------------------------------------------------------------------------
def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> Path:
    cm = pd.crosstab(
        pd.Categorical(np.where(y_true == 1, "Suitable", "Not Suitable"), ["Not Suitable", "Suitable"]),
        pd.Categorical(np.where(y_pred == 1, "Suitable", "Not Suitable"), ["Not Suitable", "Suitable"]),
        dropna=False,
    )
    matrix = cm.to_numpy()
    row_sums = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    sns.heatmap(
        matrix,
        annot=False,
        cmap="Greens",
        cbar=False,
        linewidths=1,
        linecolor="white",
        ax=ax,
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{matrix[i, j]}\n({percentages[i, j]:.1%})",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white" if matrix[i, j] > matrix.max() * 0.55 else "#264653",
            )
    ax.set_xticklabels(["Not Suitable", "Suitable"])
    ax.set_yticklabels(["Not Suitable", "Suitable"], rotation=0)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion matrix - held-out test set")
    return _save(fig, "07_confusion_matrix.png")


def plot_roc_curve(y_true: pd.Series, probabilities: np.ndarray, model_name: str) -> Path:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    ap = average_precision_score(y_true, probabilities)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    axes[0].plot(fpr, tpr, color=COLORS["suitable"], lw=2.4, label=f"{model_name} (AUC = {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#999", lw=1)
    axes[0].fill_between(fpr, tpr, alpha=0.10, color=COLORS["suitable"])
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(recall, precision, color=COLORS["accent"], lw=2.4, label=f"{model_name} (AP = {ap:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return _save(fig, "08_roc_pr_curves.png")


def plot_calibration(y_true: pd.Series, probabilities: np.ndarray) -> Path:
    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=8, strategy="quantile")

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.plot([0, 1], [0, 1], "--", color="#999", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", color=COLORS["suitable"], lw=2, label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed suitable rate")
    ax.set_title("Calibration curve")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return _save(fig, "09_calibration.png")


def plot_threshold_analysis(table: pd.DataFrame, chosen_threshold: float) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for col, color, label in [
        ("f1", COLORS["suitable"], "F1"),
        ("precision", COLORS["accent"], "Precision"),
        ("recall", COLORS["gold"], "Recall"),
        ("accuracy", COLORS["muted"], "Accuracy"),
    ]:
        ax.plot(table["threshold"], table[col], lw=2.0, color=color, label=label)
    ax.axvline(chosen_threshold, color=COLORS["not_suitable"], ls="--", lw=1.6)
    ax.text(
        chosen_threshold + 0.012,
        0.05,
        f"chosen = {chosen_threshold:.2f}",
        color=COLORS["not_suitable"],
        fontsize=9,
        rotation=90,
        va="bottom",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold tuning on out-of-fold predictions")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", ncol=2)
    ax.grid(alpha=0.3)
    return _save(fig, "10_threshold_analysis.png")


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------
def plot_permutation_importance(importance: pd.DataFrame, model_name: str, top_n: int = 15) -> Path:
    data = importance.head(top_n).iloc[::-1]
    colors = [COLORS["suitable"] if v >= 0 else COLORS["not_suitable"] for v in data["importance_mean"]]

    fig, ax = plt.subplots(figsize=(7.8, max(4.2, 0.36 * len(data) + 1.2)))
    ax.barh(
        clean_feature_names(data["feature"].tolist()),
        data["importance_mean"],
        xerr=data["importance_std"],
        color=colors,
        alpha=0.9,
        capsize=3,
        error_kw={"elinewidth": 1, "ecolor": "#555"},
    )
    ax.axvline(0, color="#777", lw=0.8)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title(f"Permutation importance - {model_name}")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "12_permutation_importance.png")


def plot_logistic_coefficients(coefficients: pd.DataFrame, top_n: int = 18) -> Path:
    data = coefficients.head(top_n).iloc[::-1]
    names = clean_feature_names(data["feature"].tolist())
    colors = [COLORS["suitable"] if v >= 0 else COLORS["not_suitable"] for v in data["coefficient"]]

    fig, ax = plt.subplots(figsize=(7.8, max(4.2, 0.36 * len(data) + 1.2)))
    ax.barh(names, data["coefficient"], color=colors, alpha=0.9)
    ax.axvline(0, color="#777", lw=0.8)
    ax.set_xlabel("Standardised coefficient (log-odds)")
    ax.set_title("What drives the logistic regression decision")
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=COLORS["suitable"], label="Increases suitability"),
            Patch(color=COLORS["not_suitable"], label="Decreases suitability"),
        ],
        loc="lower right",
        fontsize=8,
    )
    return _save(fig, "13_logistic_coefficients.png")


def plot_partial_dependence(
    estimator,
    X: pd.DataFrame,
    features: Sequence[str],
    model_name: str,
) -> Optional[Path]:
    features = [f for f in features if f in X.columns]
    if not features:
        return None

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    try:
        display = PartialDependenceDisplay.from_estimator(
            estimator,
            X,
            features=list(features),
            kind="average",
            grid_resolution=35,
            n_jobs=1,
            random_state=RANDOM_STATE,
            ax=ax,
            line_kw={"color": COLORS["suitable"], "lw": 2.2},
        )
    except Exception:
        plt.close(fig)
        return None

    # Remove unused panels and show friendly axis labels.
    for current_ax in list(fig.axes):
        if not current_ax.has_data():
            current_ax.remove()
        else:
            xlabel = current_ax.get_xlabel()
            if xlabel in FEATURE_LABELS:
                current_ax.set_xlabel(FEATURE_LABELS[xlabel])
            current_ax.set_ylabel("P(suitable)")

    fig.suptitle(f"Partial dependence - {model_name}", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "14_partial_dependence.png")


def plot_learning_curve(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
) -> Optional[Path]:
    try:
        sizes, train_scores, test_scores = learning_curve(
            estimator,
            X,
            y,
            cv=CV_FOLDS,
            scoring="roc_auc",
            train_sizes=np.linspace(0.15, 1.0, 7),
            n_jobs=-1,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
    except Exception:
        return None

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    test_mean, test_std = test_scores.mean(axis=1), test_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(sizes, train_mean, "o-", color=COLORS["accent"], label="Training score")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color=COLORS["accent"])
    ax.plot(sizes, test_mean, "o-", color=COLORS["suitable"], label="Cross-validation score")
    ax.fill_between(sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color=COLORS["suitable"])
    ax.set_xlabel("Training samples")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Learning curve - {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, "15_learning_curve.png")
