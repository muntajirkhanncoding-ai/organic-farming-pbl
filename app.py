"""Interactive Streamlit demo for the Organic Farming Suitability project.

Run with:
    streamlit run app.py

The app loads the model bundle produced by ``train.py``.  If the bundle is
missing, run ``python train.py`` first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from organic_ml.config import (  # noqa: E402
    BUNDLE_PATH,
    COLORS,
    FEATURE_LABELS,
    FEATURE_RANGES,
    FIGURE_DIR,
    KEY_FEATURES,
    RESULTS_JSON,
)
from organic_ml.explain import (  # noqa: E402
    confidence_label,
    local_explanations,
    recommendations,
)
from organic_ml.io import load_bundle, load_json  # noqa: E402

st.set_page_config(
    page_title="Organic Farming Suitability",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --green: #2a9d8f;
        --dark: #264653;
        --orange: #e76f51;
        --soft: #f3f7ef;
    }
    .main .block-container {padding-top: 1.8rem; max-width: 1220px;}
    .hero {
        background: linear-gradient(120deg, #264653 0%, #2a9d8f 100%);
        color: white; padding: 26px 30px; border-radius: 18px;
        margin-bottom: 22px;
    }
    .hero h1 {color: white; margin: 0 0 6px 0; font-size: 2.0rem;}
    .hero p {margin: 0; opacity: .9; font-size: 1.02rem;}
    .result-card {
        border-radius: 18px; padding: 26px; text-align: center;
        border: 1px solid #dce6d8; background: white;
        box-shadow: 0 6px 24px rgba(38,70,83,.08);
    }
    .result-suitable {border-left: 8px solid #2a9d8f;}
    .result-not {border-left: 8px solid #e76f51;}
    .big-label {font-size: 1.65rem; font-weight: 800; margin: 4px 0 2px 0;}
    .prob {font-size: 3.3rem; font-weight: 800; line-height: 1.0; color: #264653;}
    .muted {color: #6b7c72; font-size: .92rem;}
    .pill {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background: #eaf4ef; color: #1f6f62; font-weight: 700; font-size: .82rem;
        margin-top: 8px;
    }
    .advice {
        background: #f8faf5; border: 1px solid #e3ebdc; border-radius: 12px;
        padding: 12px 16px; margin-bottom: 8px;
    }
    .prio-high {color: #b23a2b; font-weight: 800;}
    .prio-medium {color: #c07a12; font-weight: 800;}
    .prio-low {color: #2a7f62; font-weight: 800;}
    .prio-info {color: #264653; font-weight: 800;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        background: #eef4ea; border-radius: 10px 10px 0 0; padding: 10px 20px;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained model...")
def get_bundle(path: str):
    return load_bundle(path)


@st.cache_data(show_spinner=False)
def get_results(path: str):
    try:
        return load_json(path)
    except Exception:
        return None


if not BUNDLE_PATH.exists():
    st.error(
        "**Model not found.** Run `python train.py` in the project folder first, "
        "then refresh this page."
    )
    st.stop()

bundle = get_bundle(str(BUNDLE_PATH))
model = bundle["model"]
threshold = float(bundle["threshold"])
reference = bundle["reference_stats"]
results = get_results(str(RESULTS_JSON))

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌱 Organic Farming Suitability")
    st.caption("BCA PBL · Machine Learning project · v2.0")
    st.divider()

    st.markdown("**Model**")
    st.write(bundle["model_name"])
    st.caption(f"Decision threshold: **{threshold:.2f}**")
    st.caption(f"Trained on {bundle['n_train']} fields · tested on {bundle['n_test']}")

    if results:
        st.metric(
            "Test accuracy",
            f"{results['test_metrics_tuned']['accuracy']:.1%}",
            help="Accuracy on the untouched 20% test set at the tuned threshold.",
        )
        st.metric(
            "ROC-AUC",
            f"{results['test_metrics_tuned']['roc_auc']:.3f}",
            help="0.5 = random, 1.0 = perfect ranking.",
        )

    st.divider()
    st.markdown("**Decision threshold**")
    user_threshold = st.slider(
        "Higher = stricter",
        min_value=0.20,
        max_value=0.80,
        value=float(threshold),
        step=0.01,
        help="Lower it to flag more fields as suitable; raise it to be more conservative.",
    )
    st.caption(
        "The tuned value maximises F1 on out-of-fold predictions. "
        "Use a higher value if false 'Suitable' recommendations are costly."
    )
    st.divider()
    st.warning(
        "Educational demo on **synthetic** data. It is not an official "
        "organic certification tool."
    )

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🌱 Can this field go organic?</h1>
      <p>Enter the soil test and past-input details. The model estimates the
      probability that the field is suitable for organic farming and explains
      which factors helped or hurt the decision.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Example presets
# --------------------------------------------------------------------------
EXAMPLES = {
    "Custom values": None,
    "Strong candidate": {
        "pH": 6.9,
        "Organic_Carbon_pct": 1.4,
        "Available_N_kg_ha": 420.0,
        "Available_P_kg_ha": 45.0,
        "Available_K_kg_ha": 500.0,
        "Available_S_mg_kg": 22.0,
        "Years_Chemical_Input": 1,
        "Previous_Fertilizer_Usage": "Organic",
        "Previous_Pesticide_Usage": "Low",
    },
    "Borderline field": {
        "pH": 6.95,
        "Organic_Carbon_pct": 0.482,
        "Available_N_kg_ha": 367.1,
        "Available_P_kg_ha": 42.5,
        "Available_K_kg_ha": 79.2,
        "Available_S_mg_kg": 17.36,
        "Years_Chemical_Input": 9,
        "Previous_Fertilizer_Usage": "Chemical",
        "Previous_Pesticide_Usage": "Low",
    },
    "Weak candidate": {
        "pH": 5.2,
        "Organic_Carbon_pct": 0.3,
        "Available_N_kg_ha": 120.0,
        "Available_P_kg_ha": 8.0,
        "Available_K_kg_ha": 100.0,
        "Available_S_mg_kg": 6.0,
        "Years_Chemical_Input": 12,
        "Previous_Fertilizer_Usage": "Chemical",
        "Previous_Pesticide_Usage": "High",
    },
}


def _defaults() -> dict:
    values = {}
    for col in KEY_FEATURES:
        if col in FEATURE_RANGES:
            values[col] = float(reference.get(col, FEATURE_RANGES[col][2]))
        else:
            values[col] = str(reference.get(col, "Mixed"))
    return values


def load_example() -> None:
    choice = st.session_state.get("example_choice", "Custom values")
    example = EXAMPLES.get(choice)
    if example:
        for key, value in example.items():
            st.session_state[f"in_{key}"] = value


if "example_choice" not in st.session_state:
    st.session_state["example_choice"] = "Custom values"

defaults = _defaults()
for col, value in defaults.items():
    st.session_state.setdefault(f"in_{col}", value)

st.selectbox(
    "Start from an example",
    list(EXAMPLES.keys()),
    key="example_choice",
    on_change=load_example,
)

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_predict, tab_insights, tab_about = st.tabs(
    ["🔎 Predict", "📊 Model insights", "📖 About the project"]
)

with tab_predict:
    with st.form("field_form", border=True):
        st.markdown("#### Soil test values")
        c1, c2, c3 = st.columns(3)

        with c1:
            ph = st.number_input(
                "Soil pH",
                min_value=4.5,
                max_value=9.5,
                step=0.05,
                key="in_pH",
                help="pH 6.0-7.8 is the optimal band learned from the data.",
            )
            oc = st.number_input(
                "Organic carbon (%)",
                min_value=0.1,
                max_value=2.6,
                step=0.01,
                key="in_Organic_Carbon_pct",
            )
            years = st.number_input(
                "Years of chemical input",
                min_value=0,
                max_value=20,
                step=1,
                key="in_Years_Chemical_Input",
            )

        with c2:
            n = st.number_input(
                "Available nitrogen (kg/ha)",
                min_value=60.0,
                max_value=600.0,
                step=1.0,
                key="in_Available_N_kg_ha",
            )
            p = st.number_input(
                "Available phosphorus (kg/ha)",
                min_value=2.0,
                max_value=90.0,
                step=0.5,
                key="in_Available_P_kg_ha",
            )
            k = st.number_input(
                "Available potassium (kg/ha)",
                min_value=35.0,
                max_value=660.0,
                step=1.0,
                key="in_Available_K_kg_ha",
            )

        with c3:
            s = st.number_input(
                "Available sulphur (mg/kg)",
                min_value=2.0,
                max_value=40.0,
                step=0.5,
                key="in_Available_S_mg_kg",
            )
            fert = st.selectbox(
                "Previous fertilizer use",
                ["Organic", "Mixed", "Chemical"],
                key="in_Previous_Fertilizer_Usage",
            )
            pest = st.selectbox(
                "Previous pesticide use",
                ["Low", "Moderate", "High"],
                key="in_Previous_Pesticide_Usage",
            )

        st.caption(
            "Optional context (not used by the model): "
            "location, soil type and crop are recorded for farm reports only."
        )
        o1, o2, o3 = st.columns(3)
        with o1:
            st.text_input("State / region (optional)", placeholder="e.g. Bihar")
        with o2:
            st.selectbox(
                "Soil type (optional)",
                ["", "Alluvial", "Black", "Clay Loam", "Laterite", "Loamy", "Red", "Sandy"],
            )
        with o3:
            st.text_input("Crop (optional)", placeholder="e.g. Rice")

        submitted = st.form_submit_button(
            "🌱 Check suitability", use_container_width=True, type="primary"
        )

    if submitted:
        row = pd.DataFrame(
            [
                {
                    "pH": ph,
                    "Organic_Carbon_pct": oc,
                    "Available_N_kg_ha": n,
                    "Available_P_kg_ha": p,
                    "Available_K_kg_ha": k,
                    "Available_S_mg_kg": s,
                    "Years_Chemical_Input": years,
                    "Previous_Fertilizer_Usage": fert,
                    "Previous_Pesticide_Usage": pest,
                }
            ]
        )

        probability = float(model.predict_proba(row)[0, 1])
        label = "Suitable" if probability >= user_threshold else "Not Suitable"
        card_class = "result-suitable" if label == "Suitable" else "result-not"
        color = COLORS["suitable"] if label == "Suitable" else COLORS["not_suitable"]

        st.markdown("---")
        left, right = st.columns([1.1, 1.9])

        with left:
            st.markdown(
                f"""
                <div class="result-card {card_class}">
                    <div class="muted">Organic farming suitability</div>
                    <div class="big-label" style="color:{color}">{label}</div>
                    <div class="prob">{probability:.0%}</div>
                    <div class="muted">probability of being suitable</div>
                    <div class="pill">{confidence_label(probability)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(probability, 0.0), 1.0))
            st.caption(
                f"Decision threshold: {user_threshold:.2f}. "
                "Raise it if a wrong 'Suitable' call is expensive."
            )

        with right:
            st.markdown("#### What is driving this result?")
            factors = local_explanations(model, row, reference, top_n=7)
            plot_data = factors.iloc[::-1]
            colors = [
                COLORS["suitable"] if v >= 0 else COLORS["not_suitable"]
                for v in plot_data["contribution"]
            ]
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            ax.barh(plot_data["label"], plot_data["contribution"], color=colors, alpha=0.9)
            ax.axvline(0, color="#666", lw=0.8)
            ax.set_xlabel("Change in probability vs a typical field")
            ax.set_title("Feature contributions for this field", fontsize=11)
            ax.tick_params(axis="y", labelsize=8.5)
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.caption(
                "Green bars helped the field; orange bars reduced its score. "
                "Computed by replacing one feature at a time with a typical value."
            )

        st.markdown("#### Recommended actions")
        advice = recommendations(row, probability)
        if not advice:
            st.info("No major correction needed. Keep monitoring soil health.")
        else:
            for item in advice:
                priority = item["priority"].lower()
                st.markdown(
                    f'<div class="advice"><span class="prio-{priority}">'
                    f'{item["priority"]} priority</span> &nbsp; {item["text"]}</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("See the exact input values"):
            st.dataframe(row.T.rename(columns={0: "value"}), use_container_width=True)

with tab_insights:
    st.markdown("### Model performance and explainability")
    if results:
        m1, m2, m3, m4 = st.columns(4)
        tuned = results["test_metrics_tuned"]
        m1.metric("Accuracy", f"{tuned['accuracy']:.1%}")
        m2.metric("F1 score", f"{tuned['f1']:.3f}")
        m3.metric("ROC-AUC", f"{tuned['roc_auc']:.3f}")
        m4.metric("PR-AUC", f"{tuned['pr_auc']:.3f}")
        st.caption(
            f"Selected model: **{results['selected_model']}** · "
            f"test set: {results['test_rows']} fields · "
            f"tuned threshold: {results['threshold']:.2f}"
        )

    figure_files = [
        ("06_model_comparison.png", "Cross-validated comparison of all models"),
        ("07_confusion_matrix.png", "Confusion matrix on the held-out test set"),
        ("08_roc_pr_curves.png", "ROC and precision-recall curves"),
        ("09_calibration.png", "Calibration curve"),
        ("10_threshold_analysis.png", "Threshold tuning"),
        ("12_permutation_importance.png", "Permutation importance"),
        ("13_logistic_coefficients.png", "Logistic-regression coefficients"),
        ("14_partial_dependence.png", "Partial dependence plots"),
        ("15_learning_curve.png", "Learning curve"),
    ]
    available = [(f, c) for f, c in figure_files if (FIGURE_DIR / f).exists()]
    if not available:
        st.info("Figures not found. Run `python train.py` first.")
    else:
        for i in range(0, len(available), 2):
            cols = st.columns(2)
            for col, (filename, caption) in zip(cols, available[i : i + 2]):
                with col:
                    st.image(str(FIGURE_DIR / filename), caption=caption, use_container_width=True)

    with st.expander("Model comparison table"):
        comparison_path = PROJECT_ROOT / "outputs" / "tables" / "model_comparison.csv"
        if comparison_path.exists():
            table = pd.read_csv(comparison_path)
            columns = [
                "model",
                "roc_auc_mean",
                "f1_mean",
                "accuracy_mean",
                "precision_mean",
                "recall_mean",
            ]
            st.dataframe(
                table[[c for c in columns if c in table.columns]].round(4),
                use_container_width=True,
                hide_index=True,
            )

with tab_about:
    st.markdown(
        """
        ### What this project does
        The model predicts whether a field is **suitable for organic farming**
        from nine soil and management inputs. It was trained on 1,000
        crop-field records with four model families and hyper-parameter
        tuning, then evaluated on a held-out 20% test set.

        ### How the decision is made
        1. Inputs are cleaned and 20 agronomic features are engineered
           (pH bands, nutrient sufficiency flags, chemical burden, etc.).
        2. Categorical values are one-hot encoded and numeric values are
           scaled inside a scikit-learn pipeline.
        3. The trained classifier outputs a probability of "Suitable".
        4. The probability is compared with a tuned threshold (not always 0.5).

        ### Why this version is better than v1
        | Aspect | v1 | v2 |
        |---|---|---|
        | Features | all raw columns | signal-only + engineered bands |
        | Tuning | RF only | every model tuned |
        | Selection | best *test* score (leakage) | repeated CV, test used once |
        | Metrics | 5 | 9 + bootstrap confidence intervals |
        | Explainability | global only | global + per-field contributions |
        | Deployment | none | Streamlit app + batch CLI |
        | Report | missing | auto-generated Word report |

        ### Important limitations
        * The dataset is **synthetic / source-calibrated**, not real field data.
        * Labels come from an instructional rubric, not organic certification.
        * The model is a screening aid. Always confirm with a soil-test lab
          and a local extension officer.
        * The optional context fields (state, soil type, crop) are not used by
          the model because they showed no measurable signal in this dataset.

        ### Dataset
        `data/organic_farming_suitability.csv` — 1,000 rows, 24 columns.
        Target: `Organic_Farming_Suitability` (Suitable / Not Suitable).
        """
    )

st.markdown(
    '<p style="text-align:center;color:#8a9a90;font-size:.8rem;">'
    "Organic Farming Suitability · BCA PBL · built with scikit-learn and Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
