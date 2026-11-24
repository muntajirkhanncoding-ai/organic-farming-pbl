"""Central configuration: paths, feature lists and modelling constants.

Keeping these values in one place makes the pipeline easy to reproduce and
easy to explain during a viva.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

DEFAULT_DATA_PATH = DATA_DIR / "organic_farming_suitability.csv"
BUNDLE_PATH = MODEL_DIR / "best_model.joblib"
RESULTS_JSON = OUTPUT_DIR / "results_summary.json"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 3          # used for the final repeated cross-validation table
N_BOOTSTRAP = 1000      # bootstrap confidence intervals

# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------
ID_COL = "Sample_ID"
TARGET_COL = "Organic_Farming_Suitability"
SCORE_COL = "Suitability_Score"

# Raw columns that actually carry signal (selected after EDA + permutation
# importance).  Location / crop / micronutrients / weather are kept in the
# CSV for context but are NOT used by the model: they showed no measurable
# relationship with the label and only added noise.
NUMERIC_FEATURES = [
    "pH",
    "Organic_Carbon_pct",
    "Available_N_kg_ha",
    "Available_P_kg_ha",
    "Available_K_kg_ha",
    "Available_S_mg_kg",
    "Years_Chemical_Input",
]

CATEGORICAL_FEATURES = [
    "Previous_Fertilizer_Usage",
    "Previous_Pesticide_Usage",
]

KEY_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Engineered features created in features.py.  They encode the *banded*
# agronomic rubric that a plain linear model cannot discover by itself.
ENGINEERED_NUMERIC = [
    "chem_score",
    "pest_score",
    "ph_in_optimal",
    "ph_distance",
    "ph_low",
    "ph_high",
    "ph_too_high",
    "oc_sufficient",
    "oc_high",
    "n_sufficient",
    "n_high",
    "p_sufficient",
    "k_sufficient",
    "k_high",
    "s_sufficient",
    "nutrient_sufficiency_index",
    "chemical_burden",
    "stewardship",
    "ph_x_chemical",
    "organic_history",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + ENGINEERED_NUMERIC + CATEGORICAL_FEATURES

# Valid input ranges.  Used for validation and for the Streamlit sliders.
FEATURE_RANGES = {
    "pH": (4.5, 9.5, 6.8, 0.05),
    "Organic_Carbon_pct": (0.1, 2.6, 0.7, 0.01),
    "Available_N_kg_ha": (60.0, 600.0, 250.0, 1.0),
    "Available_P_kg_ha": (2.0, 90.0, 25.0, 0.5),
    "Available_K_kg_ha": (35.0, 660.0, 230.0, 1.0),
    "Available_S_mg_kg": (2.0, 40.0, 14.0, 0.5),
    "Years_Chemical_Input": (0, 20, 3, 1),
}

# Human-readable labels used in plots, the app and the report.
FEATURE_LABELS = {
    "pH": "Soil pH",
    "Organic_Carbon_pct": "Organic carbon (%)",
    "Available_N_kg_ha": "Available nitrogen (kg/ha)",
    "Available_P_kg_ha": "Available phosphorus (kg/ha)",
    "Available_K_kg_ha": "Available potassium (kg/ha)",
    "Available_S_mg_kg": "Available sulphur (mg/kg)",
    "Years_Chemical_Input": "Years of chemical input",
    "Previous_Fertilizer_Usage": "Previous fertilizer use",
    "Previous_Pesticide_Usage": "Previous pesticide use",
    "chem_score": "Organic fertilizer history score",
    "pest_score": "Low-pesticide history score",
    "ph_in_optimal": "pH in optimal band (6.0-7.8)",
    "ph_distance": "Distance from ideal pH (6.9)",
    "ph_low": "pH below 6.0",
    "ph_high": "pH above 7.8",
    "ph_too_high": "pH above 8.2",
    "oc_sufficient": "Organic carbon sufficient (>= 0.75%)",
    "oc_high": "Organic carbon high (>= 1.15%)",
    "n_sufficient": "Nitrogen sufficient (>= 280 kg/ha)",
    "n_high": "Nitrogen high (>= 375 kg/ha)",
    "p_sufficient": "Phosphorus sufficient (>= 25 kg/ha)",
    "k_sufficient": "Potassium sufficient (>= 250 kg/ha)",
    "k_high": "Potassium high (>= 435 kg/ha)",
    "s_sufficient": "Sulphur sufficient (>= 15 mg/kg)",
    "nutrient_sufficiency_index": "Nutrient sufficiency index (0-5)",
    "chemical_burden": "Cumulative chemical burden",
    "stewardship": "Stewardship score",
    "ph_x_chemical": "Optimal pH x organic history",
    "organic_history": "Previous fertilizer was organic",
}

# Consistent colours used in every figure and in the app.
COLORS = {
    "suitable": "#2a9d8f",
    "not_suitable": "#e76f51",
    "accent": "#264653",
    "muted": "#8ab17d",
    "gold": "#e9c46a",
    "bg": "#f7f9f4",
}
