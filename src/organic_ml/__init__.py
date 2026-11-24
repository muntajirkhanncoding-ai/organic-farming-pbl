"""Organic Farming Suitability Prediction - reusable ML package.

This package contains the full, reproducible pipeline used by:
    * train.py             - train and evaluate all models
    * predict.py           - batch prediction from a CSV file
    * app.py               - interactive Streamlit demo
    * generate_report.py   - build the Word project report
"""

from .config import (  # noqa: F401
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COL,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    KEY_FEATURES,
)

__version__ = "2.0.0"
