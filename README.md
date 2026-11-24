# Machine Learning-Based Organic Farming Suitability Prediction

**BCA 5th Semester PBL Project** — predicts whether a crop-field combination is suitable for organic farming from soil chemistry and management history.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 What it does

Given **9 soil/management inputs**:
| Input | Range / Values |
|-------|----------------|
| pH | 4.5 – 9.5 |
| Organic Carbon (%) | 0.1 – 2.6 |
| Available N (kg/ha) | 60 – 600 |
| Available P (kg/ha) | 2 – 90 |
| Available K (kg/ha) | 35 – 660 |
| Available S (mg/kg) | 2 – 40 |
| Previous Fertilizer Use | Chemical / Mixed / Organic |
| Previous Pesticide Use | High / Moderate / Low |
| Years of Chemical Input | 0 – 20 |

The model outputs:
- **Probability of "Suitable"** (0–100%)
- **Binary label** at the tuned threshold (default ≈ 0.47)
- **Per-field explanation** (which factors helped/hurt)
- **Agronomic recommendations**

---

## 📊 Results (v2)

| Metric | v1 (original) | **v2 (this repo)** |
|--------|---------------|-------------------|
| Accuracy | 80.5% | **85%** |
| F1 Score | 0.764 | **0.83** |
| ROC-AUC | 0.889 | **0.917** |
| Models compared | 4 (only RF tuned) | **7–9 (all tuned)** |
| Test-set leakage | Yes (best test score) | **No (repeated CV)** |
| Threshold | Fixed 0.5 | **Tuned on OOF** |
| Explainability | Global only | **Global + per-field** |
| Deployment | None | **Streamlit + CLI** |

**Selected model:** Logistic Regression (parsimony rule — same AUC as boosting, fully interpretable).

---

## 🚀 Quick start

```bash
# 1. Clone
git clone https://github.com/muntajirkhanncoding-ai/organic-farming-pbl.git
cd organic-farming-pbl

# 2. Create venv & install
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt

# 3. Train (optional — model already in outputs/models/)
python train.py --fast          # ~60s, quick run
python train.py                 # full tuning, ~8–10 min

# 4. Batch prediction
python predict.py --input data/sample_fields.csv

# 5. Interactive web app
streamlit run app.py
# Opens http://localhost:8501

# 6. Generate Word report
python generate_report.py
# Output: docs/Organic_Farming_Project_Report.docx
```

---

## 📁 Project structure

```
.
├── data/
│   ├── organic_farming_suitability.csv   # 1000-row dataset (synthetic)
│   └── sample_fields.csv                 # 6 demo fields
├── docs/
│   ├── Organic_Farming_Project_Report.docx
│   ├── Organic_Farming_Project_Report.pdf
│   ├── DATA_DICTIONARY.md                # Column reference
│   └── VIVA_QA.md                        # 32 viva questions
├── outputs/
│   ├── figures/                          # 14 PNG charts
│   ├── tables/                           # 10 CSV result tables
│   ├── models/best_model.joblib          # Trained pipeline + threshold
│   └── results_summary.json              # All metrics
├── src/organic_ml/                       # Reusable ML package
│   ├── config.py       # Paths, features, constants
│   ├── data.py         # Loading, validation, audit
│   ├── features.py     # Engineering + preprocessing
│   ├── modeling.py     # Tuning, comparison, evaluation
│   ├── plots.py        # All figures
│   ├── explain.py      # Permutation importance + per-field
│   └── io.py           # Save/load bundles & JSON
├── train.py              # Train & evaluate
├── predict.py            # Batch CSV scoring
├── app.py                # Streamlit web app
├── generate_report.py    # Build Word report
├── requirements.txt
├── requirements-dev.txt
├── tests/test_pipeline.py
└── .gitignore
```

---

## 🧪 Run tests

```bash
pytest -q
```

---

## ⚠️ Important limitations

- **Dataset is synthetic** — created for teaching, not real field records.
- **Labels come from an instructional rubric** — not official organic certification.
- The model is a **screening aid**, not a substitute for lab soil tests or extension advice.
- Context columns (State, Soil Type, Crop) showed no signal in this dataset and are excluded from the model.

---

## 📚 References

- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, 2nd ed.
- FAO Organic Agriculture Guidelines
- ICAR Organic Farming Guidelines

---

## 👨‍💻 Author

**Md Sharik**  
BCA 5th Semester, PBL Project  
Guided by Dr. Sreekumar Biswas (Asst. Prof. of ML)

---

## 📄 License

MIT — free to use, modify, and distribute.