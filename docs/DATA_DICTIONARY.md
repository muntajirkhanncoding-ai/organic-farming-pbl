# Data Dictionary

**File:** `data/organic_farming_suitability.csv`
**Rows:** 1,000 · **Columns:** 24 · **Task:** binary classification

> **Note:** this is a source-calibrated **synthetic** dataset built for
> teaching. The labels follow an instructional rubric, not official organic
> certification. See the "Leakage warning" below.

## Columns

| # | Column | Type | Description | Used by model? |
|---|---|---|---|---|
| 1 | `Sample_ID` | text | Row identifier | No |
| 2 | `State` | text | Indian state (15 values) | No (no signal) |
| 3 | `Soil_Type` | text | Alluvial, Black, Clay Loam, Laterite, Loamy, Red, Sandy | No (no signal) |
| 4 | `Crop` | text | 12 crop names | No (no signal) |
| 5 | `pH` | float | Soil reaction, 4.5–9.5 | **Yes** |
| 6 | `Organic_Carbon_pct` | float | Soil organic carbon, % | **Yes** |
| 7 | `Available_N_kg_ha` | float | Available nitrogen, kg/ha | **Yes** |
| 8 | `Available_P_kg_ha` | float | Available phosphorus, kg/ha | **Yes** |
| 9 | `Available_K_kg_ha` | float | Available potassium, kg/ha | **Yes** |
| 10 | `Available_S_mg_kg` | float | Available sulphur, mg/kg | **Yes** |
| 11–15 | `Available_Zn/Fe/Mn/Cu/B_mg_kg` | float | Micronutrients | No (no signal) |
| 16 | `EC_dS_m` | float | Electrical conductivity, dS/m | No (no signal) |
| 17 | `Annual_Rainfall_mm` | float | Annual rainfall, mm | No (no signal) |
| 18 | `Mean_Temperature_C` | float | Mean temperature, °C | No (no signal) |
| 19 | `Soil_Moisture_pct` | float | Soil moisture, % | No (no signal) |
| 20 | `Previous_Fertilizer_Usage` | text | Chemical / Mixed / Organic | **Yes** |
| 21 | `Previous_Pesticide_Usage` | text | High / Moderate / Low | **Yes** |
| 22 | `Years_Chemical_Input` | int | Years of synthetic input use | **Yes** |
| 23 | `Suitability_Score` | float | Rubric score, roughly 4.5–89.3 | **No — leakage** |
| 24 | `Organic_Farming_Suitability` | text | Target: Suitable / Not Suitable | Target |

## Target distribution

| Class | Count | Share |
|---|---|---|
| Not Suitable | 577 | 57.7% |
| Suitable | 423 | 42.3% |

## Leakage warning

`Suitability_Score` agrees with the target for **99.9%** of rows — the label is
essentially a threshold on that score. It must be dropped from the feature set,
otherwise the model would look almost perfect but could never be used on a new
field (where no rubric score exists).

The pipeline detects this automatically in `validate_dataset()` and excludes the
column.

## Why only nine raw features?

Exploratory analysis and permutation importance showed that the following
columns have essentially no relationship with the target in this dataset:

- `State`, `Soil_Type`, `Crop`
- Zn, Fe, Mn, Cu, B micronutrients
- `EC_dS_m`
- `Annual_Rainfall_mm`, `Mean_Temperature_C`, `Soil_Moisture_pct`

Keeping them made the model slightly worse because they add variance without
signal. They remain in the CSV for context and could become useful with a
larger, real-world dataset.

## Valid category values

- `Previous_Fertilizer_Usage`: `Chemical`, `Mixed`, `Organic`
- `Previous_Pesticide_Usage`: `High`, `Moderate`, `Low`
- `Organic_Farming_Suitability`: `Suitable`, `Not Suitable`

The loader accepts case-insensitive variants and common spellings such as
`not_suitable` / `unsuitable`, but raises an error for genuinely unknown labels.
