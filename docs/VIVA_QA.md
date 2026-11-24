# Viva Questions and Answers

A ready-to-revise set of questions for the PBL viva. Answers are deliberately
short enough to say out loud.

## Basics

**1. What is the project about?**
It predicts whether a field is suitable for organic farming using soil test
values and past input-management history. The output is a probability plus a
Suitable / Not Suitable label, along with an explanation of the main factors.

**2. Why is this a machine-learning problem?**
The decision depends on many interacting factors (pH, organic carbon, several
nutrients, past chemical use) and the relationships are non-linear. A fixed
if-else rule would be arbitrary; a model can learn the pattern from labelled
examples.

**3. What type of ML task is this?**
Supervised binary classification. The two classes are Suitable and Not Suitable.

**4. What is the input and output?**
Input: nine raw features — pH, organic carbon, N, P, K, S, previous fertilizer
use, previous pesticide use and years of chemical input. Output: probability of
Suitable, then a label after applying a threshold.

## Dataset

**5. How big is the dataset?**
1,000 rows and 24 columns. 423 suitable fields (42.3%) and 577 not-suitable
(57.7%).

**6. Is the data real?**
No. It is a source-calibrated synthetic dataset for teaching. The labels come
from an instructional rubric, not organic certification. This is stated in the
README and the report.

**7. What is data leakage and where could it occur here?**
Leakage is using information at training time that would not be available when
predicting a new field. `Suitability_Score` is the rubric from which the label
is derived, so it is dropped. Scaling, imputation and encoding are fitted inside
the pipeline on training folds only.

**8. Why did you drop State, Soil_Type, Crop and micronutrients?**
EDA and permutation importance showed near-zero relationship with the target.
Keeping them added noise and slightly reduced accuracy. They remain in the CSV
for context.

**9. How did you handle missing values and duplicates?**
There are none in this dataset. The pipeline still handles them: median
imputation for numeric features, most-frequent imputation for categorical
features, and exact duplicate rows are removed with a log message.

## Features

**10. What feature engineering did you do?**
pH optimal-band indicators, distance from ideal pH, low/high/very-high pH flags,
nutrient sufficiency flags for OC, N, P, K and S, a 0–5 nutrient sufficiency
index, chemical burden, stewardship score, an interaction between optimal pH and
organic history, and an organic-history flag.

**11. Why is feature engineering important here?**
The rubric is banded: pH has an optimum, and nutrients have sufficiency
thresholds. A plain linear model cannot represent a peak or a step. The
engineered flags let even a simple model capture those effects. This is the main
reason v2 beats v1.

**12. Is feature engineering a form of leakage?**
No. Every engineered feature is computed row-by-row from that row's inputs. It
never uses the target or information from other rows.

## Models

**13. Which models did you compare?**
Logistic regression, decision tree, KNN, random forest, extra trees, gradient
boosting, histogram gradient boosting, and optionally XGBoost and LightGBM.

**14. How did you tune them?**
`RandomizedSearchCV` with 5-fold stratified cross-validation on the training
set, using ROC-AUC as the scoring metric. Every model got the same treatment,
which makes the comparison fair.

**15. Why did logistic regression win?**
After feature engineering, the decision boundary is close to linear in the
engineered space. The boosting models had similar AUC but no meaningful gain, so
the parsimony rule (prefer the simpler model when AUC is within 0.002) selected
logistic regression. It is also much easier to explain.

**16. Why not use accuracy for model selection?**
The classes are imbalanced, and accuracy ignores the ranking quality. ROC-AUC is
threshold-independent and measures how well the model separates the classes.
Accuracy, F1, precision and recall are still reported.

**17. What is cross-validation?**
The training data is divided into k folds. The model trains on k−1 folds and
validates on the held-out fold, repeated k times. The mean and standard
deviation give a more stable estimate than a single split.

**18. What is the test set used for?**
Only the final evaluation, after the model and threshold are frozen. This avoids
the v1 mistake of selecting the model using test-set scores.

**19. What is the decision threshold?**
It converts a probability into a class. The default is 0.5. The pipeline tunes
it on out-of-fold predictions to maximise F1, but only adopts a non-default value
if the improvement is at least 0.02, to avoid over-fitting the threshold.

## Evaluation

**20. Which metrics did you report?**
Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Matthews
correlation coefficient and Brier score, with bootstrap 95% confidence
intervals for accuracy, F1 and ROC-AUC.

**21. What does ROC-AUC mean?**
The probability that a randomly chosen suitable field gets a higher predicted
score than a randomly chosen unsuitable field. 0.5 is random, 1.0 is perfect.

**22. What is the confusion matrix telling us?**
It shows true positives, true negatives, false positives and false negatives.
For this run, the model catches most suitable fields but also flags some
unsuitable fields as suitable. The threshold can be raised to reduce false
positives if that error is more expensive.

**23. Why report a Brier score?**
It measures both discrimination and calibration. A lower Brier score means the
predicted probabilities are closer to the observed outcomes.

## Explainability

**24. How do you explain a single prediction?**
Occlusion: replace one feature at a time with a typical training value and
measure the change in predicted probability. A positive change means the actual
value is helping; a negative change means it is hurting. The app shows this as a
bar chart.

**25. What is permutation importance?**
Shuffle one feature and measure how much ROC-AUC drops. A large drop means the
model relies on that feature. It is model-agnostic and computed on the test set.

## Engineering

**26. Why use a scikit-learn Pipeline?**
It bundles feature engineering, imputation, scaling, encoding and the model into
one object. It prevents leakage during cross-validation and makes saving and
loading the model safe.

**27. What is saved in the model bundle?**
The fitted pipeline, the decision threshold, the reference values used for
explanations, the feature list, the test metrics and metadata such as the
training date and version.

**28. How is the app connected to the model?**
The Streamlit app loads `outputs/models/best_model.joblib` with joblib, collects
the nine inputs, calls `predict_proba`, applies the threshold and shows the
per-field explanation and recommendations.

**29. How did you test the code?**
`pytest` checks label parsing, required columns, leakage detection, feature
engineering completeness and idempotence, pipeline training, unseen categories
and the saved bundle.

## Limitations and future work

**30. What are the main limitations?**
Synthetic data, rubric-derived labels, a single country/region, a pooled model
across crops, and recommendations that are general rather than field-specific.
The model is a screening aid, not a certification tool.

**31. How would you improve it with real data?**
Collect real soil-test records with certified outcomes, validate across regions,
add crop-specific models, include multi-year soil-carbon trends, and use a
cost-sensitive threshold based on the farmer's budget.

**32. What did you learn from the project?**
That data quality and feature engineering matter more than model complexity on
small tabular datasets; that test-set discipline is essential; and that an
explainable model is more useful to farmers than a slightly more accurate black
box.
