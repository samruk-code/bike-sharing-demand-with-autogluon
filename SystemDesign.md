# Requirements: Predict Bike Sharing Demand with AutoGluon

This document defines the requirements and design decisions for the bike-sharing demand prediction project, built for the [Kaggle Bike Sharing Demand competition](https://www.kaggle.com/c/bike-sharing-demand) as part of the Udacity Machine Learning Engineer Nanodegree.

> **Scope note:** This is a competition/notebook project with **offline batch inference** (CSV submissions to Kaggle). There is no deployed real-time serving endpoint, so the Serving & Inference section covers the batch prediction pipeline only, and Monitoring & Iteration covers leaderboard-driven experiment tracking rather than production model monitoring.

---

## 1. Clarify Requirements & Constraints

### Business Context
Bike-sharing companies need accurate hourly demand forecasts to allocate bikes across stations efficiently. Under-supply loses revenue and frustrates riders; over-supply wastes rebalancing effort.

### Functional Requirements
- Predict the total number of bike rentals (`count`) for each hour in the Kaggle test set.
- Produce a valid Kaggle submission file: `datetime, count` with **no negative values** (Kaggle rejects them; predictions must be clipped to ≥ 0).
- Demonstrate measurable, iterative improvement across at least three training runs (baseline → feature engineering → hyperparameter tuning).

### Constraints
- **Data:** Only the competition dataset may be used — hourly rentals with weather and calendar metadata. The `casual` and `registered` columns exist only in the training set and must be dropped to avoid leakage.
- **Compute/time:** Training is capped at a **10-minute time limit** per AutoGluon run (`time_limit=600`), suitable for a single workstation or SageMaker notebook instance.
- **Tooling:** AutoGluon `TabularPredictor` as the AutoML framework; Kaggle API for data download and submission.
- **Evaluation gate:** Success is measured by the public Kaggle leaderboard score (RMSLE-style RMSE on log counts); each iteration should not regress the score.

---

## 2. Define the ML Problem Formally

- **Task type:** Supervised **regression** on tabular data.
- **Input (X):** Hourly records with weather features (`temp`, `atemp`, `humidity`, `windspeed`, `weather`), calendar features (`season`, `holiday`, `workingday`), and a `datetime` timestamp.
- **Target (y):** `count` — total rentals in that hour (non-negative integer).
- **Prediction function:** `f(weather, calendar, time features) → count ≥ 0`.
- **Evaluation metric:** **Root Mean Squared Error (RMSE)** as scored by Kaggle (computed on log-transformed counts, i.e. RMSLE). AutoGluon is configured with `eval_metric="root_mean_squared_error"`.
- **Leakage rule:** `casual` and `registered` sum to `count` and are absent from the test set — they must be excluded from training features.
- **Baseline & targets:**
  | Run | Approach | Kaggle RMSE |
  |-----|----------|-------------|
  | 1 | Raw features (baseline) | 1.400 |
  | 2 | + time features & categorical dtypes | 0.486 |
  | 3 | + hyperparameter tuning | 0.484 |

---

## 3. Data Pipeline & Feature Engineering

### Data Acquisition
- Download `train.csv`, `test.csv`, and `sampleSubmission.csv` via the **Kaggle API**.

### Exploratory Data Analysis
- Plot per-feature histograms to inspect distributions.
- Build a correlation matrix; EDA showed `hour` correlates 0.40 with demand, with peaks at morning/evening commute hours.

### Preprocessing
- Parse `datetime` as a timestamp on load.
- Drop `casual` and `registered` from the training set (leakage prevention).

### Feature Engineering
- Extract from `datetime`:
  - `hour` — hour of day (strongest single new signal),
  - `day` — day of month,
  - `month` — month of year.
- Cast `season` and `weather` to `category` dtype so AutoGluon treats them as categorical rather than ordinal numeric values.
- Apply identical transformations to train and test sets to keep schemas consistent.

### Future Feature Work (identified, not implemented)
- Lag features and rolling averages of demand.
- Weather × time interaction terms.

---

## 4. Model Selection & Training Strategy

### Framework
AutoGluon `TabularPredictor` — trains a portfolio of models (LightGBM, XGBoost, Random Forest, CatBoost, neural networks, KNN) and stacks them into weighted ensembles.

### Training Configuration
- Preset: `best_quality` (enables multi-layer stacking/bagging).
- `time_limit = 600` seconds per run.
- `eval_metric = root_mean_squared_error`.
- Validation handled internally by AutoGluon (bagged out-of-fold estimates); model ranking inspected via `predictor.fit_summary()` / leaderboard. Top model in the initial run: **WeightedEnsemble_L3**.

### Iteration Strategy
1. **Baseline** on raw features to establish a reference score.
2. **Feature engineering** run with the same training budget — delivered the dominant gain (RMSE 1.400 → 0.486, −65%).
3. **Hyperparameter optimization** — manually specified search values passed to AutoGluon for three model families:

   | Model | Hyperparameter | Value |
   |-------|---------------|-------|
   | GBM (LightGBM) | `max_depth` / `n_estimators` / `learning_rate` | 8 / 80 / 0.01 |
   | XGBoost | `max_depth` / `n_estimators` / `learning_rate` | 10 / 110 / 0.01 |
   | Random Forest | `max_depth` / `n_estimators` | 9 / 95 |

   Result: marginal gain (0.486 → 0.484), confirming AutoGluon's defaults are already well-calibrated. **Conclusion: prioritize feature work over further tuning.**

---

## 5. Serving & Inference Architecture (Batch)

This project uses **offline batch inference** — there is no online endpoint.

- **Flow:** load test set → apply the same feature transformations as training → `predictor.predict(test)` → post-process → write submission CSV.
- **Post-processing:** clip negative predictions to **0** (required for a valid Kaggle submission and physically meaningful counts).
- **Output artifacts:** one submission file per experiment for traceability:
  - `submission.csv` — baseline,
  - `submission_new_features.csv` — feature-engineered model,
  - `submission_new_hpo.csv` — tuned model.
- **Delivery:** submissions uploaded via the Kaggle API with a descriptive message per run.

---

## 6. Monitoring & Iteration

Monitoring in this project means **experiment tracking against the Kaggle leaderboard**, not production telemetry.

- **Score tracking:** record the Kaggle RMSE for every submission and compare it to the internal (training) score to detect overfitting or validation/leaderboard divergence.
- **Visualization:** line plots of top model score per training run (`model_train_score.png`) and Kaggle score per run (`model_test_score.png`).
- **Iteration loop:** hypothesis (EDA insight) → change (features or hyperparameters) → retrain under the same time budget → submit → compare leaderboard score → keep or revert.
- **Documented learnings feeding the next iteration:**
  - Feature engineering yields far larger gains than hyperparameter tuning for this dataset.
  - Next candidates: lag/rolling demand features and weather interaction terms.
- **Reporting:** results, model rankings, and score progression are documented in `README.md` and the project report (`report-template.md`).
