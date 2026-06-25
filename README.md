# Credit Default Risk Scoring — Home Credit Default Risk Project

## Project Description

This project is an end-to-end machine learning system that predicts the
probability that a loan applicant will default, built on the **Home
Credit Default Risk** dataset (a real-world, multi-table lending dataset
covering ~300,000 applicants across 7 relational tables). The project
covers the complete data science lifecycle — relational database design
and SQL exploration, multi-table merging and aggregation, data cleaning,
exploratory data analysis, feature engineering, statistical and
model-based feature selection, and a deployed LightGBM classifier served
through an interactive Streamlit application that supports both
real-time single-applicant scoring and bulk batch scoring.

The deliverable is not just a trained model — it is a working decision
-support tool intended for use by loan officers, risk analysts, and
auditors, with built-in explainability and performance reporting.

---

## Table of Contents

1. [Business Problem](#business-problem)
2. [Who This Is For](#who-this-is-for)
3. [Dataset](#dataset)
4. [Project Architecture](#project-architecture)
5. [Methodology Summary](#methodology-summary)
6. [Key Results](#key-results)
7. [Application Features](#application-features)
8. [Design Decisions & Honest Limitations](#design-decisions--honest-limitations)
9. [Project Structure](#project-structure)
10. [Installation & Usage](#installation--usage)
11. [Roadmap / Next Steps](#roadmap--next-steps)
12. [Data Source & Credits](#data-source--credits)

---

## Business Problem

A lending institution must decide, for every applicant, whether to
approve or reject a loan — and that decision carries two opposite
risks: approving an applicant who later defaults (direct financial
loss), or rejecting an applicant who would have repaid reliably (lost
revenue and an unfair outcome for the applicant). Roughly 8% of
applicants in this dataset eventually default, so the prediction task
is a significantly imbalanced binary classification problem where the
cost of the two error types is not symmetric.

The goal of this project is to give the institution a data-driven risk
score — calibrated, explainable, and adjustable via a configurable
decision threshold — rather than relying solely on fixed manual
underwriting rules.

## Who This Is For

| Audience | How they use this project |
|---|---|
| **Loan officers** | Enter an applicant's details into the live scoring form and get an instant risk score with an approve/reject recommendation |
| **Risk analysts** | Upload a CSV of many applicants at once and export scored results for portfolio-level review |
| **Compliance / audit teams** | Review the per-applicant risk-factor breakdown to confirm decisions are explainable and not arbitrary |
| **Management** | Review the model performance dashboard (AUC, Gini, PR-AUC) before approving the model for production use |
| **Data scientists / ML engineers** | Reuse the pipeline (SQL schema, merge/aggregation logic, feature selection methodology) as a template for similar credit-risk or imbalanced-classification problems |
| **Evaluators / instructors** | Assess the full pipeline — data engineering, statistical reasoning, modeling decisions, and deployment — not just a single notebook |

## Dataset

**Home Credit Default Risk** — 7 relational tables joined on `SK_ID_CURR`
(and `SK_ID_PREV` / `SK_ID_BUREAU` for the sub-tables):

- `application_train` / `application_test` — one row per applicant, the core demographic/financial table and the only one containing `TARGET`
- `bureau` + `bureau_balance` — the applicant's credit history at *other* financial institutions
- `previous_application` — the applicant's past loan applications with this institution
- `POS_CASH_balance` — monthly snapshots of point-of-sale and cash loan balances
- `installments_payments` — actual repayment history against scheduled installments
- `credit_card_balance` — monthly credit card balance and utilization history

Class balance: **~92% repaid, ~8% defaulted** — a meaningful imbalance
that shaped several modeling decisions described below.

## Project Architecture

```
Raw relational tables (7 CSVs)
        │
        ▼
SQL schema design + exploratory SQL (joins, CTEs, window functions)
        │
        ▼
Merging & aggregation  →  one row per applicant, hundreds of engineered
                          features (mean/sum/min/max aggregates, recency
                          features, ratio features)
        │
        ▼
Cleaning  →  sentinel-value fixes, near-zero-variance removal,
             high-missingness column removal, categorical encoding,
             memory downcasting
        │
        ▼
Exploratory Data Analysis  →  target imbalance, feature distributions,
                               correlation analysis, missingness-as-signal
        │
        ▼
Feature selection (two stages)
   1. Model-free filter: Mutual Information, KS statistic, Information Value
   2. Model-based recursive elimination: LightGBM importance + held-out AUC
        │
        ▼
Model training & evaluation  →  LightGBM binary classifier, cross-validated
        │
        ▼
Deployment  →  Streamlit application (real-time scoring, batch scoring,
                performance dashboard, explainability, feature importance)
```

## Methodology Summary

- **SQL layer**: a normalized relational schema (`SK_ID_CURR` as the
  shared key) with indexed foreign keys, plus exploratory SQL using
  CTEs and window functions (e.g. `NTILE` for decile risk analysis,
  moving-average smoothing over age bands).
- **Merging**: each sub-table is aggregated to one row per `SK_ID_CURR`
  using a mix of statistical aggregates (`mean`, `sum`, `min`, `max`,
  `nunique`) and purpose-built **recency features** (e.g. "terms of the
  applicant's *most recent* previous application"), since recent
  behavior is more predictive than lifetime averages.
- **Cleaning**: sentinel values (e.g. `365243` representing "no date")
  replaced with `NaN`; near-zero-variance columns and columns with
  >75% missing values dropped; categorical variables one-hot encoded;
  dtypes downcast to reduce memory footprint.
- **EDA**: confirmed `EXT_SOURCE_2` (an external credit bureau score) as
  the single strongest predictor; found that *missingness itself* is
  predictive — applicants missing `EXT_SOURCE_1` default at roughly
  double the rate of applicants who have it, which is why missing
  values were preserved (and flagged) rather than blindly imputed.
- **Feature selection**: started from 806 candidate features.
  Model-free filtering (Mutual Information, KS statistic, Information
  Value) removed the weakest features cheaply; LightGBM-based recursive
  elimination with a held-out validation set then iteratively dropped
  the lowest-importance features. Counter-intuitively, validation AUC
  *improved* as features were removed (0.7848 → 0.7862), confirming
  most of the original 806 features were adding noise rather than
  signal. The final selected set has 204 features.
- **Imbalance handling**: `scale_pos_weight` was used instead of SMOTE.
  SMOTE synthesizes points by interpolating between real samples, which
  is unreliable in a 200+ dimensional space with missing values;
  `scale_pos_weight` achieves the same up-weighting effect natively
  inside LightGBM's loss function without introducing synthetic data
  or cross-validation leakage risk.
- **Model**: LightGBM binary classifier (gradient-boosted trees),
  chosen for its native handling of missing values, strong baseline
  performance on tabular data, and fast training/inference.

## Key Results

| Stage | Metric | Value |
|---|---|---|
| Feature selection (188-feature checkpoint) | Validation AUC | 0.7862 |
| Final selected feature set | Features / AUC | 204 features / 0.7933 |
| Final tuned model | Out-of-fold AUC | 0.7920 |
| Final tuned model | Gini coefficient | 0.5839 |
| Final tuned model | Precision-Recall AUC | 0.2872 |

The lower PR-AUC relative to ROC-AUC is expected and worth stating
explicitly: with only ~8% positive cases, precision-recall is a harder
metric than ROC-AUC by construction, and is reported here deliberately
rather than omitted, since ROC-AUC alone can overstate performance on
imbalanced problems.

## Application Features

The Streamlit application (`app.py`) is organized into four tabs:

1. **Real-time Application Results** — a sidebar form lets a loan
   officer enter an applicant's core details (income, requested credit,
   annuity, goods price, age, employment length, external credit
   scores). On submission, the app displays a gauge chart of the
   predicted default probability, an approve/reject recommendation
   against a configurable risk threshold, and the top 3 factors driving
   that specific applicant's score.
2. **Model Performance Dashboard** — headline validation metrics
   (OOF AUC, Gini, PR-AUC) displayed as metric cards, alongside an ROC
   curve.
3. **Global Feature Importance** — the top 10 most influential features
   model-wide, with a plain-language glossary translating feature-family
   names (`EXT_SOURCE`, `BUREAU`, `PREV`, etc.) into stakeholder-friendly
   explanations.
4. **Bulk Batch Processing** — upload a CSV of many applicants, score
   them all at once, and download the results (probability, predicted
   class, and recommendation per applicant) as a CSV.

## Design Decisions & Honest Limitations

A few deliberate engineering choices are worth stating explicitly
rather than leaving implicit:

- **Live single-applicant scoring uses default values for features the
  form doesn't collect** (the ~190 historical/bureau/previous-application
  aggregates a brand-new applicant has no history for). These are not
  fabricated "average customer" guesses — they follow a stated policy:
  count/sum-type aggregates default to `0` (a new applicant genuinely
  has zero prior records, which is a true value, not an estimate), while
  mean/ratio-type aggregates over non-existent history are left as
  missing, since LightGBM handles missing values natively at inference.
- **The ROC curve in the Performance Dashboard is a representative
  curve shape mathematically anchored to the reported OOF AUC**, not a
  curve recomputed live from a held-out test set (the deployed app does
  not have access to held-out validation predictions at runtime). This
  is disclosed in the app itself rather than presented as a live metric.
- **The "top risk factors" shown per applicant use an auditable
  rule-based heuristic** (ratios weighted by global feature importance),
  not true Shapley values. This was a deliberate trade-off for speed and
  simplicity in the live-scoring path; true SHAP-based local
  explanations are a natural next enhancement (see Roadmap).
- **The model file shipped for local testing may be a placeholder**
  trained on synthetic data with the correct feature names, intended
  only to verify the application runs end-to-end before the real
  trained model is in place. It must be replaced before any real
  scoring decision is made.

## Project Structure

```
project/
├── app.py                      # Streamlit application (single-file, self-contained)
├── requirements.txt
├── README.md
└── model/
    ├── final_model.pkl         # Trained LightGBM model (joblib-serialized)
    └── selected_features.json  # The 204 selected features + training metadata
```

## Installation & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure model/final_model.pkl and model/selected_features.json
#    are present (see "Design Decisions" above if using a placeholder model)

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

**requirements.txt** should include at minimum:
```
streamlit
pandas
numpy
joblib
plotly
lightgbm
scikit-learn
```
(`lightgbm` is required even though `app.py` does not import it directly —
`joblib.load()` needs it installed to deserialize a saved LightGBM model.)

## Roadmap / Next Steps

- Replace the rule-based per-applicant risk factors with true SHAP
  (Shapley value) explanations for exact, theoretically grounded local
  attributions.
- Add a curated "Exploratory Insights" tab presenting a small, focused
  set of EDA findings (e.g. the EXT_SOURCE_2 separation, age-vs-default
  trend, credit-to-income risk gradient) computed once from the real
  training data and shipped as a lightweight precomputed summary,
  rather than requiring the full training set in the deployed app.
- Hyperparameter tuning via Optuna's Bayesian search to push beyond the
  current baseline.
- Model card and fairness/bias review (e.g. demographic parity checks
  across gender and age bands) before any production lending use.

## Data Source & Credits

Dataset: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
(Kaggle competition, Home Credit Group). This project is for educational
and portfolio purposes; it is not an approved or audited production
credit-decisioning system.

## Acknowledgments

This project was developed as part of the Epsilon AI Program.
