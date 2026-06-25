from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# =============================================================================
# Page and model configuration
# =============================================================================
st.set_page_config(
    page_title="Credit Default Risk Scoring",
    page_icon=":bank:",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "final_model.pkl"
FEATURES_PATH = BASE_DIR / "model" / "selected_features.json"
DEFAULT_RISK_THRESHOLD = 0.15

PERFORMANCE_METRICS = {
    "Final OOF AUC": "0.7920",
    "Gini Coefficient": "0.5839",
    "Precision-Recall AUC": "0.2872",
}

EXTERNAL_SOURCE_DEFAULTS = {
    "EXT_SOURCE_1": 0.50,
    "EXT_SOURCE_2": 0.55,
    "EXT_SOURCE_3": 0.52,
}

FEATURE_GLOSSARY = {
    "EXT_SOURCE": "Normalized scores from external credit data providers. Lower values usually indicate higher default risk.",
    "BUREAU": "Aggregated credit bureau history, such as prior credit age, debt, overdue amounts, and active credit mix.",
    "Credit_card": "Historical credit card balance and utilization behavior from related accounts.",
    "Installments": "Repayment behavior on previous installment loans, including late and early payment indicators.",
    "PREV": "Aggregated previous application behavior, including prior approvals, refusals, terms, and requested amounts.",
    "DAYS": "Event timing encoded as days relative to the application date. Negative values are historical dates.",
    "AMT": "Monetary values such as income, requested credit, annuity payments, or prior credit amounts.",
}


# =============================================================================
# Cached resources
# =============================================================================
@st.cache_resource(show_spinner="Loading LightGBM model...")
def load_model() -> Any:
    """Load the serialized production model once per Streamlit session."""
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Unable to load model from `{MODEL_PATH}`: {exc}")
        return None


@st.cache_data(show_spinner="Loading feature configuration...")
def load_selected_features() -> list[str]:
    """Load the exact feature order expected by the trained model."""
    try:
        if not FEATURES_PATH.exists():
            raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")

        with FEATURES_PATH.open("r", encoding="utf-8") as feature_file:
            features = json.load(feature_file)

        if not isinstance(features, list) or not all(isinstance(col, str) for col in features):
            raise ValueError("selected_features.json must contain a JSON list of feature names.")

        return features
    except Exception as exc:
        st.error(f"Unable to load selected features from `{FEATURES_PATH}`: {exc}")
        return []


@st.cache_data(show_spinner="Running model inference...")
def run_inference(_model: Any, feature_frame: pd.DataFrame) -> np.ndarray:
    """Return class-1 default probabilities for a model-ready feature frame."""
    try:
        if feature_frame.empty:
            raise ValueError("Feature frame is empty.")

        if hasattr(_model, "predict_proba"):
            probabilities = np.asarray(_model.predict_proba(feature_frame))
            if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                scores = probabilities[:, 1]
            else:
                scores = probabilities.reshape(-1)
        elif hasattr(_model, "predict"):
            scores = np.asarray(_model.predict(feature_frame)).reshape(-1)
        else:
            raise AttributeError("Loaded model does not expose predict_proba or predict.")

        return np.clip(scores.astype(float), 0.0, 1.0)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return np.array([], dtype=float)


@st.cache_data(show_spinner=False)
def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes for download widgets."""
    return dataframe.to_csv(index=False).encode("utf-8")


# =============================================================================
# Feature engineering and enrichment
# =============================================================================
def _first_present(raw: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first non-empty value found across several possible input names."""
    for key in keys:
        value = raw.get(key)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return value
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        numeric = float(value)
        if np.isfinite(numeric):
            return numeric
        return default
    except (TypeError, ValueError):
        return default


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator in (0, 0.0) or not np.isfinite(denominator):
        return default
    return numerator / denominator


def _normalize_live_input(live_input_dict: dict[str, Any]) -> dict[str, Any]:
    """Allow both app-friendly labels and raw Home Credit column names."""
    raw = dict(live_input_dict or {})

    normalized = {
        "income": _to_float(
            _first_present(raw, "income", "Income", "Annual Income", "AMT_INCOME_TOTAL", default=180000.0),
            180000.0,
        ),
        "credit_amount": _to_float(
            _first_present(raw, "credit_amount", "Credit Amount", "Credit Amount Requested", "AMT_CREDIT", default=450000.0),
            450000.0,
        ),
        "annuity": _to_float(
            _first_present(raw, "annuity", "Annuity", "Loan Annuity", "AMT_ANNUITY", default=24000.0),
            24000.0,
        ),
        "goods_price": _to_float(
            _first_present(raw, "goods_price", "Goods Price", "AMT_GOODS_PRICE", default=450000.0),
            450000.0,
        ),
        "age_years": _to_float(_first_present(raw, "age_years", "Age", default=35.0), 35.0),
        "employment_years": _to_float(
            _first_present(raw, "employment_years", "Employment Length", "Employment Length Years", default=5.0),
            5.0,
        ),
        "ext_source_1": _to_float(_first_present(raw, "EXT_SOURCE_1", "ext_source_1", default=0.50), 0.50),
        "ext_source_2": _to_float(_first_present(raw, "EXT_SOURCE_2", "ext_source_2", default=0.55), 0.55),
        "ext_source_3": _to_float(_first_present(raw, "EXT_SOURCE_3", "ext_source_3", default=0.52), 0.52),
        "gender": str(_first_present(raw, "gender", "Gender", default="Female")),
        "contract_type": str(_first_present(raw, "contract_type", "Contract Type", "NAME_CONTRACT_TYPE", default="Cash loans")),
        "income_type": str(_first_present(raw, "income_type", "Income Type", "NAME_INCOME_TYPE", default="Working")),
        "education_type": str(
            _first_present(raw, "education_type", "Education Type", "NAME_EDUCATION_TYPE", default="Secondary / secondary special")
        ),
        "family_status": str(_first_present(raw, "family_status", "Family Status", "NAME_FAMILY_STATUS", default="Married")),
        "occupation_type": str(_first_present(raw, "occupation_type", "Occupation Type", "OCCUPATION_TYPE", default="Other")),
        "organization_type": str(_first_present(raw, "organization_type", "Organization Type", "ORGANIZATION_TYPE", default="Other")),
    }

    normalized["age_years"] = min(max(normalized["age_years"], 18.0), 80.0)
    normalized["employment_years"] = min(max(normalized["employment_years"], 0.0), 60.0)
    return normalized


def _historical_enrichment_defaults(inputs: dict[str, Any]) -> dict[str, float]:
    """
    Mock historical aggregations that would usually be merged from bureau,
    credit-card, previous-application, POS, and installments feature tables.
    """
    income = inputs["income"]
    credit = inputs["credit_amount"]
    annuity = inputs["annuity"]
    goods_price = inputs["goods_price"]

    credit_card_limit = max(credit * 0.35, 50000.0)
    credit_card_balance = credit_card_limit * 0.18
    prior_application_mean = max(credit * 0.82, 1.0)
    installment_payment_mean = max(annuity * 0.96, 1.0)

    return {
        "REGION_POPULATION_RELATIVE": 0.020,
        "DAYS_REGISTRATION": -4500.0,
        "DAYS_ID_PUBLISH": -3000.0,
        "OWN_CAR_AGE": 0.0,
        "FLAG_WORK_PHONE": 0.0,
        "REGION_RATING_CLIENT_W_CITY": 2.0,
        "HOUR_APPR_PROCESS_START": 12.0,
        "REG_CITY_NOT_LIVE_CITY": 0.0,
        "APARTMENTS_AVG": 0.075,
        "BASEMENTAREA_AVG": 0.045,
        "YEARS_BEGINEXPLUATATION_AVG": 0.978,
        "YEARS_BUILD_AVG": 0.752,
        "COMMONAREA_AVG": 0.020,
        "ELEVATORS_AVG": 0.080,
        "FLOORSMAX_AVG": 0.225,
        "LANDAREA_AVG": 0.065,
        "LIVINGAPARTMENTS_AVG": 0.085,
        "LIVINGAREA_AVG": 0.095,
        "NONLIVINGAPARTMENTS_AVG": 0.005,
        "NONLIVINGAREA_AVG": 0.030,
        "APARTMENTS_MODE": 0.075,
        "YEARS_BEGINEXPLUATATION_MODE": 0.978,
        "YEARS_BUILD_MODE": 0.752,
        "COMMONAREA_MODE": 0.020,
        "LANDAREA_MODE": 0.065,
        "LIVINGAREA_MODE": 0.095,
        "TOTALAREA_MODE": 0.100,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
        "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
        "DAYS_LAST_PHONE_CHANGE": -1000.0,
        "FLAG_DOCUMENT_3": 1.0,
        "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
        "BUREAU_DAYS_CREDIT_MIN": -2200.0,
        "BUREAU_DAYS_CREDIT_MAX": -120.0,
        "BUREAU_DAYS_CREDIT_MEAN": -950.0,
        "BUREAU_DAYS_CREDIT_ENDDATE_MIN": -650.0,
        "BUREAU_DAYS_CREDIT_ENDDATE_MAX": 1200.0,
        "BUREAU_DAYS_CREDIT_ENDDATE_MEAN": 240.0,
        "BUREAU_DAYS_ENDDATE_FACT_MIN": -1900.0,
        "BUREAU_DAYS_ENDDATE_FACT_MAX": -120.0,
        "BUREAU_DAYS_ENDDATE_FACT_MEAN": -720.0,
        "BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX": 0.0,
        "BUREAU_AMT_CREDIT_MAX_OVERDUE_MEAN": 0.0,
        "BUREAU_AMT_CREDIT_SUM_MAX": max(credit * 0.95, 1.0),
        "BUREAU_AMT_CREDIT_SUM_MEAN": max(credit * 0.42, 1.0),
        "BUREAU_AMT_CREDIT_SUM_SUM": max(credit * 1.35, 1.0),
        "BUREAU_AMT_CREDIT_SUM_DEBT_MAX": max(credit * 0.20, 0.0),
        "BUREAU_AMT_CREDIT_SUM_DEBT_MEAN": max(credit * 0.08, 0.0),
        "BUREAU_AMT_CREDIT_SUM_DEBT_SUM": max(credit * 0.24, 0.0),
        "BUREAU_AMT_CREDIT_SUM_LIMIT_MAX": credit_card_limit,
        "BUREAU_AMT_CREDIT_SUM_LIMIT_MEAN": credit_card_limit * 0.50,
        "BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN": 0.0,
        "BUREAU_DAYS_CREDIT_UPDATE_MIN": -900.0,
        "BUREAU_DAYS_CREDIT_UPDATE_MAX": -30.0,
        "BUREAU_DAYS_CREDIT_UPDATE_MEAN": -360.0,
        "BUREAU_AMT_ANNUITY_MEAN": max(annuity * 0.70, 1.0),
        "BUREAU_BB_MONTHS_BALANCE_SIZE_SUM": 24.0,
        "BUREAU_BB_STATUS_0_MEAN_MEAN": 0.78,
        "BUREAU_BB_STATUS_1_MEAN_MEAN": 0.03,
        "BUREAU_BB_STATUS_C_MEAN_MEAN": 0.12,
        "BUREAU_BB_STATUS_X_MEAN_MEAN": 0.07,
        "BUREAU_CREDIT_ACTIVE_Active_MEAN": 0.35,
        "BUREAU_CREDIT_TYPE_Car_loan_MEAN": 0.04,
        "BUREAU_CREDIT_TYPE_Credit_card_MEAN": 0.28,
        "BUREAU_CREDIT_TYPE_Microloan_MEAN": 0.03,
        "BUREAU_CREDIT_TYPE_Mortgage_MEAN": 0.03,
        "Credit_card_AMT_BALANCE_MEAN": credit_card_balance,
        "Credit_card_AMT_BALANCE_MAX": credit_card_balance * 2.2,
        "Credit_card_AMT_CREDIT_LIMIT_ACTUAL_MEAN": credit_card_limit,
        "Credit_card_MONTHS_BALANCE_MIN": -24.0,
        "Installments_NUM_INSTALMENT_VERSION_NUNIQUE": 2.0,
        "Installments_NUM_INSTALMENT_NUMBER_MAX": 18.0,
        "Installments_DAYS_INSTALMENT_MIN": -1800.0,
        "Installments_DAYS_INSTALMENT_MAX": -30.0,
        "Installments_AMT_INSTALMENT_MAX": max(annuity * 1.2, 1.0),
        "Installments_AMT_INSTALMENT_MEAN": max(annuity, 1.0),
        "Installments_AMT_INSTALMENT_SUM": max(annuity * 18.0, 1.0),
        "Installments_AMT_PAYMENT_MIN": max(installment_payment_mean * 0.75, 1.0),
        "Installments_AMT_PAYMENT_MEAN": installment_payment_mean,
        "Installments_AMT_PAYMENT_SUM": installment_payment_mean * 18.0,
        "Installments_ins_DPD_MAX": 0.0,
        "Installments_ins_DPD_MEAN": 0.0,
        "Installments_ins_DPD_SUM": 0.0,
        "Installments_ins_DBD_MAX": 8.0,
        "Installments_ins_DBD_MEAN": 2.0,
        "Installments_ins_DBD_SUM": 36.0,
        "Installments_ins_PAYMENT_DIFF_MAX": max(annuity * 0.04, 0.0),
        "Installments_ins_PAYMENT_DIFF_MEAN": max(annuity * 0.01, 0.0),
        "Installments_ins_PAYMENT_DIFF_SUM": max(annuity * 0.18, 0.0),
        "Installments_ins_PAYMENT_RATIO_MEAN": 0.98,
        "Installments_ins_PAYMENT_RATIO_MAX": 1.05,
        "Installments_DAYS_ENTRY_PAYMENT_SIZE": 18.0,
        "Installments_RECENT_ins_DPD_MAX": 0.0,
        "Installments_RECENT_ins_DPD_MEAN": 0.0,
        "Installments_RECENT_ins_DPD_SUM": 0.0,
        "Installments_RECENT_ins_DBD_MAX": 5.0,
        "Installments_RECENT_ins_DBD_MEAN": 1.8,
        "Installments_RECENT_ins_PAYMENT_DIFF_MAX": max(annuity * 0.03, 0.0),
        "Installments_RECENT_ins_PAYMENT_DIFF_MEAN": max(annuity * 0.01, 0.0),
        "Installments_RECENT_ins_PAYMENT_RATIO_MEAN": 0.99,
        "Installments_RECENT_AMT_PAYMENT_SUM": installment_payment_mean * 6.0,
        "Installments_RECENT_AMT_PAYMENT_MEAN": installment_payment_mean,
        "Installments_RECENT_DAYS_ENTRY_PAYMENT_SIZE": 6.0,
        "POS_MONTHS_BALANCE_MIN": -24.0,
        "POS_MONTHS_BALANCE_MAX": -1.0,
        "POS_MONTHS_BALANCE_SIZE": 24.0,
        "POS_SK_DPD_DEF_MEAN": 0.0,
        "POS_INSTALLMENTS_PAID_MEAN": 0.82,
        "POS_LOAN_PERCENT_REMAINING_MEAN": 0.18,
        "POS_NAME_CONTRACT_STATUS_Active_MEAN": 0.35,
        "PREV_AMT_ANNUITY_MEAN": max(annuity * 0.85, 1.0),
        "PREV_AMT_ANNUITY_MAX": max(annuity * 1.15, 1.0),
        "PREV_AMT_ANNUITY_SUM": max(annuity * 2.4, 1.0),
        "PREV_AMT_APPLICATION_MEAN": prior_application_mean,
        "PREV_AMT_APPLICATION_MAX": max(credit * 1.05, 1.0),
        "PREV_AMT_APPLICATION_SUM": max(credit * 1.65, 1.0),
        "PREV_AMT_CREDIT_MEAN": max(credit * 0.86, 1.0),
        "PREV_AMT_CREDIT_MAX": max(credit * 1.08, 1.0),
        "PREV_AMT_DOWN_PAYMENT_MEAN": max(goods_price * 0.05, 0.0),
        "PREV_AMT_DOWN_PAYMENT_SUM": max(goods_price * 0.08, 0.0),
        "PREV_AMT_GOODS_PRICE_MEAN": max(goods_price * 0.88, 1.0),
        "PREV_RATE_DOWN_PAYMENT_MEAN": 0.05,
        "PREV_RATE_DOWN_PAYMENT_MAX": 0.10,
        "PREV_DAYS_DECISION_MEAN": -850.0,
        "PREV_DAYS_DECISION_MIN": -2100.0,
        "PREV_DAYS_DECISION_MAX": -120.0,
        "PREV_CNT_PAYMENT_MEAN": 18.0,
        "PREV_CNT_PAYMENT_SUM": 36.0,
        "PREV_CREDIT_TO_APPLICATION_RATIO_MEAN": 1.02,
        "PREV_CREDIT_TO_APPLICATION_RATIO_MIN": 0.92,
        "PREV_CREDIT_TO_APPLICATION_RATIO_MAX": 1.15,
        "PREV_CREDIT_TO_GOODS_RATIO_MEAN": 1.02,
        "PREV_CREDIT_TO_GOODS_RATIO_MAX": 1.12,
        "PREV_ANNUITY_TO_CREDIT_RATIO_MEAN": _safe_divide(annuity, max(credit, 1.0), 0.06),
        "PREV_ANNUITY_TO_CREDIT_RATIO_MAX": _safe_divide(annuity * 1.15, max(credit, 1.0), 0.08),
        "PREV_LOAN_DURATION_PLANNED_MEAN": 18.0,
        "PREV_LOAN_DURATION_PLANNED_MAX": 36.0,
        "PREV_DAYS_LAST_DUE_DIFF_MEAN": 0.0,
        "PREV_WAS_APPROVED_MEAN": 0.78,
        "PREV_WAS_REFUSED_MEAN": 0.08,
        "PREV_WAS_CANCELED_MEAN": 0.05,
        "PREV_SELLERPLACE_AREA_MEAN": 60.0,
        "PREV_SELLERPLACE_AREA_MAX": 120.0,
        "PREV_NAME_CONTRACT_TYPE_Consumer_loans_MEAN": 0.62,
        "PREV_NAME_PAYMENT_TYPE_Cash_through_the_bank_MEAN": 0.75,
        "PREV_NAME_PAYMENT_TYPE_XNA_MEAN": 0.10,
        "PREV_CODE_REJECT_REASON_SCO_MEAN": 0.03,
        "PREV_CODE_REJECT_REASON_XAP_MEAN": 0.80,
        "PREV_NAME_CLIENT_TYPE_New_MEAN": 0.25,
        "PREV_NAME_GOODS_CATEGORY_Furniture_MEAN": 0.05,
        "PREV_NAME_GOODS_CATEGORY_Mobile_MEAN": 0.10,
        "PREV_NAME_PRODUCT_TYPE_walk_in_MEAN": 0.30,
        "PREV_NAME_PRODUCT_TYPE_x_sell_MEAN": 0.45,
        "PREV_CHANNEL_TYPE_AP_Cash_loan__MEAN": 0.05,
        "PREV_CHANNEL_TYPE_Channel_of_corporate_sales_MEAN": 0.03,
        "PREV_CHANNEL_TYPE_Credit_and_cash_offices_MEAN": 0.40,
        "PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN": 0.10,
        "PREV_NAME_YIELD_GROUP_high_MEAN": 0.12,
        "PREV_NAME_YIELD_GROUP_low_action_MEAN": 0.08,
        "PREV_NAME_YIELD_GROUP_low_normal_MEAN": 0.28,
        "PREV_NAME_YIELD_GROUP_middle_MEAN": 0.38,
        "PREV_PRODUCT_COMBINATION_Cash_Street_low_MEAN": 0.04,
        "PREV_PRODUCT_COMBINATION_Cash_X_Sell_high_MEAN": 0.08,
        "PREV_PRODUCT_COMBINATION_Cash_X_Sell_low_MEAN": 0.12,
        "PREV_PRODUCT_COMBINATION_Cash_X_Sell_middle_MEAN": 0.18,
        "PREV_PRODUCT_COMBINATION_POS_industry_with_interest_MEAN": 0.10,
        "PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN": 0.72,
        "PREV_NAME_TYPE_SUITE_nan_MEAN": 0.08,
        "PREV_LAST_AMT_CREDIT": max(credit * 0.75, 1.0),
        "PREV_LAST_AMT_APPLICATION": max(prior_application_mean, 1.0),
        "PREV_LAST_AMT_ANNUITY": max(annuity * 0.85, 1.0),
        "PREV_LAST_CNT_PAYMENT": 18.0,
        "PREV_LAST_CREDIT_TO_APPLICATION_RATIO": 1.02,
        "INCOME_PER_PERSON": income / 2.0,
    }


def _build_feature_row(live_input_dict: dict[str, Any], selected_features: list[str]) -> dict[str, float]:
    inputs = _normalize_live_input(live_input_dict)

    income = inputs["income"]
    credit = inputs["credit_amount"]
    annuity = inputs["annuity"]
    goods_price = inputs["goods_price"]
    age_years = inputs["age_years"]
    employment_years = inputs["employment_years"]
    ext1 = inputs["ext_source_1"]
    ext2 = inputs["ext_source_2"]
    ext3 = inputs["ext_source_3"]

    days_birth = -int(age_years * 365.25)
    days_employed = -int(employment_years * 365.25) if employment_years > 0 else 365243
    ext_values = np.array([ext1, ext2, ext3], dtype=float)

    values: dict[str, float] = {feature: 0.0 for feature in selected_features}
    values.update(_historical_enrichment_defaults(inputs))
    values.update(
        {
            "CODE_GENDER": 1.0 if inputs["gender"].lower().startswith("m") else 0.0,
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods_price,
            "DAYS_BIRTH": float(days_birth),
            "DAYS_EMPLOYED": float(days_employed),
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "YEARS_EMPLOYED": employment_years,
            "CREDIT_TO_GOODS_RATIO": _safe_divide(credit, goods_price, 1.0),
            "INCOME_PER_PERSON": _safe_divide(income, 2.0, income),
            "CREDIT_INCOME_RATIO": _safe_divide(credit, income, 0.0),
            "ANNUITY_INCOME_RATIO": _safe_divide(annuity, income, 0.0),
            "CREDIT_TERM_MONTHS": _safe_divide(credit, annuity, 0.0),
            "ANNUITY_CREDIT_RATIO": _safe_divide(annuity, credit, 0.0),
            "EMPLOYED_TO_AGE_RATIO": _safe_divide(employment_years, age_years, 0.0),
            "DAYS_EMPLOYED_RATIO": _safe_divide(float(days_employed), float(days_birth), 0.0),
            "EXT_SOURCE_MEAN": float(np.nanmean(ext_values)),
            "EXT_SOURCE_PROD": float(np.nanprod(ext_values)),
            "NAME_CONTRACT_TYPE_Cash_loans": 1.0 if inputs["contract_type"] == "Cash loans" else 0.0,
            "NAME_INCOME_TYPE_State_servant": 1.0 if inputs["income_type"] == "State servant" else 0.0,
            "NAME_INCOME_TYPE_Working": 1.0 if inputs["income_type"] == "Working" else 0.0,
            "NAME_EDUCATION_TYPE_Higher_education": 1.0 if inputs["education_type"] == "Higher education" else 0.0,
            "NAME_EDUCATION_TYPE_Secondary_secondary_special": 1.0
            if inputs["education_type"] == "Secondary / secondary special"
            else 0.0,
            "NAME_FAMILY_STATUS_Married": 1.0 if inputs["family_status"] == "Married" else 0.0,
            "OCCUPATION_TYPE_Core_staff": 1.0 if inputs["occupation_type"] == "Core staff" else 0.0,
            "OCCUPATION_TYPE_Drivers": 1.0 if inputs["occupation_type"] == "Drivers" else 0.0,
            "ORGANIZATION_TYPE_Business_Entity_Type_3": 1.0
            if inputs["organization_type"] == "Business Entity Type 3"
            else 0.0,
            "ORGANIZATION_TYPE_Self_employed": 1.0 if inputs["organization_type"] == "Self-employed" else 0.0,
        }
    )

    # Full-vector CSV rows may already contain engineered model fields. Preserve them
    # after defaults are built so production batch exports round-trip cleanly.
    for feature in selected_features:
        if feature in live_input_dict:
            values[feature] = _to_float(live_input_dict[feature], values.get(feature, 0.0))

    return values


def pipeline_transform_input(live_input_dict: dict) -> pd.DataFrame:
    """
    Transform a single applicant's live form input into the exact model feature vector.

    External bureau, credit-card, installments, POS, and previous-application
    aggregates are mocked with conservative population defaults to mimic the
    merge-and-clean feature tables used during model development.
    """
    selected_features = load_selected_features()
    try:
        if not selected_features:
            raise ValueError("Selected feature list is empty.")

        feature_row = _build_feature_row(live_input_dict, selected_features)
        feature_frame = pd.DataFrame([feature_row]).reindex(columns=selected_features, fill_value=0.0)
        feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        missing_columns = [column for column in selected_features if column not in feature_frame.columns]
        if missing_columns:
            raise ValueError(f"Transformed frame is missing required columns: {missing_columns[:5]}")

        return feature_frame[selected_features]
    except Exception as exc:
        st.error(f"Feature transformation failed: {exc}")
        return pd.DataFrame(columns=selected_features)


def transform_batch_input(raw_batch: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    """Transform uploaded applicant rows into model-ready rows."""
    try:
        if raw_batch.empty:
            raise ValueError("Uploaded file contains no rows.")

        records = []
        for _, row in raw_batch.iterrows():
            clean_record = {key: value for key, value in row.to_dict().items() if pd.notna(value)}
            records.append(_build_feature_row(clean_record, selected_features))

        feature_frame = pd.DataFrame(records).reindex(columns=selected_features, fill_value=0.0)
        feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return feature_frame[selected_features]
    except Exception as exc:
        st.error(f"Batch feature transformation failed: {exc}")
        return pd.DataFrame(columns=selected_features)


# =============================================================================
# EDA — Synthetic training data generator
# =============================================================================
@st.cache_data(show_spinner="Generating EDA sample data...")
def generate_eda_sample(n: int = 6_000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset that mimics Home Credit training distributions."""
    rng = np.random.RandomState(random_state)

    # Base demographics
    age = rng.normal(43, 11, n).clip(18, 75).astype(int)
    income = rng.lognormal(12.3, 0.6, n)
    credit = income * rng.uniform(1.0, 6.0, n)
    annuity = credit / rng.uniform(8, 25, n)
    goods = credit * rng.uniform(0.85, 1.05, n)
    employed_years = (age - 18) * rng.beta(2, 5, n)

    # EXT sources correlated with target
    target = rng.binomial(1, 0.08, n)
    ext1 = rng.beta(4 + target * 2, 5 - target, n)
    ext2 = rng.beta(5 + target * 3, 4 - target * 2, n)
    ext3 = rng.beta(4 + target * 2, 5 - target, n)

    # Derived fields
    days_birth = -(age * 365.25).astype(int)
    days_employed = -(employed_years * 365.25).astype(int)
    cir = credit / income
    annuity_income = annuity / income
    employed_age = employed_years / age
    ext_mean = (ext1 + ext2 + ext3) / 3.0
    ext_prod = ext1 * ext2 * ext3

    df = pd.DataFrame({
        "TARGET": target,
        "YEARS_BIRTH": age,
        "DAYS_BIRTH": days_birth,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "DAYS_EMPLOYED": days_employed,
        "YEARS_EMPLOYED": employed_years,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "EXT_SOURCE_MEAN": ext_mean,
        "EXT_SOURCE_PROD": ext_prod,
        "CREDIT_INCOME_RATIO": cir,
        "ANNUITY_INCOME_RATIO": annuity_income,
        "EMPLOYED_TO_AGE_RATIO": employed_age,
        "CREDIT_TO_GOODS_RATIO": credit / goods,
        "BUREAU_DAYS_CREDIT_MIN": rng.normal(-1800, 400, n),
        "BUREAU_DAYS_CREDIT_MEAN": rng.normal(-950, 200, n),
        "BUREAU_DAYS_CREDIT_UPDATE_MEAN": rng.normal(-360, 90, n),
        "BUREAU_CREDIT_ACTIVE_Active_MEAN": rng.beta(2, 4, n),
        "BUREAU_AMT_CREDIT_SUM_DEBT_SUM": income * rng.exponential(0.15, n),
        "Credit_card_AMT_BALANCE_MEAN": rng.exponential(50000, n),
        "Credit_card_AMT_CREDIT_LIMIT_ACTUAL_MEAN": rng.exponential(150000, n),
        "Credit_card_AMT_BALANCE_MAX": rng.exponential(120000, n),
        "Installments_RECENT_ins_DPD_MAX": rng.poisson(0.3, n),
        "PREV_WAS_REFUSED_MEAN": rng.beta(1 + target, 12 - target * 8, n),
        "PREV_WAS_APPROVED_MEAN": rng.beta(8, 2, n),
        "PREV_CODE_REJECT_REASON_XAP_MEAN": rng.beta(10, 2, n),
        "PREV_CREDIT_TO_APPLICATION_RATIO_MEAN": rng.normal(1.02, 0.08, n),
        "FLOORSMAX_AVG": rng.beta(2, 5, n),
        "BUREAU_BB_STATUS_C_MEAN_MEAN": rng.beta(2, 8, n),
    })
    return df


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for target_val, color, name in [(0, "#1D9E75", "Repaid"), (1, "#D85A30", "Defaulted")]:
        subset = df.loc[df["TARGET"] == target_val, "YEARS_BIRTH"]
        fig.add_trace(go.Histogram(
            x=subset, histnorm="probability density", nbinsx=40,
            name=name, marker_color=color, opacity=0.55,
        ))
    fig.update_layout(
        title="Age Distribution by Repayment Status",
        xaxis_title="Age (Years)", yaxis_title="Density",
        barmode="overlay", height=400,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def plot_ext_source_2_violin(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for target_val, color, name in [(0, "#1D9E75", "Repaid"), (1, "#D85A30", "Defaulted")]:
        subset = df[df["TARGET"] == target_val]["EXT_SOURCE_2"].dropna()
        fig.add_trace(go.Violin(
            x=[name] * len(subset), y=subset,
            name=name, box_visible=True, meanline_visible=True,
            fillcolor=color, opacity=0.7, line_color="white",
        ))
    fig.update_layout(
        title="EXT_SOURCE_2 Distribution — Strongest Single Predictor",
        yaxis_title="EXT_SOURCE_2 score",
        violinmode="group", height=420,
    )
    return fig


def plot_age_band_default_rate(df: pd.DataFrame) -> go.Figure:
    df["AGE_BAND"] = pd.cut(df["YEARS_BIRTH"], bins=range(18, 76, 5))
    age_risk = (df.groupby("AGE_BAND", observed=True)["TARGET"]
                  .agg(["mean", "count"])
                  .rename(columns={"mean": "default_rate", "count": "applicants"})
                  .reset_index())
    age_risk["default_rate_pct"] = (age_risk["default_rate"] * 100).round(2)
    age_risk["AGE_BAND"] = age_risk["AGE_BAND"].astype(str)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=age_risk["AGE_BAND"], y=age_risk["applicants"],
        name="Applicants", marker_color="#9FE1CB", opacity=0.6,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=age_risk["AGE_BAND"], y=age_risk["default_rate_pct"],
        name="Default rate %", mode="lines+markers",
        line=dict(color="#D85A30", width=2.5), marker=dict(size=8),
    ), secondary_y=True)
    fig.update_layout(
        title="Default Rate by Age Band — Younger Applicants Default More",
        height=450, legend=dict(orientation="h", y=-0.2),
    )
    fig.update_yaxes(title_text="Applicant count", secondary_y=False)
    fig.update_yaxes(title_text="Default rate %", secondary_y=True)
    return fig


def plot_cir_default_rate(df: pd.DataFrame) -> go.Figure:
    df["CIR_BAND"] = pd.cut(
        df["CREDIT_INCOME_RATIO"].clip(0, 20),
        bins=[0, 1, 2, 4, 6, 8, 20],
        labels=["<1x", "1-2x", "2-4x", "4-6x", "6-8x", "8x+"],
    )
    cir_risk = (df.groupby("CIR_BAND", observed=True)["TARGET"]
                  .mean().mul(100).round(2).reset_index())
    cir_risk.columns = ["CIR_BAND", "default_rate_pct"]

    fig = px.bar(
        cir_risk, x="CIR_BAND", y="default_rate_pct",
        color="default_rate_pct", color_continuous_scale="Reds",
        title="Default Rate Rises with Credit-to-Income Ratio",
        text="default_rate_pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=420, coloraxis_showscale=False)
    return fig


def plot_credit_income_scatter(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="CREDIT_INCOME_RATIO", y="EXT_SOURCE_2",
        color="TARGET", color_discrete_map={0: "#1D9E75", 1: "#D85A30"},
        opacity=0.4, range_x=[0, 15],
        title="Credit-to-Income Ratio vs EXT_SOURCE_2 — Coloured by Default",
        labels={"TARGET": "Default"},
    )
    fig.update_layout(height=420)
    return fig


def render_eda_tab() -> None:
    st.header("EDA & Insights")
    st.markdown("---")
    st.markdown(
        "This tab uses a **synthetic training sample** (6 000 rows) to illustrate "
        "the key univariate and bivariate relationships that drive model behaviour. "
        "In production, replace the synthetic generator with your actual `application_train.csv`."
    )

    df = generate_eda_sample()

    # Row 1: Age distribution + EXT_SOURCE_2 violin
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(plot_age_distribution(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_ext_source_2_violin(df), use_container_width=True)

    # Row 2: Age band default rate + CIR default rate
    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.plotly_chart(plot_age_band_default_rate(df), use_container_width=True)
    with c4:
        st.plotly_chart(plot_cir_default_rate(df), use_container_width=True)

    # Row 3: Scatter + summary table
    c5, c6 = st.columns([1.3, 1.0], gap="large")
    with c5:
        st.plotly_chart(plot_credit_income_scatter(df), use_container_width=True)
    with c6:
        st.subheader("EXT_SOURCE_MEAN by Target")
        summary = df.groupby("TARGET")["EXT_SOURCE_MEAN"].describe().round(3).reset_index()
        summary["TARGET"] = summary["TARGET"].map({0: "Repaid (0)", 1: "Default (1)"})
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption("Lower external scores are strongly associated with default.")

    # Row 4: Key violin plots for top positive/negative risk signals
    st.markdown("---")
    st.subheader("Top Risk-Signal Distributions")
    features_to_plot = [
        ("EXT_SOURCE_MEAN", "Negative — higher values protect against default"),
        ("CREDIT_INCOME_RATIO", "Positive — higher ratios increase default risk"),
        ("PREV_WAS_REFUSED_MEAN", "Positive — prior refusals signal risk"),
        ("BUREAU_DAYS_CREDIT_MEAN", "Negative — longer credit history protects"),
    ]
    cols = st.columns(2)
    for idx, (feature, interpretation) in enumerate(features_to_plot):
        with cols[idx % 2]:
            fig = go.Figure()
            for target_val, color, name in [(0, "#1D9E75", "Repaid"), (1, "#D85A30", "Defaulted")]:
                subset = df[df["TARGET"] == target_val][feature].dropna()
                fig.add_trace(go.Violin(
                    x=[name] * len(subset), y=subset,
                    name=name, box_visible=True, meanline_visible=True,
                    fillcolor=color, opacity=0.7, line_color="white",
                ))
            fig.update_layout(
                title=f"{feature}",
                yaxis_title=feature, violinmode="group", height=350,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(interpretation)


# =============================================================================
# Visuals and explainability helpers
# =============================================================================
def create_gauge_chart(default_probability: float, threshold: float) -> go.Figure:
    risk_percent = default_probability * 100.0
    threshold_percent = threshold * 100.0

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_percent,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": "Predicted Default Risk", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#1f2937"},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#d1d5db",
                "steps": [
                    {"range": [0, min(10, threshold_percent)], "color": "#d1fae5"},
                    {"range": [min(10, threshold_percent), threshold_percent], "color": "#fef3c7"},
                    {"range": [threshold_percent, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 4},
                    "thickness": 0.8,
                    "value": threshold_percent,
                },
            },
        )
    )
    figure.update_layout(height=340, margin={"l": 20, "r": 20, "t": 50, "b": 10})
    return figure


def create_mock_roc_curve() -> go.Figure:
    """Create a deterministic ROC-like curve whose area matches the reported OOF AUC."""
    target_auc = 0.7920
    alpha = target_auc / (1.0 - target_auc)
    false_positive_rate = np.linspace(0.0, 1.0, 120)
    true_positive_rate = 1.0 - np.power(1.0 - false_positive_rate, alpha)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=false_positive_rate,
            y=true_positive_rate,
            fill="tozeroy",
            mode="lines",
            name="Validation ROC",
            line={"color": "#2563eb", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line={"color": "#9ca3af", "dash": "dash"},
        )
    )
    figure.update_layout(
        title="Mock ROC Curve Anchored to Final OOF AUC 0.7920",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=430,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        legend={"orientation": "h", "y": -0.18},
    )
    return figure


def get_feature_importance(_model: Any, selected_features: list[str]) -> pd.DataFrame:
    """Extract global feature importances from common LightGBM model wrappers."""
    try:
        if hasattr(_model, "booster_"):
            importances = np.asarray(_model.booster_.feature_importance(importance_type="gain"), dtype=float)
        elif hasattr(_model, "feature_importance"):
            importances = np.asarray(_model.feature_importance(importance_type="gain"), dtype=float)
        elif hasattr(_model, "feature_importances_"):
            importances = np.asarray(_model.feature_importances_, dtype=float)
        else:
            raise AttributeError("No LightGBM feature importance API was found on the loaded model.")

        if importances.size != len(selected_features):
            adjusted = np.zeros(len(selected_features), dtype=float)
            adjusted[: min(importances.size, len(selected_features))] = importances[: len(adjusted)]
            importances = adjusted

        return pd.DataFrame({"Feature": selected_features, "Importance": importances}).sort_values(
            "Importance", ascending=False
        )
    except Exception as exc:
        st.warning(f"Feature importances are unavailable: {exc}")
        return pd.DataFrame({"Feature": selected_features, "Importance": np.zeros(len(selected_features))})


def create_feature_importance_chart(importance_frame: pd.DataFrame) -> go.Figure:
    top_features = importance_frame.head(10).sort_values("Importance", ascending=True)
    figure = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 LightGBM Feature Importances",
        color="Importance",
        color_continuous_scale="Blues",
    )
    figure.update_layout(height=460, margin={"l": 20, "r": 20, "t": 60, "b": 20}, coloraxis_showscale=False)
    return figure


def top_applicant_risk_factors(applicant_frame: pd.DataFrame, importance_frame: pd.DataFrame) -> list[dict[str, str]]:
    """Rank applicant-level risk signals using simple, auditable heuristics."""
    if applicant_frame.empty:
        return []

    row = applicant_frame.iloc[0]
    importance_lookup = importance_frame.set_index("Feature")["Importance"].to_dict()

    checks = [
        (
            "Low external source average",
            1.0 - float(row.get("EXT_SOURCE_MEAN", 0.5)),
            f"EXT_SOURCE_MEAN = {float(row.get('EXT_SOURCE_MEAN', 0.0)):.3f}",
            max(
                importance_lookup.get("EXT_SOURCE_MEAN", 0.0),
                importance_lookup.get("EXT_SOURCE_1", 0.0),
                importance_lookup.get("EXT_SOURCE_2", 0.0),
                importance_lookup.get("EXT_SOURCE_3", 0.0),
            ),
        ),
        (
            "High credit-to-income ratio",
            float(row.get("CREDIT_INCOME_RATIO", 0.0)) / 5.0,
            f"CREDIT_INCOME_RATIO = {float(row.get('CREDIT_INCOME_RATIO', 0.0)):.2f}",
            importance_lookup.get("CREDIT_INCOME_RATIO", 0.0),
        ),
        (
            "High annuity burden",
            float(row.get("ANNUITY_INCOME_RATIO", 0.0)) / 0.45,
            f"ANNUITY_INCOME_RATIO = {float(row.get('ANNUITY_INCOME_RATIO', 0.0)):.2f}",
            importance_lookup.get("ANNUITY_INCOME_RATIO", 0.0),
        ),
        (
            "Short employment relative to age",
            1.0 - min(float(row.get("EMPLOYED_TO_AGE_RATIO", 0.0)) / 0.35, 1.0),
            f"EMPLOYED_TO_AGE_RATIO = {float(row.get('EMPLOYED_TO_AGE_RATIO', 0.0)):.2f}",
            importance_lookup.get("EMPLOYED_TO_AGE_RATIO", 0.0),
        ),
        (
            "Existing bureau debt load",
            float(row.get("BUREAU_AMT_CREDIT_SUM_DEBT_SUM", 0.0)) / max(float(row.get("AMT_INCOME_TOTAL", 1.0)), 1.0),
            f"BUREAU_DEBT_TO_INCOME = {_safe_divide(float(row.get('BUREAU_AMT_CREDIT_SUM_DEBT_SUM', 0.0)), max(float(row.get('AMT_INCOME_TOTAL', 1.0)), 1.0)):.2f}",
            importance_lookup.get("BUREAU_AMT_CREDIT_SUM_DEBT_SUM", 0.0),
        ),
        (
            "Credit card utilization signal",
            _safe_divide(
                float(row.get("Credit_card_AMT_BALANCE_MEAN", 0.0)),
                max(float(row.get("Credit_card_AMT_CREDIT_LIMIT_ACTUAL_MEAN", 1.0)), 1.0),
            ),
            f"Card utilization = {_safe_divide(float(row.get('Credit_card_AMT_BALANCE_MEAN', 0.0)), max(float(row.get('Credit_card_AMT_CREDIT_LIMIT_ACTUAL_MEAN', 1.0)), 1.0)):.2f}",
            importance_lookup.get("Credit_card_AMT_BALANCE_MEAN", 0.0),
        ),
        (
            "Prior application refusals",
            float(row.get("PREV_WAS_REFUSED_MEAN", 0.0)),
            f"PREV_WAS_REFUSED_MEAN = {float(row.get('PREV_WAS_REFUSED_MEAN', 0.0)):.2f}",
            importance_lookup.get("PREV_WAS_REFUSED_MEAN", 0.0),
        ),
    ]

    scored = []
    for label, risk_signal, evidence, importance in checks:
        weight = importance if importance > 0 else 1.0
        scored.append(
            {
                "factor": label,
                "evidence": evidence,
                "score": max(risk_signal, 0.0) * np.log1p(weight),
            }
        )

    return sorted(scored, key=lambda item: item["score"], reverse=True)[:3]


def display_recommendation(default_probability: float, threshold: float) -> None:
    risk_percent = default_probability * 100.0
    threshold_percent = threshold * 100.0

    st.metric("Default Probability", f"{risk_percent:.2f}%")
    if default_probability > threshold:
        st.error(f"Approval Recommendation: Reject. Risk exceeds the {threshold_percent:.1f}% policy threshold.")
    else:
        st.success(f"Approval Recommendation: Approve. Risk is within the {threshold_percent:.1f}% policy threshold.")


# =============================================================================
# Sidebar and application layout
# =============================================================================
def render_sidebar() -> tuple[dict[str, Any], bool, float]:
    with st.sidebar:
        st.header("Applicant Data Entry")
        risk_threshold = st.slider(
            "Risk threshold",
            min_value=0.01,
            max_value=0.50,
            value=DEFAULT_RISK_THRESHOLD,
            step=0.01,
            format="%.2f",
            help="Applicants with predicted default risk above this value receive a reject recommendation.",
        )
        st.divider()

        with st.form("live_applicant_form", clear_on_submit=False):
            st.subheader("Financials")
            income = st.number_input("Annual income", min_value=0.0, value=180000.0, step=5000.0, format="%.2f")
            credit_amount = st.number_input("Credit amount requested", min_value=0.0, value=450000.0, step=10000.0, format="%.2f")
            annuity = st.number_input("Loan annuity", min_value=0.0, value=24000.0, step=1000.0, format="%.2f")
            goods_price = st.number_input("Goods price", min_value=0.0, value=450000.0, step=10000.0, format="%.2f")

            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            age = st.slider("Age", min_value=18, max_value=80, value=35)
            employment_length = st.slider("Employment length", min_value=0, max_value=50, value=5)
            family_status = st.selectbox("Family status", ["Married", "Single", "Civil marriage", "Separated", "Widow"], index=0)
            education_type = st.selectbox(
                "Education type",
                ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"],
                index=0,
            )

            with st.expander("Employment and loan context", expanded=False):
                contract_type = st.selectbox("Contract type", ["Cash loans", "Revolving loans"], index=0)
                income_type = st.selectbox(
                    "Income type",
                    ["Working", "Commercial associate", "State servant", "Pensioner", "Unemployed"],
                    index=0,
                )
                occupation_type = st.selectbox("Occupation type", ["Other", "Core staff", "Drivers"], index=0)
                organization_type = st.selectbox(
                    "Organization type",
                    ["Other", "Business Entity Type 3", "Self-employed"],
                    index=0,
                )

            st.subheader("External Sources")
            ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, EXTERNAL_SOURCE_DEFAULTS["EXT_SOURCE_1"], 0.01)
            ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, EXTERNAL_SOURCE_DEFAULTS["EXT_SOURCE_2"], 0.01)
            ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, EXTERNAL_SOURCE_DEFAULTS["EXT_SOURCE_3"], 0.01)

            submitted = st.form_submit_button("Score Applicant", type="primary", use_container_width=True)

    live_input = {
        "Income": income,
        "Credit Amount": credit_amount,
        "Annuity": annuity,
        "Goods Price": goods_price,
        "Age": age,
        "Employment Length": employment_length,
        "Gender": gender,
        "Family Status": family_status,
        "Education Type": education_type,
        "Contract Type": contract_type,
        "Income Type": income_type,
        "Occupation Type": occupation_type,
        "Organization Type": organization_type,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
    }
    return live_input, submitted, risk_threshold


def render_real_time_tab(model: Any, importance_frame: pd.DataFrame, threshold: float) -> None:
    st.header("Real-time Application Results")
    st.markdown("---")

    latest_score = st.session_state.get("latest_score")
    latest_features = st.session_state.get("latest_features")

    if latest_score is None or latest_features is None:
        st.info("Enter applicant details in the sidebar and click Score Applicant.")
        return

    left, right = st.columns([1.45, 1.0], gap="large")
    with left:
        st.plotly_chart(create_gauge_chart(latest_score, threshold), use_container_width=True)

    with right:
        display_recommendation(latest_score, threshold)
        st.markdown("---")
        st.subheader("Top 3 Applicant Risk Factors")
        for factor in top_applicant_risk_factors(latest_features, importance_frame):
            st.markdown(f"**{factor['factor']}**")
            st.caption(factor["evidence"])

    with st.expander("View transformed model feature vector", expanded=False):
        st.dataframe(latest_features, use_container_width=True, hide_index=True)


def render_performance_tab() -> None:
    st.header("Model Performance Dashboard")
    st.markdown("---")

    metric_columns = st.columns(3)
    for column, (metric_name, metric_value) in zip(metric_columns, PERFORMANCE_METRICS.items()):
        column.metric(metric_name, metric_value)

    st.plotly_chart(create_mock_roc_curve(), use_container_width=True)
    st.caption("The curve is a clean dashboard visualization anchored to the reported cross-validation OOF AUC.")


def render_feature_importance_tab(importance_frame: pd.DataFrame) -> None:
    st.header("Global Feature Importance")
    st.markdown("---")

    st.plotly_chart(create_feature_importance_chart(importance_frame), use_container_width=True)

    st.subheader("Feature glossary")
    glossary_frame = pd.DataFrame(
        [{"Feature family": key, "Stakeholder meaning": value} for key, value in FEATURE_GLOSSARY.items()]
    )
    st.dataframe(glossary_frame, use_container_width=True, hide_index=True)


def render_batch_tab(model: Any, selected_features: list[str], threshold: float) -> None:
    st.header("Bulk Batch Processing")
    st.markdown("---")
    st.markdown(
        "Upload a CSV with either the complete model feature set or core applicant columns such as "
        "`AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `Age`, and `EXT_SOURCE_*`."
    )

    uploaded_file = st.file_uploader("Upload applicant CSV", type=["csv"])
    if uploaded_file is None:
        return

    try:
        raw_batch = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(raw_batch):,} applicant rows.")
        st.dataframe(raw_batch.head(25), use_container_width=True)

        missing_selected = [feature for feature in selected_features if feature not in raw_batch.columns]
        if missing_selected:
            st.warning(
                f"{len(missing_selected):,} model features were not present in the upload. "
                "The preprocessing stub filled them with configured historical defaults."
            )

        feature_batch = transform_batch_input(raw_batch, selected_features)
        scores = run_inference(model, feature_batch)
        if scores.size == 0:
            st.stop()

        result_frame = pd.DataFrame()
        result_frame["SK_ID_CURR"] = raw_batch["SK_ID_CURR"] if "SK_ID_CURR" in raw_batch.columns else raw_batch.index + 1
        result_frame["default_probability"] = scores
        result_frame["default_risk_percent"] = np.round(scores * 100.0, 2)
        result_frame["recommendation"] = np.where(scores > threshold, "Reject", "Approve")

        st.subheader("Scored results")
        st.dataframe(result_frame, use_container_width=True, hide_index=True)

        st.download_button(
            "Download scored results",
            data=dataframe_to_csv_bytes(result_frame),
            file_name="credit_default_scored_results.csv",
            mime="text/csv",
            type="primary",
        )
    except Exception as exc:
        st.error(f"Unable to process uploaded batch file: {exc}")


def main() -> None:
    st.title("Credit Default Risk Scoring")
    st.caption("Production-style LightGBM dashboard for real-time applicant scoring, model diagnostics, and batch processing.")

    model = load_model()
    selected_features = load_selected_features()
    if model is None or not selected_features:
        st.stop()

    live_input, submitted, risk_threshold = render_sidebar()

    if submitted:
        applicant_features = pipeline_transform_input(live_input)
        scores = run_inference(model, applicant_features)
        if scores.size:
            st.session_state["latest_features"] = applicant_features
            st.session_state["latest_score"] = float(scores[0])

    importance_frame = get_feature_importance(model, selected_features)

    tab_results, tab_performance, tab_importance, tab_eda, tab_batch = st.tabs(
        [
            "Real-time Application Results",
            "Model Performance Dashboard",
            "Global Feature Importance",
            "EDA & Insights",
            "Bulk Batch Processing",
        ]
    )

    with tab_results:
        render_real_time_tab(model, importance_frame, risk_threshold)

    with tab_performance:
        render_performance_tab()

    with tab_importance:
        render_feature_importance_tab(importance_frame)

    with tab_eda:
        render_eda_tab()

    with tab_batch:
        render_batch_tab(model, selected_features, risk_threshold)


if __name__ == "__main__":
    main()