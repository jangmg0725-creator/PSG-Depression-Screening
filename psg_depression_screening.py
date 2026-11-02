# ============================================================
# PSG SUMMARY-BASED DEPRESSION SCREENING
# Reproducible model development and external validation pipeline
#
# Study objective
# Develop and externally validate a screening model for clinically
# significant depressive symptoms using all routinely available
# polysomnography summary variables collected in sleep-laboratory practice.
#
# Analysis flow
# 1. Validate KUAH development and CNUH external-validation datasets
# 2. Use the prespecified full predictor set (23 variables)
# 3. Split KUAH into training and fixed internal hold-out partitions
# 4. Compare candidate algorithms using repeated nested cross-validation
# 5. Select the algorithm with the highest training-set OOF AUROC
# 6. Tune and refit the selected algorithm using the full training partition
# 7. Derive screening thresholds only from training-set OOF predictions
# 8. Evaluate internal hold-out and independent CNUH external validation
# 9. Generate discrimination, calibration, DCA, SHAP, subgroup,
#    and error-analysis outputs
#
# Important design rules
# - M1-M3 comparisons are intentionally not performed.
# - All model and threshold selection occurs in the KUAH training partition.
# - Internal hold-out and CNUH data are used only for evaluation.
# - The depression questionnaire used to define the outcome must never be
#   included as a predictor.
# ============================================================

from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.svm import SVC


# ============================================================
# 0. USER SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

OUTER_FOLDS = 5
OUTER_REPEATS = 3
INNER_FOLDS = 3
FINAL_TUNING_FOLDS = 5

# Optuna trials per inner optimization. Increase to 80-120 for the locked final analysis.
N_OPTUNA_TRIALS = 60

# Bootstrap iterations for AUROC confidence intervals.
N_BOOTSTRAP = 2000

# Primary operating point: maximize the Youden index on training OOF predictions.
# Sensitivity-target thresholds are retained only as secondary sensitivity analyses.
SECONDARY_SENSITIVITY_TARGETS = [0.80, 0.85, 0.90]

ANAM_PATH = Path("data/ANAM.csv")
CNUH_PATH = Path("data/CNUH.csv")
OUT_DIR = Path("outputs/psg_depression_screening")

LABEL_COL = "DEP label"
ID_COL = "number"

# Missing-data policy:
# - "drop": complete-case analysis; rows with any missing predictor are excluded.
# - "error": stop and report missing predictors.
# The manuscript currently describes complete-case analysis, so "drop" is the default.
MISSING_DATA_STRATEGY = "drop"

# Save excluded rows for transparent participant-flow reporting.
SAVE_EXCLUDED_MISSING_ROWS = True

# Selection rule for the final feature block.
# "internal_oof_auc" is the prespecified, development-only option.
FINAL_BLOCK_SELECTION_RULE = "internal_oof_auc"

# Set to True only when all optional packages are installed.
USE_XGBOOST = True
USE_LIGHTGBM = True
USE_CATBOOST = True

# Keep individual learners single-threaded to avoid nested oversubscription.
MODEL_N_JOBS = 1
OPTUNA_SHOW_PROGRESS = False

# Manuscript-facing error-analysis variables. These were prespecified because
# they directly reflect the subjective sleep, age, and respiratory patterns
# discussed in the Results. Change only before the final locked analysis.
ERROR_FIGURE_FEATURES = [
    "PSQI",
    "age",
    "Total AHI_Total",
    "HI_total",
    "LowestO2SatInTIB",
]

ERROR_FIGURE_LABELS = {
    "PSQI": "PSQI",
    "age": "Age (years)",
    "Total AHI_Total": "AHI (events/h)",
    "HI_total": "Hypopnea index (events/h)",
    "LowestO2SatInTIB": "Lowest oxygen saturation (%)",
}


# ============================================================
# 1. OUTPUT DIRECTORIES
# ============================================================

CSV_DIR = OUT_DIR / "csv"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "models"
SHAP_DIR = OUT_DIR / "shap"

for directory in [OUT_DIR, CSV_DIR, FIG_DIR, MODEL_DIR, SHAP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
# 2. PRESPECIFIED FULL SUMMARY PREDICTOR SET
# ============================================================

DEMOGRAPHIC_FEATURES = [
    "age",
    "sex",
    "BMI",
]

ANTHROPOMETRIC_FEATURES = [
    "neckcir_sitting",
    "wasistcir",  # Verify this spelling in both CSV files.
]

PSQI_FEATURES = [
    "PSQI",
]

PSG_FEATURES = [
    "TIB",
    "TST",
    "TWT %TIB",
    "WASO %TIB",
    "Sleep Latency",
    "REM Latency",
    "Sleep efficiency %",
    "Stage N1N2 % TST",
    "Stage N3 Sleep % TST",
    "REM sleep % TST",
    "Total AHI_Total",
    "HI_total",
    "LowestO2SatInTIB",
    "Supine_%",
    "Lateral_%",
    "LMS_total_I",
    "PLMS_total_I",
]

FULL_FEATURES = (
    DEMOGRAPHIC_FEATURES
    + ANTHROPOMETRIC_FEATURES
    + PSQI_FEATURES
    + PSG_FEATURES
)

MODEL_NAME_FOR_OUTPUT = "All_Routine_Summary_Data"

# Variables that must never be predictors.
FORBIDDEN_PREDICTOR_KEYWORDS = [
    "dep label",
    "scl",
    "bdi",
    "depression",
    "outcome",
    "target",
    "prediction",
    "prob",
]


# ============================================================
# 3. DATA VALIDATION
# ============================================================

def normalize_binary_label(series: pd.Series, name: str) -> pd.Series:
    """Validate and return an integer 0/1 outcome."""
    numeric = pd.to_numeric(series, errors="raise")
    values = set(numeric.dropna().unique().tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0 and 1. Found: {sorted(values)}")
    if numeric.isna().any():
        raise ValueError(f"{name} contains missing values.")
    return numeric.astype(int)


def validate_feature_names() -> None:
    """Check the full predictor set for count, duplicates, and leakage."""
    if len(FULL_FEATURES) != 23:
        raise ValueError(
            f"The full routine-summary model must contain 23 predictors; "
            f"found {len(FULL_FEATURES)}."
        )

    if len(set(FULL_FEATURES)) != len(FULL_FEATURES):
        duplicates = (
            pd.Series(FULL_FEATURES)
            [pd.Series(FULL_FEATURES).duplicated()]
            .tolist()
        )
        raise ValueError(f"Duplicate predictors found: {duplicates}")

    suspicious = []
    for feature in FULL_FEATURES:
        lower = feature.lower()
        if any(keyword in lower for keyword in FORBIDDEN_PREDICTOR_KEYWORDS):
            suspicious.append(feature)

    if suspicious:
        raise ValueError(
            f"Possible outcome leakage predictors detected: {suspicious}"
        )


def validate_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate columns, numeric conversion, missingness, coding, and basic ranges.

    Returns
    -------
    cleaned_df:
        Dataset after the configured missing-data policy.
    audit:
        Counts required for participant-flow reporting.
    """
    required = [LABEL_COL] + FULL_FEATURES
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{dataset_name}: missing required columns:\n{missing_columns}\n"
            "Correct the feature names in Section 2 before rerunning."
        )

    df = df.copy()
    n_original = len(df)

    df[LABEL_COL] = normalize_binary_label(
        df[LABEL_COL],
        f"{dataset_name}.{LABEL_COL}",
    )

    conversion_failures: Dict[str, int] = {}
    for feature in FULL_FEATURES:
        original_nonmissing = df[feature].notna()
        converted = pd.to_numeric(df[feature], errors="coerce")
        newly_missing = original_nonmissing & converted.isna()
        if newly_missing.any():
            conversion_failures[feature] = int(newly_missing.sum())
        df[feature] = converted

    if conversion_failures:
        raise ValueError(
            f"{dataset_name}: non-numeric predictor values were found:\n"
            f"{conversion_failures}"
        )

    missing_by_variable = df[FULL_FEATURES].isna().sum()
    missing_by_variable = missing_by_variable[missing_by_variable > 0]
    missing_case_mask = df[FULL_FEATURES].isna().any(axis=1)
    n_missing_cases = int(missing_case_mask.sum())

    if MISSING_DATA_STRATEGY not in {"drop", "error"}:
        raise ValueError(
            "MISSING_DATA_STRATEGY must be either 'drop' or 'error'."
        )

    if n_missing_cases > 0:
        print(f"\n{dataset_name}: missing predictor values detected")
        print(missing_by_variable.to_string())
        print(
            f"{dataset_name}: {n_missing_cases} unique participants have at "
            "least one missing predictor."
        )

        if MISSING_DATA_STRATEGY == "error":
            raise ValueError(
                f"{dataset_name}: missing predictors were found and "
                "MISSING_DATA_STRATEGY='error'."
            )

        if SAVE_EXCLUDED_MISSING_ROWS:
            excluded = df.loc[missing_case_mask].copy()
            excluded.insert(0, "Exclusion_reason", "Missing predictor")
            excluded.to_csv(
                CSV_DIR / f"{dataset_name}_EXCLUDED_MISSING_CASES.csv",
                index=False,
                encoding="utf-8-sig",
            )

        df = df.loc[~missing_case_mask].copy()
        print(
            f"{dataset_name}: complete-case exclusion applied; "
            f"{n_missing_cases} excluded and {len(df)} retained."
        )
    else:
        print(
            f"{dataset_name}: no missing values among the "
            f"{len(FULL_FEATURES)} prespecified predictors."
        )

    if len(df) == 0:
        raise ValueError(f"{dataset_name}: no participants remain after cleaning.")

    class_counts = df[LABEL_COL].value_counts().sort_index()
    if len(class_counts) < 2:
        raise ValueError(
            f"{dataset_name}: both outcome classes are required after cleaning. "
            f"Observed counts: {class_counts.to_dict()}"
        )

    warning_messages = []

    if not df["sex"].dropna().isin([0, 1]).all():
        warning_messages.append(
            f"{dataset_name}: sex is not uniformly coded as 0/1. "
            f"Observed values: "
            f"{sorted(df['sex'].dropna().unique().tolist())[:10]}"
        )

    # These variables may legitimately be stored as proportions. The code does
    # not rescale them automatically; it only checks that the two institutions
    # are interpreted consistently.
    proportion_candidates = [
        "TWT %TIB",
        "WASO %TIB",
        "Stage N1N2 % TST",
        "Stage N3 Sleep % TST",
        "REM sleep % TST",
        "Supine_%",
        "Lateral_%",
    ]
    for feature in proportion_candidates:
        maximum = float(df[feature].max())
        if maximum <= 1.5:
            warning_messages.append(
                f"{dataset_name}: {feature} appears to use a 0-1 proportion "
                f"scale (max={maximum:.3f}). This is acceptable for modeling "
                "when KUAH and CNUH use the same scale; multiply by 100 only "
                "when producing percentage-formatted descriptive tables."
            )

    for message in warning_messages:
        print("WARNING:", message)

    audit = {
        "Dataset": dataset_name,
        "N_original": int(n_original),
        "N_excluded_missing": int(n_missing_cases),
        "N_retained": int(len(df)),
        "Positive_N_retained": int(df[LABEL_COL].sum()),
        "Negative_N_retained": int((df[LABEL_COL] == 0).sum()),
        "Prevalence_retained": float(df[LABEL_COL].mean()),
        "Missing_data_strategy": MISSING_DATA_STRATEGY,
    }

    return df.reset_index(drop=True), audit

def save_data_audit(anam: pd.DataFrame, cnuh: pd.DataFrame) -> None:
    rows = []
    for dataset_name, df in [("KUAH", anam), ("CNUH", cnuh)]:
        for feature in [LABEL_COL] + FULL_FEATURES:
            values = pd.to_numeric(df[feature], errors="coerce")
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Variable": feature,
                    "N": len(df),
                    "Missing": int(values.isna().sum()),
                    "Unique": int(values.nunique(dropna=True)),
                    "Min": values.min(),
                    "Median": values.median(),
                    "Max": values.max(),
                }
            )
    pd.DataFrame(rows).to_csv(
        CSV_DIR / "DATA_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 4. METRICS
# ============================================================

def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan


def npv_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected calibration error using equal-frequency bins.
    This matches the quantile-binned calibration plots.
    """
    frame = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
    try:
        frame["bin"] = pd.qcut(frame["p"], q=n_bins, duplicates="drop")
    except ValueError:
        return np.nan

    total = len(frame)
    ece = 0.0
    for _, group in frame.groupby("bin", observed=True):
        observed = group["y"].mean()
        predicted = group["p"].mean()
        ece += abs(observed - predicted) * len(group) / total
    return float(ece)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    aucs = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_true), len(y_true))
        if np.unique(y_true[indices]).size < 2:
            continue
        aucs.append(roc_auc_score(y_true[indices], y_prob[indices]))

    if not aucs:
        return np.nan, np.nan
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def threshold_for_target_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float,
) -> float:
    """
    Select the highest-specificity threshold among thresholds achieving
    at least the requested sensitivity.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    eligible = np.where(tpr >= target_sensitivity)[0]
    if len(eligible) == 0:
        return 0.0

    specificities = 1.0 - fpr[eligible]
    best_local = eligible[np.argmax(specificities)]
    return float(thresholds[best_local])


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def evaluate_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob)

    return {
        "N": int(len(y_true)),
        "Positive_N": int(np.sum(y_true == 1)),
        "Prevalence": float(np.mean(y_true)),
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUROC_CI_low": float(ci_low),
        "AUROC_CI_high": float(ci_high),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "ECE": float(compute_ece(y_true, y_prob)),
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(specificity_score(y_true, y_pred)),
        "PPV": float(precision_score(y_true, y_pred, zero_division=0)),
        "NPV": float(npv_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def create_paper_performance_table(
    performance_df: pd.DataFrame,
    save_path: Path,
) -> pd.DataFrame:
    """Create the requested manuscript-facing performance columns."""
    table = performance_df.copy()
    table["AUROC (95% CI)"] = table.apply(
        lambda row: (
            f"{row['AUROC']:.3f} "
            f"({row['AUROC_CI_low']:.3f}–{row['AUROC_CI_high']:.3f})"
        ),
        axis=1,
    )

    table["AUPRC"] = table["AUPRC"].map(lambda value: f"{value:.3f}")
    table["Accuracy"] = table["Accuracy"].map(lambda value: f"{value:.3f}")
    table["Sensitivity"] = table["Sensitivity"].map(lambda value: f"{value:.3f}")
    table["Specificity"] = table["Specificity"].map(lambda value: f"{value:.3f}")
    table["F1-score"] = table["F1"].map(lambda value: f"{value:.3f}")

    table = table[
        [
            "Algorithm",
            "AUROC (95% CI)",
            "AUPRC",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "F1-score",
        ]
    ]
    table.to_csv(save_path, index=False, encoding="utf-8-sig")
    return table


# ============================================================
# 5. OPTUNA MODEL FACTORIES AND IMBALANCE STRATEGIES
# ============================================================

@dataclass
class ModelSpec:
    supports_class_weight: bool = True


def build_model_specs() -> Dict[str, ModelSpec]:
    specs = {
        "LogisticRegression": ModelSpec(True),
        "ElasticNetLogistic": ModelSpec(True),
        "QuadraticLogistic": ModelSpec(True),
        "ShrinkageLDA": ModelSpec(False),
        "RBF_SVM": ModelSpec(True),
        "RandomForest": ModelSpec(True),
        "ExtraTrees": ModelSpec(True),
        "HistGradientBoosting": ModelSpec(False),
    }
    if USE_XGBOOST:
        try:
            import xgboost  # noqa: F401
            specs["XGBoost"] = ModelSpec(True)
        except ImportError:
            print("WARNING: xgboost unavailable; skipped.")
    if USE_LIGHTGBM:
        try:
            import lightgbm  # noqa: F401
            specs["LightGBM"] = ModelSpec(True)
        except ImportError:
            print("WARNING: lightgbm unavailable; skipped.")
    if USE_CATBOOST:
        try:
            import catboost  # noqa: F401
            specs["CatBoost"] = ModelSpec(True)
        except ImportError:
            print("WARNING: catboost unavailable; skipped.")
    return specs


def suggest_imbalance_method(trial: optuna.Trial, supports_class_weight: bool) -> str:
    methods = ["none", "random_over", "smote"]
    if supports_class_weight:
        methods.insert(1, "class_weight")
    return trial.suggest_categorical("imbalance_method", methods)


def build_sampler(trial: optuna.Trial, method: str):
    if method == "random_over":
        ratio = trial.suggest_float("sampling_strategy", 0.20, 1.00, step=0.10)
        return RandomOverSampler(sampling_strategy=ratio, random_state=RANDOM_STATE)
    if method == "smote":
        ratio = trial.suggest_float("sampling_strategy", 0.20, 1.00, step=0.10)
        k = trial.suggest_int("smote_k_neighbors", 3, 7)
        return SMOTE(
            sampling_strategy=ratio,
            k_neighbors=k,
            random_state=RANDOM_STATE,
        )
    return "passthrough"


def build_trial_pipeline(
    trial: optuna.Trial,
    model_name: str,
    supports_class_weight: bool,
    y_reference: np.ndarray,
) -> Pipeline:
    method = suggest_imbalance_method(trial, supports_class_weight)
    class_weight = "balanced" if method == "class_weight" else None
    sampler = build_sampler(trial, method)
    scaler: Any = StandardScaler()
    poly: Any = "passthrough"

    if model_name == "LogisticRegression":
        estimator = LogisticRegression(
            C=trial.suggest_float("C", 1e-4, 1e3, log=True),
            penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
            solver="saga", max_iter=20000, class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
    elif model_name == "ElasticNetLogistic":
        estimator = LogisticRegression(
            C=trial.suggest_float("C", 1e-4, 1e3, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
            penalty="elasticnet", solver="saga", max_iter=25000,
            class_weight=class_weight, random_state=RANDOM_STATE,
        )
    elif model_name == "QuadraticLogistic":
        poly = PolynomialFeatures(
            degree=2, include_bias=False,
            interaction_only=trial.suggest_categorical("interaction_only", [True, False]),
        )
        estimator = LogisticRegression(
            C=trial.suggest_float("C", 1e-5, 10.0, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.2, 1.0),
            penalty="elasticnet", solver="saga", max_iter=30000,
            class_weight=class_weight, random_state=RANDOM_STATE,
        )
    elif model_name == "ShrinkageLDA":
        estimator = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage=trial.suggest_categorical(
                "shrinkage", ["auto", 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
            ),
        )
    elif model_name == "RBF_SVM":
        estimator = SVC(
            kernel="rbf", probability=True,
            C=trial.suggest_float("C", 1e-3, 1e3, log=True),
            gamma=trial.suggest_float("gamma", 1e-5, 1.0, log=True),
            class_weight=class_weight, random_state=RANDOM_STATE,
        )
    elif model_name in {"RandomForest", "ExtraTrees"}:
        scaler = "passthrough"
        cls = RandomForestClassifier if model_name == "RandomForest" else ExtraTreesClassifier
        estimator = cls(
            n_estimators=trial.suggest_int("n_estimators", 300, 1600, step=100),
            max_depth=trial.suggest_categorical("max_depth", [None, 2, 3, 4, 5, 7, 10]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 15),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            class_weight=class_weight,
            random_state=RANDOM_STATE, n_jobs=MODEL_N_JOBS,
        )
    elif model_name == "HistGradientBoosting":
        scaler = "passthrough"
        estimator = HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 800, step=50),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 3, 31),
            max_depth=trial.suggest_categorical("max_depth", [None, 2, 3, 4, 5]),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 50),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            early_stopping=True, validation_fraction=0.15,
            random_state=RANDOM_STATE,
        )
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier
        scaler = "passthrough"
        positives = max(int(np.sum(y_reference == 1)), 1)
        negatives = max(int(np.sum(y_reference == 0)), 1)
        spw = negatives / positives if method == "class_weight" else 1.0
        estimator = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            max_depth=trial.suggest_int("max_depth", 1, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            gamma=trial.suggest_float("gamma", 0.0, 2.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            scale_pos_weight=spw, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=RANDOM_STATE, n_jobs=MODEL_N_JOBS,
        )
    elif model_name == "LightGBM":
        from lightgbm import LGBMClassifier
        scaler = "passthrough"
        estimator = LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            num_leaves=trial.suggest_int("num_leaves", 3, 31),
            max_depth=trial.suggest_categorical("max_depth", [-1, 2, 3, 4, 5, 7]),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 60),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-5, 20.0, log=True),
            class_weight=class_weight, objective="binary", verbosity=-1,
            random_state=RANDOM_STATE, n_jobs=MODEL_N_JOBS,
        )
    elif model_name == "CatBoost":
        from catboost import CatBoostClassifier
        scaler = "passthrough"
        estimator = CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 200, 1200, step=100),
            depth=trial.suggest_int("depth", 2, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
            auto_class_weights="Balanced" if method == "class_weight" else None,
            loss_function="Logloss", eval_metric="AUC", verbose=False,
            allow_writing_files=False, random_seed=RANDOM_STATE, thread_count=MODEL_N_JOBS,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([
        ("poly", poly),
        ("scaler", scaler),
        ("sampler", sampler),
        ("model", estimator),
    ])


def optimize_model(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    spec: ModelSpec,
    cv: StratifiedKFold,
    seed: int,
) -> Tuple[Pipeline, Dict[str, Any], float, pd.DataFrame]:
    def objective(trial: optuna.Trial) -> float:
        fold_scores = []
        for fold_no, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
            pipeline = build_trial_pipeline(trial, model_name, spec.supports_class_weight, y[tr_idx])
            try:
                pipeline.fit(X.iloc[tr_idx], y[tr_idx])
                prob = pipeline.predict_proba(X.iloc[va_idx])[:, 1]
                score = roc_auc_score(y[va_idx], prob)
            except Exception as exc:
                trial.set_user_attr("failure", str(exc))
                return 0.5
            fold_scores.append(score)
            trial.report(float(np.mean(fold_scores)), step=fold_no)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed, multivariate=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=OPTUNA_SHOW_PROGRESS,
        gc_after_trial=True,
    )
    best_trial = study.best_trial
    final_pipeline = build_trial_pipeline(
        best_trial, model_name, spec.supports_class_weight, y
    )
    final_pipeline.fit(X, y)
    trials_df = study.trials_dataframe()
    return final_pipeline, best_trial.params, float(best_trial.value), trials_df


# ============================================================
# 6. REPEATED NESTED CROSS-VALIDATION WITH OPTUNA
# ============================================================

def nested_cv_oof_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    spec: ModelSpec,
    model_name: str,
    feature_set_name: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_FOLDS, n_repeats=OUTER_REPEATS,
        random_state=RANDOM_STATE,
    )
    oof_sum = np.zeros(len(y), dtype=float)
    oof_count = np.zeros(len(y), dtype=int)
    fold_rows = []
    total_folds = OUTER_FOLDS * OUTER_REPEATS

    for split_number, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):
        repeat = (split_number - 1) // OUTER_FOLDS + 1
        fold = (split_number - 1) % OUTER_FOLDS + 1
        inner_cv = StratifiedKFold(
            n_splits=INNER_FOLDS, shuffle=True,
            random_state=RANDOM_STATE + split_number,
        )
        fitted, params, inner_auc, trials = optimize_model(
            X.iloc[train_idx], y[train_idx], model_name, spec,
            inner_cv, RANDOM_STATE + split_number,
        )
        fold_prob = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        oof_sum[valid_idx] += fold_prob
        oof_count[valid_idx] += 1
        outer_auc = roc_auc_score(y[valid_idx], fold_prob)
        fold_rows.append({
            "Feature_set": feature_set_name,
            "Algorithm": model_name,
            "Repeat": repeat,
            "Outer_fold": fold,
            "Inner_best_AUROC": inner_auc,
            "Outer_AUROC": outer_auc,
            "Outer_AUPRC": average_precision_score(y[valid_idx], fold_prob),
            "Best_imbalance_method": params.get("imbalance_method", "none"),
            "Best_params": json.dumps(params, ensure_ascii=False, default=str),
        })
        trials.to_csv(
            CSV_DIR / f"OPTUNA_TRIALS_{model_name}_R{repeat}_F{fold}.csv",
            index=False, encoding="utf-8-sig",
        )
        print(
            f"[{feature_set_name} | {model_name}] split {split_number}/{total_folds} "
            f"(repeat {repeat}, fold {fold}): AUROC={outer_auc:.3f}, "
            f"imbalance={params.get('imbalance_method', 'none')}"
        )
    if np.any(oof_count != OUTER_REPEATS):
        raise RuntimeError("Repeated OOF prediction counts are inconsistent.")
    return oof_sum / oof_count, pd.DataFrame(fold_rows)


def fit_final_tuned_model(
    X: pd.DataFrame,
    y: np.ndarray,
    spec: ModelSpec,
    model_name: str,
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    cv = StratifiedKFold(
        n_splits=FINAL_TUNING_FOLDS, shuffle=True,
        random_state=RANDOM_STATE,
    )
    estimator, params, auc, trials = optimize_model(
        X, y, model_name, spec, cv, RANDOM_STATE + 10000
    )
    trials.to_csv(
        CSV_DIR / f"OPTUNA_FINAL_TUNING_{model_name}.csv",
        index=False, encoding="utf-8-sig",
    )
    return estimator, params, auc


# ============================================================
# 6B. DELONG TEST AND DECISION CURVE ANALYSIS
# ============================================================

def _compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty(n, dtype=float)
    result[order] = midranks
    return result


def _fast_delong(
    predictions_sorted_transposed: np.ndarray,
    label_1_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast DeLong covariance estimate for correlated ROC AUCs.
    Rows are models; positive observations must precede negative observations.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for row in range(k):
        tx[row] = _compute_midrank(positive[row])
        ty[row] = _compute_midrank(negative[row])
        tz[row] = _compute_midrank(predictions_sorted_transposed[row])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    covariance = sx / m + sy / n
    return aucs, covariance


def delong_auc_test(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
) -> Dict[str, float]:
    """Two-sided DeLong test comparing two correlated AUROCs."""
    from scipy.stats import norm

    y_true = np.asarray(y_true, dtype=int)
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)

    if len(np.unique(y_true)) < 2:
        return {
            "AUROC_A": np.nan,
            "AUROC_B": np.nan,
            "Difference": np.nan,
            "Z": np.nan,
            "P_value": np.nan,
        }

    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    predictions = np.vstack([prob_a, prob_b])[:, order]
    aucs, covariance = _fast_delong(predictions, label_1_count)

    contrast = np.array([[1.0, -1.0]])
    variance = float(contrast @ covariance @ contrast.T)
    if variance <= 0:
        z_value = np.nan
        p_value = np.nan
    else:
        z_value = float((aucs[0] - aucs[1]) / np.sqrt(variance))
        p_value = float(2.0 * norm.sf(abs(z_value)))

    return {
        "AUROC_A": float(aucs[0]),
        "AUROC_B": float(aucs[1]),
        "Difference": float(aucs[0] - aucs[1]),
        "Z": z_value,
        "P_value": p_value,
    }


def run_delong_algorithm_comparisons(
    y_true: np.ndarray,
    prediction_map: Dict[str, np.ndarray],
    selected_algorithm: str,
) -> pd.DataFrame:
    """Compare the selected algorithm with every other candidate by DeLong test."""
    selected_prob = prediction_map[selected_algorithm]
    rows = []
    for comparator, comparator_prob in prediction_map.items():
        if comparator == selected_algorithm:
            continue
        result = delong_auc_test(y_true, selected_prob, comparator_prob)
        rows.append(
            {
                "Selected_algorithm": selected_algorithm,
                "Comparator": comparator,
                **result,
            }
        )
    results = pd.DataFrame(rows)
    if not results.empty:
        results["Bonferroni_alpha"] = 0.05 / len(results)
        results["Bonferroni_significant"] = (
            results["P_value"] < results["Bonferroni_alpha"]
        )
        results = results.sort_values("P_value", na_position="last")
    return results


def decision_curve_values(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Calculate model, treat-all, and treat-none net benefit."""
    y_true = np.asarray(y_true, dtype=int)
    n = len(y_true)
    prevalence = float(y_true.mean())
    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        odds = threshold / (1.0 - threshold)
        model_nb = tp / n - fp / n * odds
        treat_all_nb = prevalence - (1.0 - prevalence) * odds

        rows.append(
            {
                "Threshold_probability": float(threshold),
                "Model_net_benefit": float(model_nb),
                "Treat_all_net_benefit": float(treat_all_nb),
                "Treat_none_net_benefit": 0.0,
            }
        )

    return pd.DataFrame(rows)


def plot_decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    dataset_name: str,
    save_path: Path,
) -> pd.DataFrame:
    thresholds = np.arange(0.01, 0.61, 0.01)
    values = decision_curve_values(y_true, y_prob, thresholds)

    plt.figure(figsize=(7, 6))
    plt.plot(
        values["Threshold_probability"],
        values["Model_net_benefit"],
        label="Final model",
    )
    plt.plot(
        values["Threshold_probability"],
        values["Treat_all_net_benefit"],
        linestyle="--",
        label="Treat all",
    )
    plt.plot(
        values["Threshold_probability"],
        values["Treat_none_net_benefit"],
        linestyle=":",
        label="Treat none",
    )
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"Decision curve analysis: {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return values


# ============================================================
# 7. PLOTS
# ============================================================

def plot_roc_curves(
    prediction_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    save_path: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    for label, (y_true, y_prob) in prediction_map.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{label} (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pr_curves(
    prediction_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    save_path: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    for label, (y_true, y_prob) in prediction_map.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"{label} (AUPRC={auprc:.3f})")
        plt.axhline(np.mean(y_true), linestyle=":", linewidth=1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_final_calibration(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Path,
) -> pd.DataFrame:
    """Save a combined calibration curve and the underlying bin-level data."""
    rows = []
    plt.figure(figsize=(7, 6))
    for dataset_name, (y_true, y_prob) in datasets.items():
        observed, predicted = calibration_curve(
            y_true,
            y_prob,
            n_bins=10,
            strategy="quantile",
        )
        brier = brier_score_loss(y_true, y_prob)
        ece = compute_ece(y_true, y_prob)
        plt.plot(
            predicted,
            observed,
            marker="o",
            label=f"{dataset_name} (Brier={brier:.3f}, ECE={ece:.3f})",
        )
        for bin_number, (pred, obs) in enumerate(zip(predicted, observed), start=1):
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Bin": bin_number,
                    "Mean_predicted_probability": float(pred),
                    "Observed_event_proportion": float(obs),
                    "Brier": float(brier),
                    "ECE": float(ece),
                }
            )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event proportion")
    plt.title("Calibration of the final model")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return pd.DataFrame(rows)


def plot_calibration_panels(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Path,
) -> None:
    """Paper-facing two-panel calibration figure for hold-out and external data."""
    n_panels = len(datasets)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.2), squeeze=False)
    for panel_index, (dataset_name, (y_true, y_prob)) in enumerate(datasets.items()):
        ax = axes[0, panel_index]
        observed, predicted = calibration_curve(
            y_true,
            y_prob,
            n_bins=10,
            strategy="quantile",
        )
        brier = brier_score_loss(y_true, y_prob)
        ece = compute_ece(y_true, y_prob)
        ax.plot(predicted, observed, marker="o", linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed event proportion")
        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(f"({panel_letter}) {dataset_name}")
        ax.text(
            0.04,
            0.96,
            f"Brier = {brier:.3f}\\nECE = {ece:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
        )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8. ERROR AND SUBGROUP ANALYSIS
# ============================================================

def assign_confusion_group(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    groups = np.empty(len(y_true), dtype=object)
    groups[(y_true == 1) & (y_pred == 1)] = "TP"
    groups[(y_true == 0) & (y_pred == 0)] = "TN"
    groups[(y_true == 0) & (y_pred == 1)] = "FP"
    groups[(y_true == 1) & (y_pred == 0)] = "FN"
    return groups


def median_iqr(series: pd.Series) -> str:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return ""
    return (
        f"{numeric.median():.3f} "
        f"[{numeric.quantile(0.25):.3f}, {numeric.quantile(0.75):.3f}]"
    )


def run_error_group_tests(
    prediction_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Compare FP vs TN and FN vs TP.
    Continuous variables: Mann-Whitney U.
    Sex: Fisher exact test.
    """
    rows = []
    comparisons = [("FP", "TN"), ("FN", "TP")]

    for group_a, group_b in comparisons:
        frame_a = prediction_df[prediction_df["confusion_group"] == group_a]
        frame_b = prediction_df[prediction_df["confusion_group"] == group_b]

        for feature in FULL_FEATURES:
            if feature == "sex":
                table = pd.crosstab(
                    prediction_df[
                        prediction_df["confusion_group"].isin([group_a, group_b])
                    ]["confusion_group"],
                    prediction_df[
                        prediction_df["confusion_group"].isin([group_a, group_b])
                    ]["sex"],
                )
                p_value = np.nan
                if table.shape == (2, 2):
                    _, p_value = fisher_exact(table.values)

                rows.append(
                    {
                        "Dataset": dataset_name,
                        "Comparison": f"{group_a}_vs_{group_b}",
                        "Variable": feature,
                        f"{group_a}_summary": frame_a[feature].value_counts().to_dict(),
                        f"{group_b}_summary": frame_b[feature].value_counts().to_dict(),
                        "P_value": p_value,
                        "Test": "Fisher exact",
                    }
                )
            else:
                values_a = pd.to_numeric(frame_a[feature], errors="coerce").dropna()
                values_b = pd.to_numeric(frame_b[feature], errors="coerce").dropna()

                p_value = np.nan
                if len(values_a) > 0 and len(values_b) > 0:
                    _, p_value = mannwhitneyu(
                        values_a,
                        values_b,
                        alternative="two-sided",
                    )

                rows.append(
                    {
                        "Dataset": dataset_name,
                        "Comparison": f"{group_a}_vs_{group_b}",
                        "Variable": feature,
                        f"{group_a}_summary": median_iqr(values_a),
                        f"{group_b}_summary": median_iqr(values_b),
                        "P_value": p_value,
                        "Test": "Mann-Whitney U",
                    }
                )

    results = pd.DataFrame(rows)
    number_of_tests = len(FULL_FEATURES)
    results["Bonferroni_alpha"] = 0.05 / number_of_tests
    results["Bonferroni_significant"] = (
        results["P_value"] < results["Bonferroni_alpha"]
    )
    return results


def add_subgroups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age_group"] = pd.cut(
        df["age"],
        bins=[-np.inf, 39, 49, 59, np.inf],
        labels=["<40", "40-49", "50-59", ">=60"],
    )
    df["BMI_group"] = pd.cut(
        df["BMI"],
        bins=[-np.inf, 22.9, 24.9, 29.9, np.inf],
        labels=["<23", "23-24.9", "25-29.9", ">=30"],
    )
    df["PSQI_group"] = pd.cut(
        df["PSQI"],
        bins=[-np.inf, 5, 10, 15, np.inf],
        labels=["0-5", "6-10", "11-15", ">=16"],
    )
    df["OSA_severity"] = pd.cut(
        df["Total AHI_Total"],
        bins=[-np.inf, 4.999, 14.999, 29.999, np.inf],
        labels=["Normal", "Mild", "Moderate", "Severe"],
    )
    return df


def wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion."""
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1.0 + z**2 / total
    center = (p + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt((p * (1.0 - p) / total) + (z**2 / (4.0 * total**2)))
        / denominator
    )
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def subgroup_performance(
    prediction_df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    frame = add_subgroups(prediction_df)
    subgroup_variables = [
        "sex",
        "Age_group",
        "BMI_group",
        "PSQI_group",
        "OSA_severity",
    ]

    rows = []
    for variable in subgroup_variables:
        for subgroup, group in frame.groupby(variable, observed=True, dropna=False):
            if len(group) == 0:
                continue
            y_true = group[LABEL_COL].to_numpy(dtype=int)
            y_pred = group["pred"].to_numpy(dtype=int)
            tn, fp, fn, tp = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            ).ravel()
            sensitivity = recall_score(y_true, y_pred, zero_division=0)
            specificity = specificity_score(y_true, y_pred)
            sens_low, sens_high = wilson_ci(int(tp), int(tp + fn))
            spec_low, spec_high = wilson_ci(int(tn), int(tn + fp))

            rows.append(
                {
                    "Dataset": dataset_name,
                    "Subgroup_variable": variable,
                    "Subgroup": str(subgroup),
                    "N": len(group),
                    "Positive_N": int(np.sum(y_true == 1)),
                    "Negative_N": int(np.sum(y_true == 0)),
                    "Sensitivity": sensitivity,
                    "Sensitivity_CI_low": sens_low,
                    "Sensitivity_CI_high": sens_high,
                    "Specificity": specificity,
                    "Specificity_CI_low": spec_low,
                    "Specificity_CI_high": spec_high,
                    "PPV": precision_score(y_true, y_pred, zero_division=0),
                    "NPV": npv_score(y_true, y_pred),
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TP": int(tp),
                }
            )
    return pd.DataFrame(rows)


def plot_error_analysis_figure(
    prediction_df: pd.DataFrame,
    save_path: Path,
    features: List[str] = ERROR_FIGURE_FEATURES,
) -> None:
    """Create a manuscript-facing error-analysis figure across both cohorts."""
    dataset_order = ["Internal_Holdout", "External_CNUH"]
    group_order = ["TN", "FP", "TP", "FN"]
    fig, axes = plt.subplots(
        len(dataset_order),
        len(features),
        figsize=(3.8 * len(features), 4.2 * len(dataset_order)),
        squeeze=False,
    )

    for row_index, dataset_name in enumerate(dataset_order):
        dataset_frame = prediction_df[prediction_df["Dataset"] == dataset_name]
        for column_index, feature in enumerate(features):
            ax = axes[row_index, column_index]
            data = []
            positions = []
            labels = []
            for position, group_name in enumerate(group_order, start=1):
                values = pd.to_numeric(
                    dataset_frame.loc[
                        dataset_frame["confusion_group"] == group_name,
                        feature,
                    ],
                    errors="coerce",
                ).dropna()
                if len(values) > 0:
                    data.append(values.to_numpy())
                    positions.append(position)
                    labels.append(group_name)
            if data:
                ax.boxplot(data, positions=positions, widths=0.6, showfliers=True)
            ax.set_xticks(range(1, len(group_order) + 1))
            ax.set_xticklabels(group_order)
            ax.set_ylabel(ERROR_FIGURE_LABELS.get(feature, feature))
            if row_index == 0:
                ax.set_title(ERROR_FIGURE_LABELS.get(feature, feature))
            if column_index == 0:
                cohort_label = "Internal hold-out" if dataset_name == "Internal_Holdout" else "External CNUH"
                ax.text(
                    -0.32,
                    0.5,
                    cohort_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                )
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Characteristics of correctly and incorrectly classified participants",
        y=1.01,
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_group_counts(
    prediction_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Display TP, TN, FP, and FN counts for each evaluation cohort."""
    counts = (
        prediction_df.groupby(["Dataset", "confusion_group"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["TN", "FP", "FN", "TP"], fill_value=0)
    )
    ax = counts.plot(kind="bar", figsize=(8, 5))
    ax.set_xlabel("")
    ax.set_ylabel("Number of participants")
    ax.set_title("Classification outcomes at the primary screening threshold")
    ax.legend(title="Outcome group")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_subgroup_performance_figure(
    subgroup_df: pd.DataFrame,
    dataset_name: str,
    save_path: Path,
) -> None:
    """Forest-style sensitivity and specificity plot with Wilson 95% CIs."""
    if subgroup_df.empty:
        return
    frame = subgroup_df.copy().reset_index(drop=True)
    frame["Label"] = (
        frame["Subgroup_variable"].astype(str)
        + ": "
        + frame["Subgroup"].astype(str)
        + " (n="
        + frame["N"].astype(str)
        + ")"
    )
    y_positions = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(6, 0.42 * len(frame))))

    for ax, metric, low, high, title in [
        (
            axes[0],
            "Sensitivity",
            "Sensitivity_CI_low",
            "Sensitivity_CI_high",
            "Sensitivity",
        ),
        (
            axes[1],
            "Specificity",
            "Specificity_CI_low",
            "Specificity_CI_high",
            "Specificity",
        ),
    ]:
        estimates = frame[metric].to_numpy(dtype=float)
        lower = frame[low].to_numpy(dtype=float)
        upper = frame[high].to_numpy(dtype=float)
        xerr = np.vstack([estimates - lower, upper - estimates])
        ax.errorbar(estimates, y_positions, xerr=xerr, fmt="o", capsize=3)
        ax.axvline(0.5, linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"{title} (95% CI)")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(frame["Label"] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(title)

    display_name = "Internal hold-out" if dataset_name == "Internal_Holdout" else "External CNUH"
    fig.suptitle(f"Subgroup performance: {display_name}", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 9. SHAP
# ============================================================

def run_shap_analysis(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    feature_names: List[str],
) -> None:
    """
    Explain the fitted final model on the internal hold-out set.
    The function uses shap.Explainer and saves values and figures.
    """
    try:
        import shap
    except ImportError as exc:
        print("WARNING: shap is unavailable; SHAP analysis was skipped.")
        print(exc)
        return

    # Limit the background to control memory usage.
    background_n = min(300, len(X_train))
    background = X_train.sample(
        n=background_n,
        random_state=RANDOM_STATE,
    )

    # Pipeline-based logistic regression needs a callable interface.
    if isinstance(estimator, Pipeline):
        explainer = shap.Explainer(
            estimator.predict_proba,
            background,
            feature_names=feature_names,
        )
        explanation = explainer(X_explain)
        values = explanation.values
        if values.ndim == 3:
            values = values[:, :, 1]
        explanation = shap.Explanation(
            values=values,
            base_values=(
                explanation.base_values[:, 1]
                if np.asarray(explanation.base_values).ndim == 2
                else explanation.base_values
            ),
            data=X_explain.to_numpy(),
            feature_names=feature_names,
        )
    else:
        explainer = shap.Explainer(
            estimator,
            background,
            feature_names=feature_names,
        )
        explanation = explainer(X_explain)
        values = explanation.values
        if values.ndim == 3:
            values = values[:, :, 1]
            explanation = shap.Explanation(
                values=values,
                base_values=(
                    explanation.base_values[:, 1]
                    if np.asarray(explanation.base_values).ndim == 2
                    else explanation.base_values
                ),
                data=X_explain.to_numpy(),
                feature_names=feature_names,
            )

    shap_values_df = pd.DataFrame(
        explanation.values,
        columns=feature_names,
        index=X_explain.index,
    )
    shap_values_df.to_csv(
        SHAP_DIR / "FINAL_MODEL_INTERNAL_HOLDOUT_SHAP_VALUES.csv",
        encoding="utf-8-sig",
    )

    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean_abs_SHAP": np.mean(
                np.abs(explanation.values),
                axis=0,
            ),
        }
    ).sort_values("Mean_abs_SHAP", ascending=False)

    importance.to_csv(
        SHAP_DIR / "FINAL_MODEL_SHAP_IMPORTANCE.csv",
        index=False,
        encoding="utf-8-sig",
    )

    shap.plots.beeswarm(explanation, max_display=23, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "FINAL_MODEL_SHAP_BEESWARM.png", dpi=300)
    plt.close()

    shap.plots.bar(explanation, max_display=23, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "FINAL_MODEL_SHAP_BAR.png", dpi=300)
    plt.close()

    top_features = importance.head(5)["Feature"].tolist()
    for feature in top_features:
        shap.plots.scatter(explanation[:, feature], show=False)
        plt.tight_layout()
        safe_name = feature.replace("/", "_").replace("%", "pct").replace(" ", "_")
        plt.savefig(
            SHAP_DIR / f"SHAP_DEPENDENCE_{safe_name}.png",
            dpi=300,
        )
        plt.close()

    # Combined manuscript-facing dependence figure for the five leading features.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for axis_index, feature in enumerate(top_features):
        ax = axes[axis_index]
        feature_index = feature_names.index(feature)
        x_values = X_explain[feature].to_numpy(dtype=float)
        y_values = explanation.values[:, feature_index]
        ax.scatter(x_values, y_values, alpha=0.55, s=22)

        # Add a robust rolling-median trend without imposing a parametric form.
        order = np.argsort(x_values)
        sorted_x = x_values[order]
        sorted_y = y_values[order]
        if len(sorted_x) >= 8:
            window = max(5, len(sorted_x) // 8)
            trend = pd.Series(sorted_y).rolling(window, center=True, min_periods=3).median()
            valid = trend.notna().to_numpy()
            ax.plot(sorted_x[valid], trend.to_numpy()[valid], linewidth=2)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel(feature)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"({chr(ord('a') + axis_index)}) {feature}")
        ax.grid(alpha=0.2)

    for unused_index in range(len(top_features), len(axes)):
        axes[unused_index].axis("off")
    fig.suptitle("SHAP dependence plots for the five leading predictors", y=1.01)
    fig.tight_layout()
    fig.savefig(
        SHAP_DIR / "FINAL_MODEL_SHAP_DEPENDENCE_TOP5_COMBINED.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    feature_group_map = {}
    for feature in feature_names:
        if feature in DEMOGRAPHIC_FEATURES:
            feature_group_map[feature] = "Demographic"
        elif feature in ANTHROPOMETRIC_FEATURES:
            feature_group_map[feature] = "Anthropometric"
        elif feature in PSQI_FEATURES:
            feature_group_map[feature] = "PSQI"
        elif feature in PSG_FEATURES:
            feature_group_map[feature] = "Objective_PSG"
        else:
            feature_group_map[feature] = "Other"

    group_rows = []
    for group_name in sorted(set(feature_group_map.values())):
        group_features = [
            feature
            for feature in feature_names
            if feature_group_map[feature] == group_name
        ]
        indices = [feature_names.index(feature) for feature in group_features]
        group_rows.append(
            {
                "Feature_group": group_name,
                "N_features": len(group_features),
                "Sum_mean_abs_SHAP": float(
                    np.mean(
                        np.sum(
                            np.abs(explanation.values[:, indices]),
                            axis=1,
                        )
                    )
                ),
                "Mean_per_feature_SHAP": float(
                    np.mean(np.abs(explanation.values[:, indices]))
                ),
            }
        )

    pd.DataFrame(group_rows).to_csv(
        SHAP_DIR / "FINAL_MODEL_SHAP_GROUP_CONTRIBUTION.csv",
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# 10. MAIN ANALYSIS
# ============================================================

def main() -> None:
    validate_feature_names()

    print("=" * 80)
    print("1. LOADING AND VALIDATING DATA")
    print("=" * 80)
    print(f"Missing-data strategy: {MISSING_DATA_STRATEGY}")
    print(f"Full routine-summary predictor count: {len(FULL_FEATURES)}")
    print(
        f"Repeated nested CV: {OUTER_FOLDS} folds x "
        f"{OUTER_REPEATS} repeats; "
        f"{N_OPTUNA_TRIALS} Optuna trials per inner optimization"
    )

    anam = pd.read_csv(ANAM_PATH)
    cnuh = pd.read_csv(CNUH_PATH)

    anam, anam_audit = validate_dataset(anam, "KUAH")
    cnuh, cnuh_audit = validate_dataset(cnuh, "CNUH")

    save_data_audit(anam, cnuh)

    pd.DataFrame([anam_audit, cnuh_audit]).to_csv(
        CSV_DIR / "PARTICIPANT_FLOW_AFTER_MISSING_DATA.csv",
        index=False,
        encoding="utf-8-sig",
    )

    anam_train, anam_internal = train_test_split(
        anam,
        test_size=TEST_SIZE,
        stratify=anam[LABEL_COL],
        random_state=RANDOM_STATE,
    )

    X_train = anam_train[FULL_FEATURES]
    X_internal = anam_internal[FULL_FEATURES]
    X_external = cnuh[FULL_FEATURES]

    y_train = anam_train[LABEL_COL].to_numpy(dtype=int)
    y_internal = anam_internal[LABEL_COL].to_numpy(dtype=int)
    y_external = cnuh[LABEL_COL].to_numpy(dtype=int)

    split_summary = pd.DataFrame(
        [
            {
                "Dataset": "KUAH_training",
                "N": len(anam_train),
                "Positive_N": int(y_train.sum()),
                "Negative_N": int((y_train == 0).sum()),
                "Prevalence": float(y_train.mean()),
            },
            {
                "Dataset": "KUAH_internal_holdout",
                "N": len(anam_internal),
                "Positive_N": int(y_internal.sum()),
                "Negative_N": int((y_internal == 0).sum()),
                "Prevalence": float(y_internal.mean()),
            },
            {
                "Dataset": "CNUH_external",
                "N": len(cnuh),
                "Positive_N": int(y_external.sum()),
                "Negative_N": int((y_external == 0).sum()),
                "Prevalence": float(y_external.mean()),
            },
        ]
    )
    split_summary.to_csv(
        CSV_DIR / "DATA_SPLIT_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(split_summary.to_string(index=False))

    # ========================================================
    # Stage 1: Algorithm comparison using all 23 predictors
    # ========================================================
    print("\n" + "=" * 80)
    print("2. CANDIDATE ALGORITHM COMPARISON USING ALL SUMMARY DATA")
    print("=" * 80)

    model_specs = build_model_specs()

    algorithm_oof_predictions: Dict[str, np.ndarray] = {}
    algorithm_fold_tables = []
    algorithm_rows = []

    for algorithm_name, spec in model_specs.items():
        print("\n" + "-" * 80)
        print(f"Algorithm: {algorithm_name}")
        print("-" * 80)

        oof_prob, fold_table = nested_cv_oof_predictions(
            X=X_train,
            y=y_train,
            spec=spec,
            model_name=algorithm_name,
            feature_set_name=MODEL_NAME_FOR_OUTPUT,
        )

        algorithm_oof_predictions[algorithm_name] = oof_prob
        algorithm_fold_tables.append(fold_table)

        algorithm_threshold = youden_threshold(y_train, oof_prob)
        metrics = evaluate_probabilities(
            y_train, oof_prob, algorithm_threshold
        )

        algorithm_rows.append(
            {
                "Algorithm": algorithm_name,
                "Predictor_set": MODEL_NAME_FOR_OUTPUT,
                "Feature_count": len(FULL_FEATURES),
                "Threshold_type": "Youden",
                **metrics,
            }
        )

        pd.DataFrame(
            {
                "Row_index": anam_train.index,
                "True_label": y_train,
                "OOF_probability": oof_prob,
            }
        ).to_csv(
            CSV_DIR / f"OOF_PREDICTIONS_{algorithm_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    algorithm_comparison = (
        pd.DataFrame(algorithm_rows)
        .sort_values(["AUROC", "AUPRC"], ascending=False)
        .reset_index(drop=True)
    )

    algorithm_comparison.to_csv(
        CSV_DIR / "TABLE_ALGORITHM_COMPARISON_ALL_METRICS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    algorithm_paper_table = create_paper_performance_table(
        algorithm_comparison,
        CSV_DIR / "TABLE_ALGORITHM_COMPARISON_PAPER_FORMAT.csv",
    )

    pd.concat(algorithm_fold_tables, ignore_index=True).to_csv(
        CSV_DIR / "ALGORITHM_COMPARISON_OUTER_FOLD_RESULTS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_algorithm = str(algorithm_comparison.iloc[0]["Algorithm"])
    best_spec = model_specs[best_algorithm]
    best_oof_prob = algorithm_oof_predictions[best_algorithm]

    print("\nSelected algorithm:", best_algorithm)
    print(algorithm_paper_table.to_string(index=False))

    delong_results = run_delong_algorithm_comparisons(
        y_true=y_train,
        prediction_map=algorithm_oof_predictions,
        selected_algorithm=best_algorithm,
    )
    delong_results.to_csv(
        CSV_DIR / "DELONG_SELECTED_VS_OTHER_ALGORITHMS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_roc_curves(
        {
            name: (y_train, probability)
            for name, probability in algorithm_oof_predictions.items()
        },
        title="Candidate algorithms using all routine PSG summary data",
        save_path=FIG_DIR / "FIG_ALGORITHM_COMPARISON_ROC.png",
    )

    plot_pr_curves(
        {
            name: (y_train, probability)
            for name, probability in algorithm_oof_predictions.items()
        },
        title="Candidate algorithms using all routine PSG summary data",
        save_path=FIG_DIR / "FIG_ALGORITHM_COMPARISON_PR.png",
    )

    # ========================================================
    # Stage 2: Tune and refit the selected algorithm
    # ========================================================
    print("\n" + "=" * 80)
    print("3. FINAL MODEL FITTING AND THRESHOLD DETERMINATION")
    print("=" * 80)

    final_estimator, best_params, final_tuning_auc = fit_final_tuned_model(
        X=X_train,
        y=y_train,
        spec=best_spec,
        model_name=best_algorithm,
    )

    internal_prob = final_estimator.predict_proba(X_internal)[:, 1]
    external_prob = final_estimator.predict_proba(X_external)[:, 1]

    thresholds = {"Youden": youden_threshold(y_train, best_oof_prob)}
    for target in SECONDARY_SENSITIVITY_TARGETS:
        thresholds[f"Sensitivity_{target:.2f}"] = threshold_for_target_sensitivity(
            y_train, best_oof_prob, target
        )

    primary_threshold = thresholds["Youden"]

    threshold_table = pd.DataFrame(
        [
            {
                "Threshold_type": threshold_name,
                "Threshold": threshold_value,
                "Derived_from": "KUAH training repeated nested-CV OOF predictions",
            }
            for threshold_name, threshold_value in thresholds.items()
        ]
    )
    threshold_table.to_csv(
        CSV_DIR / "FINAL_MODEL_THRESHOLDS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(
        final_estimator,
        MODEL_DIR / f"FINAL_{best_algorithm}.joblib",
    )

    with open(
        MODEL_DIR / f"FINAL_{best_algorithm}_PARAMS.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_params,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # ========================================================
    # Stage 3: Final performance evaluation
    # ========================================================
    print("\n" + "=" * 80)
    print("4. FINAL MODEL EVALUATION")
    print("=" * 80)

    evaluation_sets = {
        "Internal_OOF": (y_train, best_oof_prob),
        "Internal_Holdout": (y_internal, internal_prob),
        "External_CNUH": (y_external, external_prob),
    }

    final_rows = []
    for threshold_name, threshold in thresholds.items():
        for dataset_name, (y_true, y_prob) in evaluation_sets.items():
            final_rows.append(
                {
                    "Algorithm": best_algorithm,
                    "Predictor_set": MODEL_NAME_FOR_OUTPUT,
                    "Feature_count": len(FULL_FEATURES),
                    "Dataset": dataset_name,
                    "Threshold_type": threshold_name,
                    **evaluate_probabilities(y_true, y_prob, threshold),
                }
            )

    final_performance = pd.DataFrame(final_rows)
    final_performance.to_csv(
        CSV_DIR / "TABLE_FINAL_MODEL_ALL_THRESHOLDS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary_performance = final_performance[
        final_performance["Threshold_type"] == "Youden"
    ].copy()

    primary_performance.to_csv(
        CSV_DIR / "TABLE_FINAL_MODEL_PRIMARY_RESULTS.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_paper_table = primary_performance.copy()
    final_paper_table["Algorithm"] = (
        final_paper_table["Algorithm"]
        + " - "
        + final_paper_table["Dataset"]
    )
    final_paper_table = create_paper_performance_table(
        final_paper_table,
        CSV_DIR / "TABLE_FINAL_MODEL_PRIMARY_PAPER_FORMAT.csv",
    )

    print("\nFinal model primary results")
    print(final_paper_table.to_string(index=False))

    plot_roc_curves(
        {
            "Internal OOF": (y_train, best_oof_prob),
            "Internal hold-out": (y_internal, internal_prob),
            "External CNUH": (y_external, external_prob),
        },
        title=f"Final {best_algorithm} model ROC curves",
        save_path=FIG_DIR / "FIG_FINAL_MODEL_ROC.png",
    )

    plot_pr_curves(
        {
            "Internal OOF": (y_train, best_oof_prob),
            "Internal hold-out": (y_internal, internal_prob),
            "External CNUH": (y_external, external_prob),
        },
        title=f"Final {best_algorithm} model precision-recall curves",
        save_path=FIG_DIR / "FIG_FINAL_MODEL_PR.png",
    )

    calibration_data = plot_final_calibration(
        {
            "Internal OOF": (y_train, best_oof_prob),
            "Internal hold-out": (y_internal, internal_prob),
            "External CNUH": (y_external, external_prob),
        },
        save_path=FIG_DIR / "FIG_FINAL_MODEL_CALIBRATION_ALL_DATASETS.png",
    )
    calibration_data.to_csv(
        CSV_DIR / "CALIBRATION_BIN_LEVEL_DATA.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_calibration_panels(
        {
            "Internal hold-out": (y_internal, internal_prob),
            "External CNUH": (y_external, external_prob),
        },
        save_path=FIG_DIR / "FIG_FINAL_MODEL_CALIBRATION_PAPER_PANELS.png",
    )

    # ========================================================
    # Stage 4: Prediction files, error analysis, subgroups
    # ========================================================
    print("\n" + "=" * 80)
    print("5. ERROR, SUBGROUP, DCA, AND SHAP ANALYSES")
    print("=" * 80)

    prediction_frames = []
    error_test_tables = []
    subgroup_tables = []

    for dataset_name, original_df, y_true, y_prob in [
        (
            "Internal_Holdout",
            anam_internal,
            y_internal,
            internal_prob,
        ),
        (
            "External_CNUH",
            cnuh,
            y_external,
            external_prob,
        ),
    ]:
        output = original_df.copy()
        output["Dataset"] = dataset_name
        output["Algorithm"] = best_algorithm
        output["Predictor_set"] = MODEL_NAME_FOR_OUTPUT
        output["Primary_threshold"] = primary_threshold
        output["prob"] = y_prob
        output["pred"] = (y_prob >= primary_threshold).astype(int)
        output["confusion_group"] = assign_confusion_group(
            y_true,
            output["pred"].to_numpy(dtype=int),
        )
        output["correct"] = (
            output[LABEL_COL].to_numpy(dtype=int)
            == output["pred"].to_numpy(dtype=int)
        ).astype(int)

        prediction_frames.append(output)

        output.to_csv(
            CSV_DIR / f"FINAL_PREDICTIONS_{dataset_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        error_tests = run_error_group_tests(
            output,
            dataset_name,
        )
        error_tests.to_csv(
            CSV_DIR / f"ERROR_ANALYSIS_{dataset_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        error_test_tables.append(error_tests)

        subgroup_results = subgroup_performance(
            output,
            dataset_name,
        )
        subgroup_results.to_csv(
            CSV_DIR / f"SUBGROUP_PERFORMANCE_{dataset_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        subgroup_tables.append(subgroup_results)
        plot_subgroup_performance_figure(
            subgroup_df=subgroup_results,
            dataset_name=dataset_name,
            save_path=FIG_DIR / f"FIG_SUBGROUP_PERFORMANCE_{dataset_name}.png",
        )

        dca_values = plot_decision_curve(
            y_true=y_true,
            y_prob=y_prob,
            dataset_name=dataset_name,
            save_path=FIG_DIR / f"FIG_DCA_{dataset_name}.png",
        )
        dca_values.insert(0, "Dataset", dataset_name)
        dca_values.to_csv(
            CSV_DIR / f"DCA_VALUES_{dataset_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    combined_predictions.to_csv(
        CSV_DIR / "FINAL_PREDICTIONS_COMBINED.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_error_tests = pd.concat(error_test_tables, ignore_index=True)
    combined_error_tests.to_csv(
        CSV_DIR / "ERROR_ANALYSIS_COMBINED.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combined_subgroups = pd.concat(subgroup_tables, ignore_index=True)
    combined_subgroups.to_csv(
        CSV_DIR / "SUBGROUP_PERFORMANCE_COMBINED.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_error_analysis_figure(
        prediction_df=combined_predictions,
        save_path=FIG_DIR / "FIG_ERROR_ANALYSIS_PAPER.png",
    )
    plot_confusion_group_counts(
        prediction_df=combined_predictions,
        save_path=FIG_DIR / "FIG_CONFUSION_GROUP_COUNTS.png",
    )

    # SHAP is calculated on the fixed internal hold-out set.
    run_shap_analysis(
        estimator=final_estimator,
        X_train=X_train,
        X_explain=X_internal,
        feature_names=FULL_FEATURES,
    )

    # ========================================================
    # Stage 5: Save complete metadata
    # ========================================================
    metadata = {
        "analysis_name": "PSG summary-based depression screening",
        "study_objective": (
            "Screen clinically significant depressive symptoms using all "
            "routine sleep-laboratory summary data with Optuna-tuned "
            "imbalance handling and a Youden-index operating point."
        ),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "outer_folds": OUTER_FOLDS,
        "outer_repeats": OUTER_REPEATS,
        "inner_folds": INNER_FOLDS,
        "final_tuning_folds": FINAL_TUNING_FOLDS,
        "optuna_trials_per_optimization": N_OPTUNA_TRIALS,
        "bootstrap_iterations": N_BOOTSTRAP,
        "primary_threshold_rule": "Youden index from repeated nested-CV OOF predictions",
        "secondary_sensitivity_targets": SECONDARY_SENSITIVITY_TARGETS,
        "selected_algorithm": best_algorithm,
        "predictor_set": MODEL_NAME_FOR_OUTPUT,
        "feature_count": len(FULL_FEATURES),
        "selected_features": FULL_FEATURES,
        "best_hyperparameters": best_params,
        "final_tuning_cv_auc": final_tuning_auc,
        "primary_threshold": primary_threshold,
        "all_thresholds": thresholds,
        "error_figure_features": ERROR_FIGURE_FEATURES,
        "delong_comparison": "Selected algorithm versus each other candidate using training OOF predictions",
        "missing_data_strategy": MISSING_DATA_STRATEGY,
        "software": {
            "python": os.sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    with open(
        OUT_DIR / "ANALYSIS_METADATA.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Selected algorithm:", best_algorithm)
    print("Predictor set:", MODEL_NAME_FOR_OUTPUT)
    print("Number of predictors:", len(FULL_FEATURES))
    print("Primary threshold:", primary_threshold)
    print("Output directory:", OUT_DIR)


if __name__ == "__main__":
    main()
