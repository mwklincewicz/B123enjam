"""Locked final SEO Top-5 model pipeline.

This script reproduces the final 37-feature modelling workflow without tuning.
It:
1. loads and cleans the raw dataset;
2. creates the Top-5 target;
3. reproduces the fixed query-grouped development/test split;
4. engineers the locked 37 predictors;
5. compares fixed Dummy, Logistic Regression, Random Forest and XGBoost
   specifications with 5-fold StratifiedGroupKFold;
6. fits the locked final Random Forest on all development data;
7. reports comparative hold-out results and saves the fitted model.

Research-question analyses are kept in rq_analysis.py.
Run from the project root with: python main.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = Path("data/RAT_Search_Results_Trends_SEO_2025.csv")
OUTPUT_DIR = Path("outputs/final_model")

QUERY_COLUMN = "query"
ENGINE_COLUMN = "searchengine"
POSITION_COLUMN = "position"
TARGET_COLUMN = "high_ranking"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CV_SPLITS = 5
CLASSIFICATION_THRESHOLD = 0.50

EXPECTED_USABLE_ROWS = 85_872
EXPECTED_TOTAL_QUERIES = 1_519
EXPECTED_DEVELOPMENT_ROWS = 68_667
EXPECTED_TEST_ROWS = 17_205
EXPECTED_DEVELOPMENT_QUERIES = 1_215
EXPECTED_TEST_QUERIES = 304

SOCIAL_DOMAINS = {
    "instagram.com",
    "facebook.com",
    "tiktok.com",
    "youtube.com",
    "youtu.be",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "pinterest.com",
    "reddit.com",
}

FINAL_FEATURES = [
    # Technical / binary
    "https",
    "robots_txt",
    "sitemap",
    "viewport",
    "wordpress",
    "h1_present",
    "canonical_present",
    # Links / performance
    "log_external_link_count",
    "log_internal_link_count",
    "log_nofollow",
    "log_loading_time",
    "log_url_length",
    # Retained dataset scores
    "content_length_score",
    "link_quality_score",
    "navigation_score",
    # Technology tools
    "analytics_tool_count",
    "ads_tool_count",
    "seo_tool_count",
    # Structured data
    "micros_count",
    "has_json_ld",
    "has_schema_org",
    "has_rel_license",
    # URL structure
    "url_path_depth",
    "url_parameter_count",
    "url_digit_count",
    "url_slug_length",
    "url_is_homepage",
    "is_social_platform",
    # Page metadata
    "page_title_length",
    "meta_description_length",
    # Keyword locations
    "keyword_in_url",
    "keyword_in_title",
    "keyword_in_description",
    "keyword_in_headers",
    # Social metadata
    "og_tag_count",
    "twitter_tag_count",
    # Context variable
    ENGINE_COLUMN,
]

FEATURE_GROUPS = {
    "technical_binary": [
        "https",
        "robots_txt",
        "sitemap",
        "viewport",
        "wordpress",
        "h1_present",
        "canonical_present",
    ],
    "links_performance": [
        "log_external_link_count",
        "log_internal_link_count",
        "log_nofollow",
        "log_loading_time",
        "log_url_length",
    ],
    "retained_scores": [
        "content_length_score",
        "link_quality_score",
        "navigation_score",
    ],
    "technology_tools": [
        "analytics_tool_count",
        "ads_tool_count",
        "seo_tool_count",
    ],
    "structured_data": [
        "micros_count",
        "has_json_ld",
        "has_schema_org",
        "has_rel_license",
    ],
    "url_structure": [
        "url_path_depth",
        "url_parameter_count",
        "url_digit_count",
        "url_slug_length",
        "url_is_homepage",
        "is_social_platform",
    ],
    "title_description": [
        "page_title_length",
        "meta_description_length",
    ],
    "keywords": [
        "keyword_in_url",
        "keyword_in_title",
        "keyword_in_description",
        "keyword_in_headers",
    ],
    "social_metadata": [
        "og_tag_count",
        "twitter_tag_count",
    ],
    "searchengine_context": [ENGINE_COLUMN],
}

CATEGORY_FEATURES = [
    "category_content_quality",
    "category_meta_elements",
    "category_technical_seo",
    "category_user_experience",
]

RF_PARAMETERS = {
    "n_estimators": 400,
    "max_depth": 30,
    "min_samples_split": 2,
    "min_samples_leaf": 3,
    "max_features": 0.2,
    "max_samples": None,
    "class_weight": "balanced",
}

XGB_PARAMETERS = {
    "n_estimators": 700,
    "max_depth": 9,
    "learning_rate": 0.02,
    "min_child_weight": 1,
    "gamma": 0.0,
    "subsample": 0.75,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 20.0,
    "scale_pos_weight": 3.0,
}

MODEL_ORDER = ["dummy", "logistic_regression", "random_forest", "xgboost"]
LOCKED_FINAL_MODEL = "random_forest"


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def normalise_column_names(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return data


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def parse_jsonish(value: Any, default: Any) -> Any:
    if is_missing(value):
        return default
    if isinstance(value, (list, dict)):
        return value

    text = str(value).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return default


def parse_list(value: Any) -> list[str]:
    parsed = parse_jsonish(value, [])
    if not isinstance(parsed, list):
        return []
    return [
        re.sub(r"\s+", " ", str(item).strip())
        for item in parsed
        if not is_missing(item)
    ]


def parse_dict(value: Any) -> dict[str, Any]:
    parsed = parse_jsonish(value, {})
    return parsed if isinstance(parsed, dict) else {}


def to_numeric_with_boole(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map({"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0})
    return numeric.fillna(mapped)


def extract_domain(url: Any) -> str:
    if is_missing(url):
        return ""
    try:
        domain = urlparse(str(url)).netloc.lower()
    except ValueError:
        return ""
    if domain.startswith("www."):
        domain = domain[4:]
    return domain.split(":")[0]


def is_social_domain(domain: str) -> int:
    if not domain:
        return 0
    return int(
        any(domain == item or domain.endswith("." + item) for item in SOCIAL_DOMAINS)
    )


def contains_item(items: list[str], expected: str) -> int:
    expected_lower = expected.lower()
    return int(expected_lower in {item.lower() for item in items})


def contains_fragment(items: list[str], fragment: str) -> int:
    fragment_lower = fragment.lower()
    return int(any(fragment_lower in item.lower() for item in items))


def feature_group_for(feature: str) -> str:
    for group, members in FEATURE_GROUPS.items():
        if feature in members:
            return group
    return "unassigned"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


# =============================================================================
# DATA LOADING, CLEANING AND SPLITTING
# =============================================================================


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    return normalise_column_names(pd.read_csv(path, low_memory=False))


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        QUERY_COLUMN,
        ENGINE_COLUMN,
        POSITION_COLUMN,
        "seo_score",
        "url",
        "title_text",
        "description_text",
        "tools_ads",
        "tools_analytics",
        "tools_seo",
        "micros",
        "keyword_optimization_reasons",
        "social_tags",
        "https",
        "robots_txt",
        "sitemap",
        "viewport",
        "wordpress",
        "h1",
        "canonical",
        "external_link_count",
        "internal_link_count",
        "nofollow",
        "loading_time",
        "url_length",
        "content_length_score",
        "link_quality_score",
        "navigation_score",
        *CATEGORY_FEATURES,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise KeyError("Required columns are missing: " + ", ".join(missing))

    data = raw.copy()
    data[QUERY_COLUMN] = data[QUERY_COLUMN].astype("string").str.strip()
    data[ENGINE_COLUMN] = data[ENGINE_COLUMN].astype("string").str.strip()
    data[POSITION_COLUMN] = pd.to_numeric(data[POSITION_COLUMN], errors="coerce")

    status = data["seo_score"].astype("string").str.strip().str.lower()
    unusable_status = status.isin({"excluded", "error"})

    core_columns = [
        "content_length_score",
        "link_quality_score",
        "navigation_score",
        "external_link_count",
        "internal_link_count",
        "nofollow",
        "loading_time",
        "url_length",
    ]
    all_core_missing = (
        data[core_columns].apply(pd.to_numeric, errors="coerce").isna().all(axis=1)
    )
    unusable = unusable_status | all_core_missing

    summary: dict[str, Any] = {
        "rows_raw": len(data),
        "rows_status_excluded": int(status.eq("excluded").sum()),
        "rows_status_error": int(status.eq("error").sum()),
        "rows_removed_unusable": int(unusable.sum()),
    }

    data = data.loc[~unusable].copy()
    data = data.dropna(subset=[QUERY_COLUMN, ENGINE_COLUMN, POSITION_COLUMN])
    data = data[
        data[QUERY_COLUMN].ne("")
        & data[ENGINE_COLUMN].ne("")
        & data[POSITION_COLUMN].between(1, 20)
    ].copy()

    for column in ["https", "robots_txt", "sitemap", "viewport", "wordpress"]:
        data[column] = to_numeric_with_boole(data[column])

    numeric_columns = [
        "h1",
        "canonical",
        "external_link_count",
        "internal_link_count",
        "nofollow",
        "loading_time",
        "url_length",
        "content_length_score",
        "link_quality_score",
        "navigation_score",
        *CATEGORY_FEATURES,
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data.loc[data["loading_time"] < 0, "loading_time"] = np.nan
    data.loc[~data["robots_txt"].isin([0, 1]), "robots_txt"] = np.nan
    for column in [
        "h1",
        "canonical",
        "external_link_count",
        "internal_link_count",
        "nofollow",
        "url_length",
    ]:
        data.loc[data[column] < 0, column] = np.nan

    data[TARGET_COLUMN] = (data[POSITION_COLUMN] <= 5).astype(int)
    data = data.reset_index(drop=True)

    summary.update(
        {
            "rows_usable": len(data),
            "queries_usable": int(data[QUERY_COLUMN].nunique()),
            "top5_prevalence": float(data[TARGET_COLUMN].mean()),
        }
    )
    return data, summary


def make_grouped_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    development_index, test_index = next(
        splitter.split(data, y=data[TARGET_COLUMN], groups=data[QUERY_COLUMN])
    )

    development = data.iloc[development_index].copy()
    test = data.iloc[test_index].copy()

    overlap = set(development[QUERY_COLUMN]) & set(test[QUERY_COLUMN])
    if overlap:
        raise RuntimeError("Query leakage detected between development and test.")
    return development, test


def make_cv_splits(data: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = StratifiedGroupKFold(
        n_splits=N_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return list(
        cv.split(data, y=data[TARGET_COLUMN], groups=data[QUERY_COLUMN])
    )


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["h1_present"] = np.where(
        data["h1"].notna(), (data["h1"] > 0).astype(float), np.nan
    )
    data["canonical_present"] = np.where(
        data["canonical"].notna(), (data["canonical"] > 0).astype(float), np.nan
    )

    for source in [
        "external_link_count",
        "internal_link_count",
        "nofollow",
        "loading_time",
        "url_length",
    ]:
        data[f"log_{source}"] = np.log1p(data[source])

    data["analytics_tool_count"] = data["tools_analytics"].map(
        lambda value: len(set(parse_list(value)))
    )
    data["ads_tool_count"] = data["tools_ads"].map(
        lambda value: len(set(parse_list(value)))
    )
    data["seo_tool_count"] = data["tools_seo"].map(
        lambda value: len(set(parse_list(value)))
    )

    micros = data["micros"].map(parse_list)
    data["micros_count"] = micros.map(lambda items: len(set(items)))
    data["has_json_ld"] = micros.map(
        lambda items: contains_fragment(items, "json-ld")
    )
    data["has_schema_org"] = micros.map(
        lambda items: contains_fragment(items, "schema.org")
    )
    data["has_rel_license"] = micros.map(
        lambda items: contains_item(items, "rel-license")
    )

    url_rows: list[dict[str, float]] = []
    for value in data["url"]:
        if is_missing(value):
            url_rows.append(
                {
                    "url_path_depth": np.nan,
                    "url_parameter_count": np.nan,
                    "url_digit_count": np.nan,
                    "url_slug_length": np.nan,
                    "url_is_homepage": np.nan,
                    "is_social_platform": np.nan,
                }
            )
            continue

        text = str(value)
        try:
            parsed = urlparse(text)
        except ValueError:
            parsed = urlparse("")

        path_parts = [part for part in parsed.path.split("/") if part]
        parameters = parse_qsl(parsed.query, keep_blank_values=True)
        domain = extract_domain(text)
        url_rows.append(
            {
                "url_path_depth": float(len(path_parts)),
                "url_parameter_count": float(len(parameters)),
                "url_digit_count": float(sum(character.isdigit() for character in text)),
                "url_slug_length": float(len(path_parts[-1]) if path_parts else 0),
                "url_is_homepage": float(len(path_parts) == 0),
                "is_social_platform": float(is_social_domain(domain)),
            }
        )
    data = pd.concat([data, pd.DataFrame(url_rows, index=data.index)], axis=1)

    data["page_title_length"] = (
        data["title_text"].fillna("").astype(str).str.len().astype(float)
    )
    data["meta_description_length"] = (
        data["description_text"].fillna("").astype(str).str.len().astype(float)
    )

    reasons = data["keyword_optimization_reasons"].map(parse_list)
    data["keyword_in_url"] = reasons.map(
        lambda items: contains_item(items, "Keyword in URL")
    )
    data["keyword_in_title"] = reasons.map(
        lambda items: contains_item(items, "Keyword in title")
    )
    data["keyword_in_description"] = reasons.map(
        lambda items: contains_item(items, "Keyword in description")
    )
    data["keyword_in_headers"] = reasons.map(
        lambda items: contains_item(items, "Keyword in headers")
    )

    social_tags = data["social_tags"].map(parse_dict)

    def tag_list(value: dict[str, Any], key: str) -> list[str]:
        tags = value.get(key, []) if isinstance(value, dict) else []
        return tags if isinstance(tags, list) else []

    og_tags = social_tags.map(lambda value: tag_list(value, "og_tags"))
    twitter_tags = social_tags.map(lambda value: tag_list(value, "twitter_tags"))
    data["og_tag_count"] = og_tags.map(
        lambda items: len({str(item).strip() for item in items if not is_missing(item)})
    )
    data["twitter_tag_count"] = twitter_tags.map(
        lambda items: len({str(item).strip() for item in items if not is_missing(item)})
    )

    missing = sorted(set(FINAL_FEATURES) - set(data.columns))
    if missing:
        raise KeyError("Engineered features are missing: " + ", ".join(missing))
    if len(FINAL_FEATURES) != 37:
        raise AssertionError(f"Expected 37 features, found {len(FINAL_FEATURES)}.")
    if len(set(FINAL_FEATURES)) != len(FINAL_FEATURES):
        raise AssertionError("Duplicate predictors found in FINAL_FEATURES.")

    return data


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw = load_raw_data()
    cleaned, cleaning_summary = clean_data(raw)
    development_raw, test_raw = make_grouped_split(cleaned)
    development = engineer_features(development_raw)
    test = engineer_features(test_raw)

    split_summary = {
        **cleaning_summary,
        "rows_development": len(development),
        "rows_test": len(test),
        "queries_development": int(development[QUERY_COLUMN].nunique()),
        "queries_test": int(test[QUERY_COLUMN].nunique()),
        "query_overlap_count": len(
            set(development[QUERY_COLUMN]) & set(test[QUERY_COLUMN])
        ),
        "top5_prevalence_development": float(development[TARGET_COLUMN].mean()),
        "top5_prevalence_test": float(test[TARGET_COLUMN].mean()),
        "conceptual_feature_count": len(FINAL_FEATURES),
    }
    return development, test, split_summary


# =============================================================================
# MODELS AND EVALUATION
# =============================================================================


def make_one_hot_encoder(drop_first: bool) -> OneHotEncoder:
    return OneHotEncoder(
        handle_unknown="ignore",
        drop="first" if drop_first else None,
    )


def build_preprocessor(
    features: list[str],
    scale_numeric: bool,
    drop_first: bool,
) -> ColumnTransformer:
    categorical = [feature for feature in features if feature == ENGINE_COLUMN]
    numeric = [feature for feature in features if feature != ENGINE_COLUMN]

    transformers: list[tuple[str, BaseEstimator, list[str]]] = []
    numeric_steps: list[tuple[str, BaseEstimator]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers.append(("numeric", Pipeline(numeric_steps), numeric))

    if categorical:
        transformers.append(
            (
                "searchengine",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", make_one_hot_encoder(drop_first)),
                    ]
                ),
                categorical,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(
    model_name: str,
    y_train: pd.Series,
    features: list[str] | None = None,
) -> Pipeline:
    features = FINAL_FEATURES if features is None else features

    if model_name == "dummy":
        estimator: BaseEstimator = DummyClassifier(strategy="prior")
        scale_numeric = False
        drop_first = False
    elif model_name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=3_000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
        scale_numeric = True
        drop_first = True
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **RF_PARAMETERS,
        )
        scale_numeric = False
        drop_first = False
    elif model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **XGB_PARAMETERS,
        )
        scale_numeric = False
        drop_first = False
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        [
            (
                "preprocess",
                build_preprocessor(
                    features=features,
                    scale_numeric=scale_numeric,
                    drop_first=drop_first,
                ),
            ),
            ("model", estimator),
        ]
    )


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def cross_validate_model(
    model_name: str,
    data: pd.DataFrame,
    features: list[str],
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold, (train_index, valid_index) in enumerate(fold_indices, start=1):
        train = data.iloc[train_index]
        valid = data.iloc[valid_index]

        model = build_model_pipeline(model_name, train[TARGET_COLUMN], features)
        model.fit(train[features], train[TARGET_COLUMN])
        probabilities = model.predict_proba(valid[features])[:, 1]
        rows.append(
            {
                "model": model_name,
                "fold": fold,
                "rows_train": len(train),
                "rows_valid": len(valid),
                "queries_train": int(train[QUERY_COLUMN].nunique()),
                "queries_valid": int(valid[QUERY_COLUMN].nunique()),
                **calculate_metrics(valid[TARGET_COLUMN], probabilities),
            }
        )
    return rows


def summarise_cv(fold_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
    ]
    rows: list[dict[str, Any]] = []
    for model_name, group in fold_results.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model_name}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["pr_auc_mean", "roc_auc_mean"], ascending=False
    )


def evaluate_by_searchengine(
    data: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    evaluation = data[[ENGINE_COLUMN, TARGET_COLUMN]].copy()
    evaluation["probability"] = probabilities
    rows: list[dict[str, Any]] = []
    for engine, group in evaluation.groupby(ENGINE_COLUMN, dropna=False):
        rows.append(
            {
                ENGINE_COLUMN: str(engine),
                "rows": len(group),
                "top5_prevalence": float(group[TARGET_COLUMN].mean()),
                **calculate_metrics(
                    group[TARGET_COLUMN], group["probability"].to_numpy()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(ENGINE_COLUMN)


# =============================================================================
# OUTPUT HELPERS
# =============================================================================


def save_feature_configuration() -> None:
    feature_rows = [
        {
            "feature": feature,
            "group": feature_group_for(feature),
            "representation": (
                "categorical"
                if feature == ENGINE_COLUMN
                else "log_transformed"
                if feature.startswith("log_")
                else "absolute"
            ),
        }
        for feature in FINAL_FEATURES
    ]
    pd.DataFrame(feature_rows).to_csv(
        OUTPUT_DIR / "final_feature_list.csv", index=False
    )
    with (OUTPUT_DIR / "feature_groups.json").open("w", encoding="utf-8") as file:
        json.dump(FEATURE_GROUPS, file, indent=2)
    with (OUTPUT_DIR / "locked_parameters.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "random_forest": RF_PARAMETERS,
                "xgboost": XGB_PARAMETERS,
                "locked_final_model": LOCKED_FINAL_MODEL,
            },
            file,
            indent=2,
        )


def validate_expected_split(summary: dict[str, Any]) -> None:
    expected = {
        "rows_usable": EXPECTED_USABLE_ROWS,
        "queries_usable": EXPECTED_TOTAL_QUERIES,
        "rows_development": EXPECTED_DEVELOPMENT_ROWS,
        "rows_test": EXPECTED_TEST_ROWS,
        "queries_development": EXPECTED_DEVELOPMENT_QUERIES,
        "queries_test": EXPECTED_TEST_QUERIES,
        "query_overlap_count": 0,
    }
    for key, expected_value in expected.items():
        if summary[key] != expected_value:
            raise RuntimeError(
                f"Unexpected split value for {key}: {summary[key]} "
                f"(expected {expected_value})."
            )


# =============================================================================
# MAIN WORKFLOW
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    development, test, split_summary = prepare_data()
    validate_expected_split(split_summary)
    save_feature_configuration()

    print("\nDATA AND SPLIT")
    print("=" * 80)
    for key, value in split_summary.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        elif isinstance(value, int):
            print(f"- {key}: {value:,}")
        else:
            print(f"- {key}: {value}")

    folds = make_cv_splits(development)
    all_rows: list[dict[str, Any]] = []

    print("\nLOCKED MODEL COMPARISON")
    print("=" * 80)
    for model_name in MODEL_ORDER:
        if model_name == "xgboost" and XGBClassifier is None:
            print("- xgboost skipped because the package is unavailable")
            continue
        print(f"- running {model_name}")
        all_rows.extend(
            cross_validate_model(model_name, development, FINAL_FEATURES, folds)
        )

    fold_results = pd.DataFrame(all_rows)
    comparison = summarise_cv(fold_results)
    fold_results.to_csv(OUTPUT_DIR / "model_comparison_cv_folds.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "model_comparison_cv.csv", index=False)

    print("\nDEVELOPMENT RESULTS")
    print("=" * 80)
    display_columns = [
        "model",
        "pr_auc_mean",
        "pr_auc_std",
        "roc_auc_mean",
        "f1_mean",
        "balanced_accuracy_mean",
        "precision_mean",
        "recall_mean",
    ]
    print(
        comparison[display_columns].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )

    final_model = build_model_pipeline(
        LOCKED_FINAL_MODEL, development[TARGET_COLUMN], FINAL_FEATURES
    )
    final_model.fit(development[FINAL_FEATURES], development[TARGET_COLUMN])

    test_probabilities = final_model.predict_proba(test[FINAL_FEATURES])[:, 1]
    test_predictions = (test_probabilities >= CLASSIFICATION_THRESHOLD).astype(int)
    test_metrics = calculate_metrics(test[TARGET_COLUMN], test_probabilities)
    test_metrics["naive_pr_auc"] = float(test[TARGET_COLUMN].mean())

    confusion = pd.DataFrame(
        confusion_matrix(test[TARGET_COLUMN], test_predictions),
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    confusion.to_csv(OUTPUT_DIR / "test_confusion_matrix.csv")

    test_by_engine = evaluate_by_searchengine(test, test_probabilities)
    test_by_engine.to_csv(OUTPUT_DIR / "test_metrics_by_searchengine.csv", index=False)

    prediction_columns = [
        column
        for column in [
            "result",
            QUERY_COLUMN,
            ENGINE_COLUMN,
            "url",
            POSITION_COLUMN,
            TARGET_COLUMN,
        ]
        if column in test.columns
    ]
    predictions = test[prediction_columns].copy()
    predictions["predicted_probability"] = test_probabilities
    predictions["predicted_class"] = test_predictions
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    transformed_names = final_model.named_steps["preprocess"].get_feature_names_out()
    pd.DataFrame({"transformed_feature": transformed_names}).to_csv(
        OUTPUT_DIR / "transformed_feature_names.csv", index=False
    )
    joblib.dump(final_model, OUTPUT_DIR / "final_model.joblib")

    locked_cv_row = comparison.loc[
        comparison["model"] == LOCKED_FINAL_MODEL
    ].iloc[0].to_dict()
    summary = {
        "status": "locked_final_model",
        "split": split_summary,
        "conceptual_feature_count": len(FINAL_FEATURES),
        "transformed_feature_count": len(transformed_names),
        "features": FINAL_FEATURES,
        "feature_groups": FEATURE_GROUPS,
        "models_compared": comparison["model"].tolist(),
        "locked_final_model": LOCKED_FINAL_MODEL,
        "random_forest_parameters": RF_PARAMETERS,
        "xgboost_parameters": XGB_PARAMETERS,
        "locked_model_cv_metrics": locked_cv_row,
        "holdout_metrics": test_metrics,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "note": (
            "The same hold-out split was inspected during earlier development. "
            "Model and feature decisions are therefore based on development CV; "
            "the hold-out is retained as comparative evidence."
        ),
    }
    with (OUTPUT_DIR / "final_summary.json").open("w", encoding="utf-8") as file:
        json.dump(json_safe(summary), file, indent=2)

    print("\nLOCKED RANDOM FOREST HOLD-OUT RESULTS")
    print("=" * 80)
    for metric, value in test_metrics.items():
        print(f"- {metric}: {value:.4f}")
    print("\nResults by search engine")
    print(
        test_by_engine.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print(f"\nSaved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
