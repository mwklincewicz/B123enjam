"""Research-question analyses for the locked final SEO model.

Run main.py first. This script does not tune models or change the locked
37-feature specification. It creates the remaining analyses for:
- RQ1: predictive value of the four broad SEO category scores;
- RQ2: overall predictive performance;
- RQ3: Random Forest/XGBoost/baseline comparison;
- RQ4: individual feature importance and descriptive direction;
- RQ5: performance and feature importance by search engine.

Run from the project root with: python rq_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

import main as locked


OUTPUT_DIR = Path("outputs/research_questions")
FINAL_OUTPUT_DIR = locked.OUTPUT_DIR
PERMUTATION_REPEATS = 3

CATEGORY_LABELS = {
    "category_content_quality": "Content Quality",
    "category_meta_elements": "Meta Elements",
    "category_technical_seo": "Technical SEO",
    "category_user_experience": "User Experience",
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================


def require_main_outputs() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    comparison_path = FINAL_OUTPUT_DIR / "model_comparison_cv.csv"
    summary_path = FINAL_OUTPUT_DIR / "final_summary.json"
    engine_path = FINAL_OUTPUT_DIR / "test_metrics_by_searchengine.csv"

    missing = [
        str(path)
        for path in [comparison_path, summary_path, engine_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Run main.py first. Missing output files:\n- " + "\n- ".join(missing)
        )

    comparison = pd.read_csv(comparison_path)
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    test_by_engine = pd.read_csv(engine_path)
    return comparison, summary, test_by_engine


def summarise_rows(rows: list[dict[str, Any]], group_columns: list[str]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    metric_columns = [
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
    ]

    records: list[dict[str, Any]] = []
    for keys, group in data.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        for metric in metric_columns:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=1))
        records.append(record)
    return pd.DataFrame(records)


def importance_summary(detail: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = (
        detail.groupby(group_columns, as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_min=("importance", "min"),
            importance_max=("importance", "max"),
        )
        .sort_values("importance_mean", ascending=False)
    )
    return summary


def print_table(title: str, frame: pd.DataFrame, columns: list[str]) -> None:
    print(f"\n{title}")
    print("=" * 80)
    print(
        frame[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


# =============================================================================
# RQ1: BROAD SEO CATEGORIES
# =============================================================================


def run_rq1_category_analysis(
    development: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specifications = {
        "all_four_categories": locked.CATEGORY_FEATURES,
        **{
            f"only_{feature}": [feature]
            for feature in locked.CATEGORY_FEATURES
        },
    }

    performance_rows: list[dict[str, Any]] = []
    for specification, features in specifications.items():
        for fold, (train_index, valid_index) in enumerate(folds, start=1):
            train = development.iloc[train_index]
            valid = development.iloc[valid_index]
            model = locked.build_model_pipeline(
                "random_forest", train[locked.TARGET_COLUMN], features
            )
            model.fit(train[features], train[locked.TARGET_COLUMN])
            probabilities = model.predict_proba(valid[features])[:, 1]
            performance_rows.append(
                {
                    "specification": specification,
                    "fold": fold,
                    "n_features": len(features),
                    **locked.calculate_metrics(
                        valid[locked.TARGET_COLUMN], probabilities
                    ),
                }
            )

    performance = summarise_rows(performance_rows, ["specification"])
    feature_counts = {
        specification: len(features)
        for specification, features in specifications.items()
    }
    performance.insert(
        1,
        "n_features",
        performance["specification"].map(feature_counts),
    )
    performance = performance.sort_values("pr_auc_mean", ascending=False)
    performance.to_csv(OUTPUT_DIR / "rq1_category_performance.csv", index=False)

    importance_rows: list[dict[str, Any]] = []
    for fold, (train_index, valid_index) in enumerate(folds, start=1):
        train = development.iloc[train_index]
        valid = development.iloc[valid_index]
        model = locked.build_model_pipeline(
            "random_forest", train[locked.TARGET_COLUMN], locked.CATEGORY_FEATURES
        )
        model.fit(
            train[locked.CATEGORY_FEATURES],
            train[locked.TARGET_COLUMN],
        )
        result = permutation_importance(
            model,
            valid[locked.CATEGORY_FEATURES],
            valid[locked.TARGET_COLUMN],
            scoring="average_precision",
            n_repeats=PERMUTATION_REPEATS,
            random_state=locked.RANDOM_STATE + fold,
            n_jobs=1,
        )
        for feature_index, feature in enumerate(locked.CATEGORY_FEATURES):
            for repeat in range(PERMUTATION_REPEATS):
                importance_rows.append(
                    {
                        "fold": fold,
                        "repeat": repeat + 1,
                        "feature": feature,
                        "category": CATEGORY_LABELS[feature],
                        "importance": float(result.importances[feature_index, repeat]),
                    }
                )

    importance_detail = pd.DataFrame(importance_rows)
    importance_detail.to_csv(
        OUTPUT_DIR / "rq1_category_importance_detail.csv", index=False
    )
    importance = importance_summary(
        importance_detail, ["feature", "category"]
    )
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance.to_csv(OUTPUT_DIR / "rq1_category_importance.csv", index=False)
    return performance, importance


# =============================================================================
# RQ4 AND RQ5: LOCKED MODEL ANALYSES
# =============================================================================


def grouped_permutation_importance(
    model,
    validation: pd.DataFrame,
    features: list[str],
    feature_groups: dict[str, list[str]],
    fold: int,
) -> list[dict[str, Any]]:
    baseline_probabilities = model.predict_proba(validation[features])[:, 1]
    baseline_score = average_precision_score(
        validation[locked.TARGET_COLUMN], baseline_probabilities
    )

    rows: list[dict[str, Any]] = []
    for group_name, group_features in feature_groups.items():
        present_features = [feature for feature in group_features if feature in features]
        if not present_features:
            continue
        for repeat in range(PERMUTATION_REPEATS):
            rng = np.random.default_rng(
                locked.RANDOM_STATE + fold * 100 + repeat
            )
            order = rng.permutation(len(validation))
            permuted = validation[features].copy()
            for feature in present_features:
                permuted[feature] = validation[feature].iloc[order].to_numpy()
            probability = model.predict_proba(permuted)[:, 1]
            permuted_score = average_precision_score(
                validation[locked.TARGET_COLUMN], probability
            )
            rows.append(
                {
                    "fold": fold,
                    "repeat": repeat + 1,
                    "group": group_name,
                    "importance": float(baseline_score - permuted_score),
                }
            )
    return rows


def run_locked_model_analyses(
    development: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_importance_rows: list[dict[str, Any]] = []
    group_importance_rows: list[dict[str, Any]] = []
    engine_metric_rows: list[dict[str, Any]] = []
    engine_importance_rows: list[dict[str, Any]] = []

    for fold, (train_index, valid_index) in enumerate(folds, start=1):
        train = development.iloc[train_index]
        valid = development.iloc[valid_index]

        model = locked.build_model_pipeline(
            locked.LOCKED_FINAL_MODEL,
            train[locked.TARGET_COLUMN],
            locked.FINAL_FEATURES,
        )
        model.fit(train[locked.FINAL_FEATURES], train[locked.TARGET_COLUMN])

        # RQ4: individual out-of-fold permutation importance.
        result = permutation_importance(
            model,
            valid[locked.FINAL_FEATURES],
            valid[locked.TARGET_COLUMN],
            scoring="average_precision",
            n_repeats=PERMUTATION_REPEATS,
            random_state=locked.RANDOM_STATE + fold,
            n_jobs=1,
        )
        for feature_index, feature in enumerate(locked.FINAL_FEATURES):
            for repeat in range(PERMUTATION_REPEATS):
                overall_importance_rows.append(
                    {
                        "fold": fold,
                        "repeat": repeat + 1,
                        "feature": feature,
                        "group": locked.feature_group_for(feature),
                        "importance": float(result.importances[feature_index, repeat]),
                    }
                )

        group_importance_rows.extend(
            grouped_permutation_importance(
                model,
                valid,
                locked.FINAL_FEATURES,
                locked.FEATURE_GROUPS,
                fold,
            )
        )

        # RQ5: pooled locked model evaluated within each search engine.
        for engine, engine_valid in valid.groupby(locked.ENGINE_COLUMN):
            probabilities = model.predict_proba(
                engine_valid[locked.FINAL_FEATURES]
            )[:, 1]
            engine_metric_rows.append(
                {
                    "fold": fold,
                    "searchengine": str(engine),
                    "rows": len(engine_valid),
                    "top5_prevalence": float(
                        engine_valid[locked.TARGET_COLUMN].mean()
                    ),
                    **locked.calculate_metrics(
                        engine_valid[locked.TARGET_COLUMN], probabilities
                    ),
                }
            )

            engine_result = permutation_importance(
                model,
                engine_valid[locked.FINAL_FEATURES],
                engine_valid[locked.TARGET_COLUMN],
                scoring="average_precision",
                n_repeats=PERMUTATION_REPEATS,
                random_state=locked.RANDOM_STATE + fold,
                n_jobs=1,
            )
            for feature_index, feature in enumerate(locked.FINAL_FEATURES):
                if feature == locked.ENGINE_COLUMN:
                    continue
                for repeat in range(PERMUTATION_REPEATS):
                    engine_importance_rows.append(
                        {
                            "fold": fold,
                            "repeat": repeat + 1,
                            "searchengine": str(engine),
                            "feature": feature,
                            "group": locked.feature_group_for(feature),
                            "importance": float(
                                engine_result.importances[feature_index, repeat]
                            ),
                        }
                    )

        print(f"- locked-model analysis fold {fold}/{locked.N_CV_SPLITS} completed")

    overall_detail = pd.DataFrame(overall_importance_rows)
    overall_detail.to_csv(
        OUTPUT_DIR / "rq4_feature_importance_detail.csv", index=False
    )
    overall_importance = importance_summary(
        overall_detail, ["feature", "group"]
    )
    overall_importance["rank"] = np.arange(1, len(overall_importance) + 1)
    overall_importance.to_csv(
        OUTPUT_DIR / "rq4_feature_importance.csv", index=False
    )

    group_detail = pd.DataFrame(group_importance_rows)
    group_detail.to_csv(
        OUTPUT_DIR / "rq4_feature_group_importance_detail.csv", index=False
    )
    group_importance = importance_summary(group_detail, ["group"])
    group_importance["rank"] = np.arange(1, len(group_importance) + 1)
    group_importance.to_csv(
        OUTPUT_DIR / "rq4_feature_group_importance.csv", index=False
    )

    engine_metrics_detail = pd.DataFrame(engine_metric_rows)
    engine_metrics_detail.to_csv(
        OUTPUT_DIR / "rq5_performance_by_engine_folds.csv", index=False
    )
    engine_metrics = summarise_rows(
        engine_metric_rows, ["searchengine"]
    ).sort_values("pr_auc_mean", ascending=False)
    engine_metrics.to_csv(
        OUTPUT_DIR / "rq5_performance_by_engine.csv", index=False
    )

    engine_importance_detail = pd.DataFrame(engine_importance_rows)
    engine_importance_detail.to_csv(
        OUTPUT_DIR / "rq5_feature_importance_by_engine_detail.csv", index=False
    )
    engine_importance = importance_summary(
        engine_importance_detail, ["searchengine", "feature", "group"]
    )
    engine_importance["rank_within_engine"] = (
        engine_importance.groupby("searchengine")["importance_mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    engine_importance = engine_importance.sort_values(
        ["searchengine", "rank_within_engine"]
    )
    engine_importance.to_csv(
        OUTPUT_DIR / "rq5_feature_importance_by_engine.csv", index=False
    )

    top10 = engine_importance.loc[
        engine_importance["rank_within_engine"] <= 10
    ].copy()
    top10.to_csv(
        OUTPUT_DIR / "rq5_top10_features_by_engine.csv", index=False
    )

    return overall_importance, group_importance, engine_metrics, engine_importance


# =============================================================================
# RQ4: DIRECTION / DESCRIPTIVE ASSOCIATION
# =============================================================================


def create_direction_table(development: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    top5 = development[locked.TARGET_COLUMN] == 1

    for feature in locked.FINAL_FEATURES:
        if feature == locked.ENGINE_COLUMN:
            continue
        values = pd.to_numeric(development[feature], errors="coerce")
        valid = values.notna()
        top_values = values[valid & top5]
        other_values = values[valid & ~top5]
        unique_values = set(values[valid].unique().tolist())
        is_binary = unique_values.issubset({0.0, 1.0}) and len(unique_values) <= 2

        correlation = np.nan
        if values[valid].nunique() > 1:
            correlation = values[valid].corr(
                development.loc[valid, locked.TARGET_COLUMN], method="spearman"
            )

        difference = float(top_values.mean() - other_values.mean())
        rows.append(
            {
                "feature": feature,
                "group": locked.feature_group_for(feature),
                "feature_type": "binary" if is_binary else "numeric",
                "non_missing_n": int(valid.sum()),
                "mean_top5": float(top_values.mean()),
                "mean_non_top5": float(other_values.mean()),
                "mean_difference_top5_minus_non_top5": difference,
                "median_top5": float(top_values.median()),
                "median_non_top5": float(other_values.median()),
                "spearman_with_top5": float(correlation),
                "direction": (
                    "higher among Top 5"
                    if difference > 0
                    else "lower among Top 5"
                    if difference < 0
                    else "no mean difference"
                ),
            }
        )

    direction = pd.DataFrame(rows)
    direction["absolute_spearman"] = direction["spearman_with_top5"].abs()
    direction = direction.sort_values("absolute_spearman", ascending=False)
    direction.to_csv(OUTPUT_DIR / "rq4_feature_direction.csv", index=False)

    engine_prevalence = (
        development.groupby(locked.ENGINE_COLUMN, as_index=False)
        .agg(
            rows=(locked.TARGET_COLUMN, "size"),
            top5_prevalence=(locked.TARGET_COLUMN, "mean"),
        )
        .sort_values(locked.ENGINE_COLUMN)
    )
    engine_prevalence.to_csv(
        OUTPUT_DIR / "rq4_searchengine_prevalence.csv", index=False
    )
    return direction


# =============================================================================
# RQ2 / RQ3 OUTPUTS AND TEXT SUMMARY
# =============================================================================


def save_rq2_rq3_outputs(
    comparison: pd.DataFrame,
    final_summary: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rq3 = comparison.copy().sort_values("pr_auc_mean", ascending=False)
    rq3.to_csv(OUTPUT_DIR / "rq3_model_comparison.csv", index=False)

    locked_cv = final_summary["locked_model_cv_metrics"]
    holdout = final_summary["holdout_metrics"]
    rq2 = pd.DataFrame(
        [
            {
                "evaluation": "development_grouped_cv",
                "model": final_summary["locked_final_model"],
                "precision": locked_cv["precision_mean"],
                "recall": locked_cv["recall_mean"],
                "f1": locked_cv["f1_mean"],
                "balanced_accuracy": locked_cv["balanced_accuracy_mean"],
                "roc_auc": locked_cv["roc_auc_mean"],
                "pr_auc": locked_cv["pr_auc_mean"],
                "pr_auc_std": locked_cv["pr_auc_std"],
                "naive_pr_auc": final_summary["split"][
                    "top5_prevalence_development"
                ],
            },
            {
                "evaluation": "comparative_holdout",
                "model": final_summary["locked_final_model"],
                "precision": holdout["precision"],
                "recall": holdout["recall"],
                "f1": holdout["f1"],
                "balanced_accuracy": holdout["balanced_accuracy"],
                "roc_auc": holdout["roc_auc"],
                "pr_auc": holdout["pr_auc"],
                "pr_auc_std": np.nan,
                "naive_pr_auc": holdout["naive_pr_auc"],
            },
        ]
    )
    rq2.to_csv(OUTPUT_DIR / "rq2_predictive_performance.csv", index=False)
    return rq2, rq3


def write_summary(
    rq1_performance: pd.DataFrame,
    rq1_importance: pd.DataFrame,
    rq2: pd.DataFrame,
    rq3: pd.DataFrame,
    rq4_importance: pd.DataFrame,
    rq4_group_importance: pd.DataFrame,
    rq5_performance: pd.DataFrame,
    rq5_importance: pd.DataFrame,
    test_by_engine: pd.DataFrame,
) -> None:
    best_single = rq1_performance.loc[
        rq1_performance["specification"].str.startswith("only_")
    ].iloc[0]
    best_category_importance = rq1_importance.iloc[0]
    best_model = rq3.iloc[0]
    top_features = rq4_importance.head(10)
    top_groups = rq4_group_importance.head(5)

    lines = [
        "LOCKED THESIS RESEARCH-QUESTION ANALYSES",
        "=" * 80,
        "",
        "RQ1 - BROAD SEO CATEGORIES",
        f"Best standalone category specification: {best_single['specification']}",
        f"Standalone category PR-AUC: {best_single['pr_auc_mean']:.4f}",
        f"Highest category permutation importance: {best_category_importance['category']}",
        "",
        "RQ2 - OVERALL PREDICTIVE PERFORMANCE",
    ]
    for _, row in rq2.iterrows():
        lines.append(
            f"{row['evaluation']}: PR-AUC={row['pr_auc']:.4f}, "
            f"ROC-AUC={row['roc_auc']:.4f}, F1={row['f1']:.4f}, "
            f"balanced accuracy={row['balanced_accuracy']:.4f}"
        )

    lines.extend(
        [
            "",
            "RQ3 - MODEL COMPARISON",
            f"Best model by grouped-CV PR-AUC: {best_model['model']} "
            f"({best_model['pr_auc_mean']:.4f})",
            "",
            "RQ4 - MOST IMPORTANT INDIVIDUAL CHARACTERISTICS",
        ]
    )
    for _, row in top_features.iterrows():
        lines.append(
            f"{int(row['rank'])}. {row['feature']} "
            f"(importance={row['importance_mean']:.4f})"
        )

    lines.extend(["", "RQ4 - MOST IMPORTANT FEATURE GROUPS"])
    for _, row in top_groups.iterrows():
        lines.append(
            f"{int(row['rank'])}. {row['group']} "
            f"(group importance={row['importance_mean']:.4f})"
        )

    lines.extend(["", "RQ5 - PERFORMANCE BY SEARCH ENGINE (DEVELOPMENT CV)"])
    for _, row in rq5_performance.iterrows():
        lines.append(
            f"{row['searchengine']}: PR-AUC={row['pr_auc_mean']:.4f} "
            f"± {row['pr_auc_std']:.4f}"
        )

    lines.extend(["", "RQ5 - COMPARATIVE HOLD-OUT PERFORMANCE"])
    for _, row in test_by_engine.sort_values("pr_auc", ascending=False).iterrows():
        lines.append(
            f"{row['searchengine']}: PR-AUC={row['pr_auc']:.4f}, "
            f"ROC-AUC={row['roc_auc']:.4f}"
        )

    lines.extend(["", "RQ5 - TOP FEATURES BY SEARCH ENGINE"])
    for engine, group in rq5_importance.groupby("searchengine"):
        names = ", ".join(group.head(5)["feature"].tolist())
        lines.append(f"{engine}: {names}")

    lines.extend(
        [
            "",
            "INTERPRETATION NOTE",
            "Permutation importance measures predictive contribution, not causal effect.",
            "Use rq4_feature_direction.csv to describe whether feature values are "
            "higher or lower among Top-5 observations.",
        ]
    )

    (OUTPUT_DIR / "research_question_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison, final_summary, test_by_engine = require_main_outputs()
    development, _, split_summary = locked.prepare_data()
    locked.validate_expected_split(split_summary)
    folds = locked.make_cv_splits(development)

    print("\nRQ1: CATEGORY ANALYSIS")
    print("=" * 80)
    rq1_performance, rq1_importance = run_rq1_category_analysis(
        development, folds
    )
    print_table(
        "RQ1 CATEGORY PERFORMANCE",
        rq1_performance,
        ["specification", "pr_auc_mean", "pr_auc_std", "roc_auc_mean"],
    )
    print_table(
        "RQ1 CATEGORY IMPORTANCE",
        rq1_importance,
        ["rank", "category", "importance_mean", "importance_std"],
    )

    rq2, rq3 = save_rq2_rq3_outputs(comparison, final_summary)

    print("\nRQ4/RQ5: LOCKED RANDOM FOREST ANALYSES")
    print("=" * 80)
    (
        rq4_importance,
        rq4_group_importance,
        rq5_performance,
        rq5_importance,
    ) = run_locked_model_analyses(development, folds)
    create_direction_table(development)

    print_table(
        "RQ4 TOP 15 FEATURES",
        rq4_importance.head(15),
        ["rank", "feature", "group", "importance_mean", "importance_std"],
    )
    print_table(
        "RQ4 FEATURE GROUPS",
        rq4_group_importance,
        ["rank", "group", "importance_mean", "importance_std"],
    )
    print_table(
        "RQ5 PERFORMANCE BY SEARCH ENGINE",
        rq5_performance,
        [
            "searchengine",
            "pr_auc_mean",
            "pr_auc_std",
            "roc_auc_mean",
            "f1_mean",
        ],
    )

    write_summary(
        rq1_performance,
        rq1_importance,
        rq2,
        rq3,
        rq4_importance,
        rq4_group_importance,
        rq5_performance,
        rq5_importance,
        test_by_engine,
    )

    print(f"\nSaved in: {OUTPUT_DIR.resolve()}")
    print("- rq1_category_performance.csv")
    print("- rq1_category_importance.csv")
    print("- rq2_predictive_performance.csv")
    print("- rq3_model_comparison.csv")
    print("- rq4_feature_importance.csv")
    print("- rq4_feature_group_importance.csv")
    print("- rq4_feature_direction.csv")
    print("- rq5_performance_by_engine.csv")
    print("- rq5_feature_importance_by_engine.csv")
    print("- rq5_top10_features_by_engine.csv")
    print("- research_question_summary.txt")


if __name__ == "__main__":
    main()
