"""Exploratory data analysis for Top-5 search-ranking classification.

Run from the project root with:

    python src/eda.py

Outputs:
    eda/figures/
    eda/tables/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "RAT_Search_Results_Trends_SEO_2025.csv"
FIGURES_PATH = PROJECT_ROOT / "eda" / "figures"
TABLES_PATH = PROJECT_ROOT / "eda" / "tables"

CATEGORY_FEATURES = [
    "category_content_quality",
    "category_meta_elements",
    "category_technical_seo",
    "category_user_experience",
]

DETAILED_FEATURES = [
    "canonical",
    "content_length_score",
    "description_score",
    "external_link_count",
    "h1",
    "heading_structure_score",
    "https",
    "image_optimization_score",
    "internal_link_count",
    "keyword_optimization_score",
    "link_quality_score",
    "loading_time",
    "navigation_score",
    "nofollow",
    "og",
    "robots_txt",
    "sitemap",
    "social_tags_score",
    "title_score",
    "url_length",
    "viewport",
    "wordpress",
]

# Retained only for EDA because these overlap with the explicit *_link_count fields.
EXPLORATORY_ONLY_FEATURES = [
    "external_links",
    "internal_links",
]

ALL_EDA_FEATURES = (
    ["seo_score"]
    + CATEGORY_FEATURES
    + DETAILED_FEATURES
    + EXPLORATORY_ONLY_FEATURES
)


# =============================================================================
# HELPERS
# =============================================================================

def save_table(df, filename, index=False):
    """Save a DataFrame to eda/tables."""
    df.to_csv(TABLES_PATH / filename, index=index)


def save_figure(filename):
    """Save and close the current Matplotlib figure."""
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / filename, dpi=300, bbox_inches="tight")
    plt.close()


def print_section(title, df):
    """Print a DataFrame as a labelled console section."""
    print(f"\n{title}\n{'-' * len(title)}")

    if df.empty:
        print("None.")
    else:
        print(df.to_string(index=False))


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data():
    """Load the dataset, construct the target, and correct invalid values."""

    raw = pd.read_csv(DATA_PATH, low_memory=False)

    raw["created_at"] = pd.to_datetime(raw["created_at"], errors="coerce")
    raw["seo_score"] = pd.to_numeric(raw["seo_score"], errors="coerce")

    # Binary target:
    # 1 = position 1-5
    # 0 = position 6-20
    raw["high_ranking"] = (raw["position"] <= 5).astype(int)

    # Missing seo_score identifies rows without a completed SEO analysis.
    raw["seo_analysis_missing"] = raw["seo_score"].isna()

    cleaned = raw.copy()

    # Invalid values identified during EDA.
    cleaned.loc[cleaned["loading_time"] < 0, "loading_time"] = np.nan
    cleaned["robots_txt"] = cleaned["robots_txt"].replace(-1, np.nan)

    cleaned["https"] = cleaned["https"].replace(
        {
            True: 1,
            False: 0,
            "True": 1,
            "False": 0,
        }
    )
    cleaned["https"] = pd.to_numeric(cleaned["https"], errors="coerce")

    # Only observations with a completed SEO analysis can contribute
    # to SEO-feature models.
    analysis = cleaned.loc[~cleaned["seo_analysis_missing"]].copy()

    return raw, cleaned, analysis


# =============================================================================
# DATA QUALITY
# =============================================================================

def dataset_quality(raw, cleaned):
    """Summarize structure, missingness, duplicates, and invalid values."""

    overview = pd.DataFrame(
        {
            "variable": raw.columns,
            "data_type": raw.dtypes.astype(str).values,
            "non_missing_n": raw.notna().sum().values,
            "missing_n": raw.isna().sum().values,
            "missing_pct": (raw.isna().mean().values * 100).round(2),
            "unique_n": raw.nunique(dropna=True).values,
        }
    ).sort_values("missing_pct", ascending=False)

    save_table(overview, "dataset_overview.csv")

    save_table(
        overview.loc[overview["missing_n"] > 0],
        "missing_values.csv",
    )

    quality_checks = {
        "Rows": len(raw),
        "Columns before target/flags": 53,
        "Completely duplicated rows": raw.duplicated().sum(),
        "Duplicated result IDs": raw.duplicated(subset="result").sum(),
        "Repeated URL occurrences": raw.duplicated(subset="url").sum(),
        "Positions outside 1-20": (
            (raw["position"] < 1) | (raw["position"] > 20)
        ).sum(),
        "Missing queries": raw["query"].isna().sum(),
        "Missing search engines": raw["searchengine"].isna().sum(),
        "Missing complete SEO analyses": raw["seo_analysis_missing"].sum(),
        "Negative loading times": (raw["loading_time"] < 0).sum(),
        "robots_txt values equal to -1": (raw["robots_txt"] == -1).sum(),
        "Missing loading times after correction": cleaned["loading_time"].isna().sum(),
    }

    quality = pd.DataFrame(
        quality_checks.items(),
        columns=["check", "count"],
    )

    save_table(quality, "data_quality_summary.csv")

    missing_by_engine = (
        raw.groupby("searchengine")["seo_analysis_missing"]
        .agg(total_rows="count", missing_analyses="sum")
        .reset_index()
    )

    missing_by_engine["missing_pct"] = (
        missing_by_engine["missing_analyses"]
        / missing_by_engine["total_rows"]
        * 100
    ).round(2)

    save_table(
        missing_by_engine,
        "missing_analysis_by_engine.csv",
    )

    exclusion_reasons = (
        raw.loc[raw["seo_analysis_missing"], "exclusion_reason"]
        .fillna("No exclusion reason recorded")
        .value_counts()
        .rename_axis("exclusion_reason")
        .reset_index(name="count")
    )

    save_table(
        exclusion_reasons,
        "exclusion_reasons.csv",
    )

    return quality


# =============================================================================
# TARGET DISTRIBUTION
# =============================================================================

def target_distribution(raw):
    """Describe the overall Top-5 classification target."""

    target = (
        raw["high_ranking"]
        .value_counts()
        .sort_index()
        .rename_axis("high_ranking")
        .reset_index(name="count")
    )

    target["percentage"] = (
        target["count"] / len(raw) * 100
    ).round(2)

    save_table(
        target,
        "target_distribution.csv",
    )

    plt.figure(figsize=(7, 5))

    ax = sns.barplot(
        data=target,
        x="high_ranking",
        y="count",
    )

    ax.set(
        title="Distribution of the Binary Ranking Target",
        xlabel="Ranking class",
        ylabel="Number of results",
    )

    ax.set_xticks(
        [0, 1],
        ["Not Top-5", "Top-5"],
    )

    save_figure("target_distribution.png")


# =============================================================================
# SEARCH-ENGINE AND QUERY STRUCTURE
# =============================================================================

def search_engine_structure(raw):
    """Assess representation and query coverage across search engines."""

    # Number of observations per search engine
    engine_counts = (
        raw["searchengine"]
        .value_counts()
        .rename_axis("searchengine")
        .reset_index(name="n_results")
    )

    engine_counts["percentage"] = (
        engine_counts["n_results"]
        / len(raw)
        * 100
    ).round(2)

    save_table(
        engine_counts,
        "search_engine_distribution.csv",
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=engine_counts,
        x="searchengine",
        y="n_results",
    )

    plt.title("Number of Results by Search Engine")
    plt.xlabel("Search engine")
    plt.ylabel("Number of results")

    save_figure("search_engine_distribution.png")

    # Unique queries per engine
    queries_per_engine = (
        raw.groupby("searchengine")["query"]
        .nunique()
        .reset_index(name="unique_queries")
    )

    save_table(
        queries_per_engine,
        "unique_queries_by_engine.csv",
    )

    # Top-5 distribution per engine
    class_by_engine = (
        pd.crosstab(
            raw["searchengine"],
            raw["high_ranking"],
        )
        .rename(
            columns={
                0: "not_top_5_n",
                1: "top_5_n",
            }
        )
        .reset_index()
    )

    class_by_engine["total_n"] = (
        class_by_engine["not_top_5_n"]
        + class_by_engine["top_5_n"]
    )

    for class_name in ["not_top_5", "top_5"]:
        class_by_engine[f"{class_name}_pct"] = (
            class_by_engine[f"{class_name}_n"]
            / class_by_engine["total_n"]
            * 100
        ).round(2)

    save_table(
        class_by_engine,
        "target_distribution_by_engine.csv",
    )

    (
        class_by_engine
        .set_index("searchengine")[
            ["not_top_5_pct", "top_5_pct"]
        ]
        .plot(
            kind="bar",
            figsize=(8, 5),
        )
    )

    plt.title("Ranking Classes by Search Engine")
    plt.xlabel("Search engine")
    plt.ylabel("Percentage of results")
    plt.xticks(rotation=0)
    plt.legend(["Not Top-5", "Top-5"])

    save_figure("target_distribution_by_engine.png")

    # Check whether queries occur across all search engines
    query_engine_presence = pd.crosstab(
        raw["query"],
        raw["searchengine"],
    )

    engine_columns = query_engine_presence.columns.tolist()

    query_engine_presence["n_engines"] = (
        query_engine_presence[engine_columns]
        .gt(0)
        .sum(axis=1)
    )

    query_coverage = (
        query_engine_presence["n_engines"]
        .value_counts()
        .sort_index()
        .rename_axis("number_of_search_engines")
        .reset_index(name="n_queries")
    )

    query_coverage["percentage"] = (
        query_coverage["n_queries"]
        / len(query_engine_presence)
        * 100
    ).round(2)

    save_table(
        query_coverage,
        "query_coverage_across_engines.csv",
    )

    save_table(
        query_engine_presence.reset_index(),
        "query_engine_presence.csv",
    )

    # Number of search results per query × engine combination
    results_per_query_engine = (
        raw.groupby(["query", "searchengine"])
        .size()
        .reset_index(name="n_results")
    )

    save_table(
        results_per_query_engine,
        "results_per_query_engine.csv",
    )

    results_per_query_summary = (
        results_per_query_engine
        .groupby("searchengine")["n_results"]
        .describe()
        .reset_index()
    )

    save_table(
        results_per_query_summary,
        "results_per_query_by_engine.csv",
    )

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=results_per_query_engine,
        x="n_results",
        hue="searchengine",
        multiple="dodge",
        discrete=True,
    )

    plt.title("Number of Results per Query by Search Engine")
    plt.xlabel("Number of results")
    plt.ylabel("Number of query-engine groups")

    save_figure("results_per_query_by_engine.png")

    query_structure = pd.DataFrame(
        {
            "measure": [
                "Unique queries overall",
                "Query-search-engine groups",
                "Search engines",
                "Mean results per query-engine group",
                "Median results per query-engine group",
                "Minimum results per query-engine group",
                "Maximum results per query-engine group",
            ],
            "value": [
                raw["query"].nunique(),
                len(results_per_query_engine),
                raw["searchengine"].nunique(),
                results_per_query_engine["n_results"].mean(),
                results_per_query_engine["n_results"].median(),
                results_per_query_engine["n_results"].min(),
                results_per_query_engine["n_results"].max(),
            ],
        }
    )

    save_table(
        query_structure,
        "query_structure_summary.csv",
    )

    return {
        "engine_counts": engine_counts,
        "queries_per_engine": queries_per_engine,
        "class_by_engine": class_by_engine,
        "query_coverage": query_coverage,
        "results_per_query": results_per_query_summary,
    }


# =============================================================================
# NUMERIC DISTRIBUTIONS
# =============================================================================

def numeric_distributions(data, features):
    """Summarize distributions, skewness, missingness, and IQR outliers."""

    summary = data[features].describe().T

    summary["missing_n"] = data[features].isna().sum()
    summary["missing_pct"] = (
        data[features].isna().mean() * 100
    ).round(2)
    summary["median"] = data[features].median()
    summary["skewness"] = data[features].skew().round(3)

    def outlier_stats(series):
        values = series.dropna()

        if values.empty:
            return pd.Series(
                {
                    "iqr_outlier_n": np.nan,
                    "iqr_outlier_pct": np.nan,
                }
            )

        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((values < lower) | (values > upper)).sum()

        return pd.Series(
            {
                "iqr_outlier_n": outliers,
                "iqr_outlier_pct": round(
                    outliers / len(values) * 100,
                    2,
                ),
            }
        )

    outliers = data[features].apply(outlier_stats).T
    summary = summary.join(outliers)

    summary = summary.sort_values(
        "skewness",
        key=lambda x: x.abs(),
        ascending=False,
    )

    save_table(
        summary,
        "numeric_feature_summary.csv",
        index=True,
    )

    # Save histograms in groups of nine.
    for page, start in enumerate(
        range(0, len(features), 9),
        start=1,
    ):
        page_features = features[start:start + 9]

        fig, axes = plt.subplots(
            3,
            3,
            figsize=(15, 12),
        )

        axes = axes.flatten()

        for ax, feature in zip(axes, page_features):
            sns.histplot(
                data=data,
                x=feature,
                bins=30,
                ax=ax,
            )

            ax.set_title(feature)
            ax.set_xlabel("")

        for ax in axes[len(page_features):]:
            ax.set_visible(False)

        fig.suptitle(
            f"Numerical Predictor Distributions - Page {page}",
            fontsize=16,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plt.savefig(
            FIGURES_PATH / f"numeric_distributions_page_{page}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    return summary


# =============================================================================
# FEATURE-TARGET RELATIONSHIPS
# =============================================================================

def standardized_mean_difference(group_0, group_1):
    """Calculate standardized mean difference between two groups."""

    n0, n1 = len(group_0), len(group_1)

    pooled_variance = (
        (n0 - 1) * group_0.var()
        + (n1 - 1) * group_1.var()
    ) / (n0 + n1 - 2)

    pooled_sd = np.sqrt(pooled_variance)

    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan

    return (
        group_1.mean() - group_0.mean()
    ) / pooled_sd


def feature_target_relationships(data, features):
    """Compare Top-5 and non-Top-5 observations for each feature."""

    rows = []

    for feature in features:
        group_0 = data.loc[
            data["high_ranking"] == 0,
            feature,
        ].dropna()

        group_1 = data.loc[
            data["high_ranking"] == 1,
            feature,
        ].dropna()

        difference = standardized_mean_difference(
            group_0,
            group_1,
        )

        rows.append(
            {
                "feature": feature,
                "not_top_5_mean": group_0.mean(),
                "top_5_mean": group_1.mean(),
                "not_top_5_median": group_0.median(),
                "top_5_median": group_1.median(),
                "standardized_mean_difference": difference,
                "absolute_standardized_difference": abs(difference),
            }
        )

    relationships = (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_standardized_difference",
            ascending=False,
        )
    )

    save_table(
        relationships,
        "feature_target_relationships.csv",
    )

    top = (
        relationships
        .head(12)
        .sort_values("standardized_mean_difference")
    )

    plt.figure(figsize=(10, 7))

    sns.barplot(
        data=top,
        x="standardized_mean_difference",
        y="feature",
    )

    plt.axvline(0, linewidth=1)

    plt.title("Strongest Standardized Feature Differences")
    plt.xlabel("Standardized difference: Top-5 minus non-Top-5")
    plt.ylabel("")

    save_figure("feature_target_differences.png")

    return relationships


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def correlation_analysis(data, features):
    """Create correlation matrices and identify strong predictor correlations."""

    matrix_features = features + ["high_ranking"]

    matrices = {
        method: data[matrix_features].corr(method=method)
        for method in ["pearson", "spearman"]
    }

    for method, matrix in matrices.items():
        save_table(
            matrix,
            f"{method}_correlations.csv",
            index=True,
        )

        plt.figure(figsize=(19, 16))

        sns.heatmap(
            matrix,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
        )

        plt.title(f"{method.title()} Correlation Matrix")

        save_figure(
            f"{method}_correlation_heatmap.png"
        )

    pearson = matrices["pearson"].loc[features, features]

    upper_triangle = pearson.where(
        np.triu(
            np.ones(pearson.shape),
            k=1,
        ).astype(bool)
    )

    pairs = (
        upper_triangle
        .stack()
        .reset_index()
    )

    pairs.columns = [
        "feature_1",
        "feature_2",
        "pearson_correlation",
    ]

    pairs["absolute_correlation"] = (
        pairs["pearson_correlation"].abs()
    )

    strong_pairs = (
        pairs.loc[
            pairs["absolute_correlation"] >= 0.70
        ]
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
    )

    save_table(
        strong_pairs,
        "strong_correlation_pairs.csv",
    )

    return matrices, strong_pairs


# =============================================================================
# MULTICOLLINEARITY
# =============================================================================

def calculate_vif(data, features):
    """Calculate VIF values using complete observations."""

    X = (
        data[features]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )

    X_constant = add_constant(
        X,
        has_constant="add",
    )

    vif = pd.DataFrame(
        {
            "feature": X.columns,
            "vif": [
                variance_inflation_factor(
                    X_constant.values,
                    index,
                )
                for index in range(
                    1,
                    X_constant.shape[1],
                )
            ],
        }
    )

    return vif.sort_values(
        "vif",
        ascending=False,
    )


def multicollinearity_analysis(data):
    """Calculate VIF for the planned feature specifications."""

    specifications = {
        "vif_with_seo_score.csv":
            ["seo_score"] + CATEGORY_FEATURES,

        "vif_categories_only.csv":
            CATEGORY_FEATURES,

        "vif_detailed_features.csv":
            DETAILED_FEATURES,
    }

    results = {}

    for filename, features in specifications.items():
        table = calculate_vif(
            data,
            features,
        )

        save_table(
            table,
            filename,
        )

        results[filename] = table

    return results


# =============================================================================
# METHODOLOGICAL DECISIONS FROM EDA
# =============================================================================

def decision_tables():
    """Document modelling decisions informed by the EDA."""

    leakage = pd.DataFrame(
        [
            ["position", "Direct target leakage", "Exclude from predictors"],
            ["result", "Unique identifier", "Exclude from predictors"],
            ["created_at", "Collection-order proxy", "Exclude from primary models"],
            ["seo_score", "Composite score", "Use only as separate benchmark"],
            ["seo_classification", "Derived from seo_score", "Exclude from primary models"],
            ["analysis_explanation", "Contains score calculation", "Exclude from predictors"],
            ["query", "Repeated-observation dependence", "Use for grouped splitting"],
        ],
        columns=[
            "variable",
            "risk",
            "decision",
        ],
    )

    save_table(
        leakage,
        "leakage_assessment.csv",
    )

    preprocessing = pd.DataFrame(
        [
            [
                "Missing complete SEO analyses",
                "Exclude affected rows from SEO models",
            ],
            [
                "Negative loading times",
                "Recode as missing and impute from training data",
            ],
            [
                "robots_txt value -1",
                "Recode as missing and impute from training data",
            ],
            [
                "Skewed count predictors",
                "Consider log1p transformation for Logistic Regression",
            ],
            [
                "Different predictor scales",
                "Scale predictors for Logistic Regression",
            ],
            [
                "Approximately 25/75 class distribution",
                "Use evaluation metrics suitable for class imbalance",
            ],
            [
                "Repeated queries",
                "Use query-grouped train/test splitting",
            ],
            [
                "SEO score/category overlap",
                "Evaluate separate feature specifications",
            ],
            [
                "Redundant internal/external link fields",
                "Retain *_link_count and exclude internal_links/external_links",
            ],
        ],
        columns=[
            "finding",
            "planned_action",
        ],
    )

    save_table(
        preprocessing,
        "preprocessing_plan.csv",
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the complete exploratory data analysis."""

    FIGURES_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    sns.set_theme(style="whitegrid")

    # Load and prepare data.
    raw, cleaned, analysis = load_and_prepare_data()

    # Dataset structure.
    quality = dataset_quality(
        raw,
        cleaned,
    )

    target_distribution(raw)

    engine_results = search_engine_structure(raw)

    # Predictor-level EDA.
    numeric_distributions(
        analysis,
        ALL_EDA_FEATURES,
    )

    relationships = feature_target_relationships(
        analysis,
        ALL_EDA_FEATURES,
    )

    _, strong_pairs = correlation_analysis(
        analysis,
        ALL_EDA_FEATURES,
    )

    vif_results = multicollinearity_analysis(
        analysis
    )

    decision_tables()

    # Console summary.
    print("\n" + "=" * 70)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nOriginal rows: {len(raw):,}")
    print(f"Rows available for SEO modelling: {len(analysis):,}")

    print_section(
        "SEARCH-ENGINE DISTRIBUTION",
        engine_results["engine_counts"],
    )

    print_section(
        "UNIQUE QUERIES BY SEARCH ENGINE",
        engine_results["queries_per_engine"],
    )

    print_section(
        "TARGET DISTRIBUTION BY SEARCH ENGINE",
        engine_results["class_by_engine"],
    )

    print_section(
        "QUERY COVERAGE ACROSS SEARCH ENGINES",
        engine_results["query_coverage"],
    )

    print_section(
        "RESULTS PER QUERY BY SEARCH ENGINE",
        engine_results["results_per_query"],
    )

    print_section(
        "DATA-QUALITY CHECKS",
        quality,
    )

    print_section(
        "STRONGEST FEATURE-TARGET DIFFERENCES",
        relationships.head(10),
    )

    print_section(
        "STRONG CORRELATIONS (|r| >= 0.70)",
        strong_pairs,
    )

    for filename, table in vif_results.items():
        print_section(
            filename.removesuffix(".csv").upper(),
            table,
        )

    print("\n" + "=" * 70)
    print(f"Tables:  {TABLES_PATH}")
    print(f"Figures: {FIGURES_PATH}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
