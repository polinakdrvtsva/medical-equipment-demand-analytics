import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from config import DB_CONNECTION_STRING


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FEATURES = "data/sample/demand_features.csv"
INPUT_GROUND_TRUTH = "data/sample/ground_truth.csv"

OUTPUT_DIR = Path("data/sample")

# Keep the filename expected by validate_detection.py.
OUTPUT_DETECTION = OUTPUT_DIR / "demand_detection_results.csv"
OUTPUT_ANOMALIES = OUTPUT_DIR / "detected_anomalies.csv"


# ============================================================
# GENERAL DETECTION PARAMETERS
# ============================================================

MIN_HISTORY_MONTHS = 6
BASELINE_WINDOW_MONTHS = 12


# ============================================================
# IQR PARAMETERS
# ============================================================

DEFAULT_IQR_K = 2.5

STABLE_IQR_K = 2.5
ERRATIC_IQR_K = 3.0
INTERMITTENT_IQR_K = 2.5
LUMPY_IQR_K = 3.0
RARE_IQR_K = 3.5


# ============================================================
# SPARSE-DEMAND PARAMETERS
# ============================================================

MEDIAN_STRICT_MULTIPLIER = 2.5
MEDIAN_RELAXED_MULTIPLIER = 3.0

VERY_SPARSE_COVERAGE_LIMIT = 0.20
SPARSE_COVERAGE_LIMIT = 0.40


# ============================================================
# REACTIVATION PARAMETERS
# ============================================================

REACTIVATION_MIN_GAP_MONTHS = 6
REACTIVATION_GAP_MULTIPLIER = 1.5
REACTIVATION_DEMAND_QUANTILE = 0.25
REACTIVATION_SPARSE_COVERAGE_LIMIT = 0.40


# ============================================================
# YEARLY VALIDATION PARAMETERS
# ============================================================

YEARLY_RATIO_THRESHOLD = 2.0
YEARLY_MIN_HISTORY_MONTHS = 12


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# DATA LOADING
# ============================================================

def load_sales():
    """Load sales data from PostgreSQL."""

    logger.info(
        "Loading sales data from PostgreSQL..."
    )

    engine = create_engine(
        DB_CONNECTION_STRING
    )

    query = """
        SELECT
            s.part_id,
            s.date,
            s.demand,
            s.microarea,
            p.category,
            p.subcategory
        FROM sales s
        JOIN products p
            ON s.part_id = p.part_id
        ORDER BY s.part_id, s.date;
    """

    df = pd.read_sql(
        query,
        engine,
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    logger.info(
        "Loaded %s sales rows.",
        len(df),
    )

    return df


def load_features():
    """Load product-level demand features."""

    logger.info(
        "Loading demand features..."
    )

    path = Path(
        INPUT_FEATURES
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Demand features file not found: {path}"
        )

    features = pd.read_csv(
        path
    )

    return features


# ============================================================
# DEMAND PROFILE CLASSIFICATION
# ============================================================

def calculate_demand_features(group):
    """
    Calculate product-level demand characteristics.

    ADI measures demand intermittency.

    CV² measures variability of positive demand.

    These features are calculated from the complete observed
    product history and are used to assign an exploratory
    demand profile.
    """

    demand = group["demand"]

    history_months = len(
        demand
    )

    positive = demand[
        demand > 0
    ]

    n_positive = len(
        positive
    )

    if n_positive == 0:

        adi = np.inf
        cv2 = np.inf

    else:

        adi = (
            history_months
            / n_positive
        )

        if n_positive <= 1:

            cv2 = 0.0

        else:

            mean_positive = (
                positive.mean()
            )

            if mean_positive == 0:

                cv2 = 0.0

            else:

                cv2 = (
                    positive.std(
                        ddof=1
                    )
                    / mean_positive
                ) ** 2

    coverage = (
        n_positive
        / history_months
        if history_months > 0
        else 0.0
    )

    unique_positive = (
        positive.nunique()
    )

    return pd.Series(
        {
            "history_months": history_months,
            "n_positive": n_positive,
            "coverage": coverage,
            "adi": adi,
            "cv2": cv2,
            "unique_positive": unique_positive,
        }
    )


def classify_demand(row):
    """
    Classify demand using ADI and CV².

    Thresholds:

        ADI = 1.32
        CV² = 0.49

    The resulting profiles are:

        STABLE
        ERRATIC
        INTERMITTENT
        LUMPY

    Seasonal demand is intentionally not treated as a separate
    anomaly-detection profile. Seasonality remains available
    as an exploratory feature in demand_features.csv.
    """

    adi = row["adi"]
    cv2 = row["cv2"]

    if adi < 1.32 and cv2 < 0.49:
        return "STABLE"

    if adi < 1.32 and cv2 >= 0.49:
        return "ERRATIC"

    if adi >= 1.32 and cv2 < 0.49:
        return "INTERMITTENT"

    return "LUMPY"


# ============================================================
# GENERAL HISTORY FEATURES
# ============================================================

def add_history_features(group):
    """
    Add leakage-safe historical information.

    history_before_current contains the number of observations
    available before the current observation.

    The current observation is not included.
    """

    group = group.sort_values(
        "date"
    ).copy()

    group["history_before_current"] = (
        np.arange(
            len(group)
        )
    )

    return group


# ============================================================
# RECENT BASELINE
# ============================================================

def calculate_recent_baseline(
    group,
    profile,
):
    """
    Calculate a rolling historical baseline.

    The current observation is excluded using shift(1).

    For STABLE and ERRATIC demand, all previous observations
    are used.

    For INTERMITTENT and LUMPY demand, only previous positive
    observations are used because zero-demand months are an
    expected characteristic of these profiles.
    """

    group = group.sort_values(
        "date"
    ).copy()

    previous_demand = (
        group["demand"]
        .shift(1)
    )

    group["previous_demand"] = (
        previous_demand
    )

    if profile in {
        "STABLE",
        "ERRATIC",
    }:

        baseline_series = (
            previous_demand
        )

        min_periods = (
            MIN_HISTORY_MONTHS
        )

    else:

        baseline_series = (
            previous_demand
            .where(
                previous_demand > 0
            )
        )

        min_periods = 2

    rolling = (
        baseline_series
        .rolling(
            window=BASELINE_WINDOW_MONTHS,
            min_periods=min_periods,
        )
    )

    group["baseline_q1"] = (
        rolling.quantile(
            0.25
        )
    )

    group["baseline_q3"] = (
        rolling.quantile(
            0.75
        )
    )

    group["baseline_median"] = (
        rolling.median()
    )

    group["baseline_iqr"] = (
        group["baseline_q3"]
        - group["baseline_q1"]
    )

    return group


# ============================================================
# PROFILE-SPECIFIC THRESHOLDS
# ============================================================

def calculate_profile_threshold(group):
    """
    Calculate the historical anomaly threshold for one product.

    Seasonality is intentionally not used in the anomaly
    threshold.

    This keeps the detection model focused on exceptional
    quantity spikes relative to the product's recent demand
    behavior.
    """

    group = group.sort_values(
        "date"
    ).copy()

    profile = group[
        "profile"
    ].iloc[0]

    group = add_history_features(
        group
    )

    # --------------------------------------------------------
    # STABLE
    # --------------------------------------------------------

    if profile == "STABLE":

        group = calculate_recent_baseline(
            group,
            profile,
        )

        historical_mean = (
            group["demand"]
            .shift(1)
            .rolling(
                window=BASELINE_WINDOW_MONTHS,
                min_periods=MIN_HISTORY_MONTHS,
            )
            .mean()
        )

        poisson_std = np.sqrt(
            historical_mean.clip(
                lower=0
            )
        )

        group["upper_bound"] = (
            historical_mean
            + 3.5 * poisson_std
        )

        group["outlier_method"] = (
            "POISSON_STABLE"
        )

        return group
    # --------------------------------------------------------
    # ERRATIC
    # --------------------------------------------------------

    if profile == "ERRATIC":

        group = calculate_recent_baseline(
            group,
            profile,
        )

        group["upper_bound"] = (
            group["baseline_q3"]
            + ERRATIC_IQR_K
            * group["baseline_iqr"]
        )

        group["outlier_method"] = (
            "IQR_ERRATIC"
        )

        return group

    # --------------------------------------------------------
    # INTERMITTENT
    # --------------------------------------------------------

    if profile == "INTERMITTENT":

        group = calculate_recent_baseline(
            group,
            profile,
        )

        coverage = (
            group["coverage"].iloc[0]
        )

        if (
            coverage
            < VERY_SPARSE_COVERAGE_LIMIT
        ):

            group["upper_bound"] = (
                group["baseline_median"]
                * MEDIAN_STRICT_MULTIPLIER
            )

            group["outlier_method"] = (
                "MEDIAN_STRICT_INTERMITTENT"
            )

        elif (
            coverage
            < SPARSE_COVERAGE_LIMIT
        ):

            median_threshold = (
                group["baseline_median"]
                * MEDIAN_RELAXED_MULTIPLIER
            )

            q3_threshold = (
                group["baseline_q3"]
            )

            group["upper_bound"] = (
                np.maximum(
                    median_threshold,
                    q3_threshold,
                )
            )

            group["outlier_method"] = (
                "MEDIAN_RELAXED_INTERMITTENT"
            )

        else:

            group["upper_bound"] = (
                group["baseline_q3"]
                + INTERMITTENT_IQR_K
                * group["baseline_iqr"]
            )

            group["outlier_method"] = (
                "IQR_INTERMITTENT"
            )

        return group

    # --------------------------------------------------------
    # LUMPY
    # --------------------------------------------------------

    if profile == "LUMPY":

        group = calculate_recent_baseline(
            group,
            profile,
        )

        coverage = (
            group["coverage"].iloc[0]
        )

        if (
            coverage
            < VERY_SPARSE_COVERAGE_LIMIT
        ):

            group["upper_bound"] = (
                group["baseline_median"]
                * MEDIAN_STRICT_MULTIPLIER
            )

            group["outlier_method"] = (
                "MEDIAN_STRICT_LUMPY"
            )

        elif (
            coverage
            < SPARSE_COVERAGE_LIMIT
        ):

            median_threshold = (
                group["baseline_median"]
                * MEDIAN_RELAXED_MULTIPLIER
            )

            q3_threshold = (
                group["baseline_q3"]
            )

            group["upper_bound"] = (
                np.maximum(
                    median_threshold,
                    q3_threshold,
                )
            )

            group["outlier_method"] = (
                "MEDIAN_RELAXED_LUMPY"
            )

        else:

            group["upper_bound"] = (
                group["baseline_q3"]
                + LUMPY_IQR_K
                * group["baseline_iqr"]
            )

            group["outlier_method"] = (
                "IQR_LUMPY"
            )

        return group

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    group = calculate_recent_baseline(
        group,
        profile,
    )

    group["upper_bound"] = (
        group["baseline_q3"]
        + DEFAULT_IQR_K
        * group["baseline_iqr"]
    )

    group["outlier_method"] = (
        "IQR_DEFAULT"
    )

    return group


# ============================================================
# QUANTITY OUTLIER DETECTION
# ============================================================

def detect_quantity_outliers(group):
    """
    Detect quantity outliers using only historical information.

    The current observation is never included in the baseline.

    An observation must have at least MIN_HISTORY_MONTHS of
    previous history before it can be classified as an anomaly.
    """

    group = group.sort_values(
        "date"
    ).copy()

    enough_history = (
        group["history_before_current"]
        >= MIN_HISTORY_MONTHS
    )

    valid_threshold = (
        group["upper_bound"].notna()
        & np.isfinite(
            group["upper_bound"]
        )
    )

    above_threshold = (
        group["demand"]
        > group["upper_bound"]
    )

    group["is_quantity_outlier"] = (
        enough_history
        & valid_threshold
        & above_threshold
    )

    return group


# ============================================================
# YEARLY VALIDATION
# ============================================================

def apply_yearly_validation(df):
    """
    Calculate year-over-year demand context.

    Yearly validation is descriptive only.

    It does not block or remove an anomaly.
    """

    df = df.copy()

    df["year"] = (
        df["date"].dt.year
    )

    yearly = (
        df.groupby(
            [
                "part_id",
                "year",
            ]
        )["demand"]
        .sum()
        .reset_index(
            name="yearly_demand"
        )
    )

    yearly = yearly.sort_values(
        [
            "part_id",
            "year",
        ]
    )

    yearly["previous_yearly_demand"] = (
        yearly
        .groupby(
            "part_id"
        )[
            "yearly_demand"
        ]
        .shift(1)
    )

    yearly["yearly_ratio"] = (
        yearly["yearly_demand"]
        / yearly[
            "previous_yearly_demand"
        ].replace(
            0,
            np.nan,
        )
    )

    yearly_history = (
        df[
            [
                "part_id",
                "year",
                "history_before_current",
            ]
        ]
        .groupby(
            [
                "part_id",
                "year",
            ]
        )["history_before_current"]
        .min()
        .reset_index()
    )

    yearly = yearly.merge(
        yearly_history,
        on=[
            "part_id",
            "year",
        ],
        how="left",
    )

    enough_history = (
        yearly[
            "history_before_current"
        ]
        >= YEARLY_MIN_HISTORY_MONTHS
    )

    yearly_change_is_large = (
        yearly["yearly_ratio"]
        >= YEARLY_RATIO_THRESHOLD
    )

    yearly["passes_yearly_validation"] = (
        ~enough_history
        | yearly_change_is_large
        | yearly[
            "previous_yearly_demand"
        ].isna()
    )

    df = df.merge(
        yearly[
            [
                "part_id",
                "year",
                "previous_yearly_demand",
                "yearly_ratio",
                "passes_yearly_validation",
            ]
        ],
        on=[
            "part_id",
            "year",
        ],
        how="left",
    )

    return df


# ============================================================
# REACTIVATION FEATURES
# ============================================================

def calculate_reactivation_features(group):
    """
    Detect possible demand reactivation after a long period
    without positive demand.

    Reactivation is kept as a separate signal and does not
    automatically classify an observation as a quantity anomaly.
    """

    group = group.sort_values(
        "date"
    ).copy()

    demand = group[
        "demand"
    ]

    positive_mask = (
        demand > 0
    )

    # --------------------------------------------------------
    # Previous positive-demand date.
    # --------------------------------------------------------

    previous_positive_date = (
        group["date"]
        .where(
            positive_mask
        )
        .ffill()
        .shift(1)
    )

    group[
        "previous_positive_date"
    ] = previous_positive_date

    group[
        "has_previous_demand"
    ] = (
        previous_positive_date.notna()
    )

    group[
        "months_since_previous_demand"
    ] = (
        (
            group["date"]
            - previous_positive_date
        )
        .dt.days
        / 30.44
    )

    # --------------------------------------------------------
    # Typical historical gap.
    # --------------------------------------------------------

    positive_dates = (
        group.loc[
            positive_mask,
            "date",
        ]
        .reset_index(
            drop=True
        )
    )

    if len(positive_dates) >= 2:

        gaps = (
            positive_dates
            .diff()
            .dt.days
            / 30.44
        )

        gaps = gaps.dropna()

        if len(gaps) > 0:

            typical_gap = (
                gaps.median()
            )

        else:

            typical_gap = (
                REACTIVATION_MIN_GAP_MONTHS
            )

    else:

        typical_gap = (
            REACTIVATION_MIN_GAP_MONTHS
        )

    gap_threshold = max(
        REACTIVATION_MIN_GAP_MONTHS,
        typical_gap
        * REACTIVATION_GAP_MULTIPLIER,
    )

    # --------------------------------------------------------
    # Previous positive demand.
    # --------------------------------------------------------

    previous_demand = (
        group["demand"]
        .shift(1)
    )

    previous_positive_demand = (
        previous_demand
        .where(
            previous_demand > 0
        )
    )

    group[
        "previous_positive_demand"
    ] = previous_positive_demand

    # --------------------------------------------------------
    # Row-level historical threshold.
    #
    # Only observations before the current row are used.
    # --------------------------------------------------------

    historical_positive_count = []
    reactivation_thresholds = []

    for i in range(
        len(group)
    ):

        prior_positive = (
            group["demand"]
            .iloc[:i]
        )

        prior_positive = (
            prior_positive[
                prior_positive > 0
            ]
        )

        historical_positive_count.append(
            len(
                prior_positive
            )
        )

        if len(prior_positive) > 0:

            threshold = (
                prior_positive
                .quantile(
                    REACTIVATION_DEMAND_QUANTILE
                )
            )

        else:

            threshold = np.inf

        reactivation_thresholds.append(
            threshold
        )

    group[
        "historical_positive_count"
    ] = historical_positive_count

    group[
        "reactivation_demand_threshold"
    ] = reactivation_thresholds

    # --------------------------------------------------------
    # Gap signal.
    # --------------------------------------------------------

    group[
        "unusual_gap"
    ] = (
        group[
            "months_since_previous_demand"
        ]
        >= gap_threshold
    )

    # --------------------------------------------------------
    # Sparse products.
    # --------------------------------------------------------

    coverage = (
        group["coverage"].iloc[0]
    )

    very_sparse = (
        coverage
        < REACTIVATION_SPARSE_COVERAGE_LIMIT
    )

    # --------------------------------------------------------
    # Sparse reactivation.
    # --------------------------------------------------------

    sparse_reactivation = (
        very_sparse
        & group["unusual_gap"]
        & group["has_previous_demand"]
        & positive_mask
    )

    # --------------------------------------------------------
    # Non-sparse reactivation.
    # --------------------------------------------------------

    non_sparse_reactivation = (
        ~very_sparse
        & group["unusual_gap"]
        & group["has_previous_demand"]
        & positive_mask
        & (
            group["demand"]
            >= group[
                "reactivation_demand_threshold"
            ]
        )
    )

    group["is_reactivation"] = (
        sparse_reactivation
        | non_sparse_reactivation
    )

    return group


# ============================================================
# ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(df):
    """
    Calculate anomaly score:

        demand / upper_bound

    A score above 1 means demand exceeded the threshold.
    """

    df = df.copy()

    df["anomaly_score"] = (
        df["demand"]
        / df["upper_bound"]
    )

    df["anomaly_score"] = (
        df["anomaly_score"]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0)
    )

    return df


# ============================================================
# BUILD DETECTION DATASET
# ============================================================

def build_detection_dataset(
    sales,
    features,
):
    """
    Build the complete anomaly-detection dataset.
    """

    logger.info(
        "Building detection dataset..."
    )

    # --------------------------------------------------------
    # Product-level demand statistics.
    # --------------------------------------------------------

    product_stats = []

    for part_id, group in sales.groupby(
        "part_id"
    ):

        stats = (
            calculate_demand_features(
                group
            )
        )

        stats["part_id"] = (
            part_id
        )

        product_stats.append(
            stats
        )

    product_stats = (
        pd.DataFrame(
            product_stats
        )
    )

    # --------------------------------------------------------
    # Demand profile.
    # --------------------------------------------------------

    product_stats["profile"] = (
        product_stats.apply(
            classify_demand,
            axis=1,
        )
    )

    # --------------------------------------------------------
    # Add seasonality as an exploratory feature only.
    #
    # It is intentionally NOT used to create a separate
    # detection profile.
    # --------------------------------------------------------

    if "seasonality_strength" in features.columns:

        product_stats = (
            product_stats.merge(
                features[
                    [
                        "part_id",
                        "seasonality_strength",
                    ]
                ],
                on="part_id",
                how="left",
            )
        )

    else:

        product_stats[
            "seasonality_strength"
        ] = np.nan

    # --------------------------------------------------------
    # Merge product features into sales.
    # --------------------------------------------------------

    detection = sales.merge(
        product_stats[
            [
                "part_id",
                "history_months",
                "n_positive",
                "coverage",
                "adi",
                "cv2",
                "seasonality_strength",
                "profile",
            ]
        ],
        on="part_id",
        how="left",
    )

    # --------------------------------------------------------
    # Profile-specific baseline.
    # --------------------------------------------------------

    detection = (
        detection
        .groupby(
            "part_id",
            group_keys=False,
        )
        .apply(
            calculate_profile_threshold
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Quantity outliers.
    # --------------------------------------------------------

    detection = (
        detection
        .groupby(
            "part_id",
            group_keys=False,
        )
        .apply(
            detect_quantity_outliers
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Yearly context.
    # --------------------------------------------------------

    detection = (
        apply_yearly_validation(
            detection
        )
    )

    # --------------------------------------------------------
    # Reactivation signal.
    # --------------------------------------------------------

    detection = (
        detection
        .groupby(
            "part_id",
            group_keys=False,
        )
        .apply(
            calculate_reactivation_features
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Final anomaly flag.
    #
    # Reactivation is deliberately separate from quantity
    # anomaly detection.
    # --------------------------------------------------------

    detection["is_anomaly"] = (
        detection[
            "is_quantity_outlier"
        ]
    )

    detection["anomaly_type"] = np.where(
        detection["is_anomaly"],
        "EXCEPTIONAL_SPIKE",
        "NORMAL",
    )

    # --------------------------------------------------------
    # Anomaly score.
    # --------------------------------------------------------

    detection = (
        calculate_anomaly_score(
            detection
        )
    )

    return detection


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    detection
):
    """Save detection results and detected anomalies."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Complete detection dataset.
    # --------------------------------------------------------

    detection.to_csv(
        OUTPUT_DETECTION,
        index=False,
    )

    # --------------------------------------------------------
    # Anomalies only.
    # --------------------------------------------------------

    anomalies = detection[
        detection["is_anomaly"]
    ].copy()

    anomalies.to_csv(
        OUTPUT_ANOMALIES,
        index=False,
    )

    logger.info(
        "Saved detection results to %s",
        OUTPUT_DETECTION,
    )

    logger.info(
        "Saved %s anomalies to %s",
        len(anomalies),
        OUTPUT_ANOMALIES,
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    detection
):
    """Print a compact detection summary."""

    print(
        "\n"
        + "=" * 70
    )

    print(
        "OUTLIER DETECTION SUMMARY"
    )

    print(
        "=" * 70
    )

    total_rows = len(
        detection
    )

    anomaly_count = int(
        detection[
            "is_anomaly"
        ].sum()
    )

    anomaly_share = (
        anomaly_count
        / total_rows
        if total_rows > 0
        else 0
    )

    print(
        f"Total observations: "
        f"{total_rows:,}"
    )

    print(
        f"Detected anomalies: "
        f"{anomaly_count:,}"
    )

    print(
        f"Anomaly share: "
        f"{anomaly_share:.2%}"
    )

    print(
        "\nPROFILE DISTRIBUTION"
    )

    profile_distribution = (
        detection[
            [
                "part_id",
                "profile",
            ]
        ]
        .drop_duplicates()
        [
            "profile"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        profile_distribution
    )

    print(
        "\nDETECTION METHOD"
    )

    print(
        detection.loc[
            detection[
                "is_anomaly"
            ],
            "outlier_method",
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nANOMALIES BY PROFILE"
    )

    print(
        detection.loc[
            detection[
                "is_anomaly"
            ]
        ]
        .groupby(
            "profile"
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nREACTIVATION SIGNAL"
    )

    reactivation_count = int(
        detection[
            "is_reactivation"
        ].sum()
    )

    print(
        f"Reactivation observations: "
        f"{reactivation_count:,}"
    )

    print(
        "\nTOP ANOMALIES"
    )

    top_anomalies = (
        detection[
            detection[
                "is_anomaly"
            ]
        ]
        .sort_values(
            "anomaly_score",
            ascending=False,
        )
        [
            [
                "part_id",
                "date",
                "demand",
                "profile",
                "upper_bound",
                "anomaly_score",
                "outlier_method",
            ]
        ]
        .head(10)
    )

    if len(top_anomalies) > 0:

        print(
            top_anomalies.to_string(
                index=False
            )
        )

    else:

        print(
            "No anomalies detected."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the complete outlier-detection pipeline."""

    logger.info(
        "Starting outlier detection..."
    )

    sales = load_sales()

    features = load_features()

    detection = (
        build_detection_dataset(
            sales=sales,
            features=features,
        )
    )

    print_summary(
        detection
    )

    save_results(
        detection
    )

    logger.info(
        "Outlier detection completed successfully."
    )


if __name__ == "__main__":
    main()
