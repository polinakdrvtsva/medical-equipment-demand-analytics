import logging

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from config import DB_CONNECTION_STRING


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def load_sales(engine):
    """Load sales data from PostgreSQL."""

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

    logger.info("Loading sales data from PostgreSQL...")

    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])

    logger.info("Loaded %s sales rows.", len(df))

    return df


def calculate_adi(demand):
    """
    Calculate Average Demand Interval (ADI).

    ADI represents the average number of periods
    between non-zero demand observations.
    """

    total_periods = len(demand)
    positive_periods = (demand > 0).sum()

    if positive_periods == 0:
        return np.inf

    return total_periods / positive_periods


def calculate_cv2(demand):
    """
    Calculate squared coefficient of variation (CV²)
    using positive demand observations only.
    """

    positive_demand = demand[demand > 0]

    if len(positive_demand) <= 1:
        return 0.0

    mean_demand = positive_demand.mean()

    if mean_demand == 0:
        return 0.0

    return (
        positive_demand.std(ddof=1) / mean_demand
    ) ** 2


def calculate_seasonality_strength(demand, dates):
    """
    Estimate seasonality strength using monthly demand.

    The metric compares the difference between the highest
    and lowest average calendar-month demand with the
    overall average demand.

    This metric is stored as a feature but is not used
    directly for demand profile classification.
    """

    data = pd.DataFrame(
        {
            "date": dates,
            "demand": demand,
        }
    ).copy()

    data["month"] = data["date"].dt.month

    monthly_mean = data.groupby("month")["demand"].mean()

    if len(monthly_mean) < 12:
        return 0.0

    overall_mean = data["demand"].mean()

    if overall_mean == 0:
        return 0.0

    return (
        (monthly_mean.max() - monthly_mean.min())
        / overall_mean
    )


def classify_demand_profile(adi, cv2):
    """
    Classify demand profile using ADI and CV².

    The thresholds are generic analytical thresholds
    designed for this synthetic portfolio project.
    """

    # Very infrequent demand with high variability.
    if adi >= 12.0:
        if cv2 >= 1.0:
            return "RARE"

        return "LUMPY"

    # Intermittent demand.
    if adi >= 4.0:
        if cv2 >= 1.0:
            return "LUMPY"

        return "INTERMITTENT"

    # Frequent but highly variable demand.
    if cv2 >= 1.0:
        return "ERRATIC"

    # Frequent and relatively stable demand.
    return "STABLE"


def build_product_features(df):
    """Build product-level demand features."""

    logger.info("Calculating product-level demand features...")

    rows = []

    for part_id, group in df.groupby("part_id"):
        demand = group["demand"]
        dates = group["date"]

        avg_demand = demand.mean()
        demand_std = demand.std()

        if pd.isna(demand_std):
            demand_std = 0.0

        adi = calculate_adi(demand)

        cv2 = calculate_cv2(demand)

        seasonality_strength = calculate_seasonality_strength(
            demand,
            dates,
        )

        rows.append(
            {
                "part_id": part_id,
                "category": group["category"].iloc[0],
                "subcategory": group["subcategory"].iloc[0],
                "first_observed_month": dates.min(),
                "last_observed_month": dates.max(),
                "months_observed": len(group),
                "total_demand": demand.sum(),
                "avg_demand": avg_demand,
                "median_demand": demand.median(),
                "demand_std": demand_std,
                "max_demand": demand.max(),
                "positive_months": (demand > 0).sum(),
                "zero_months": (demand == 0).sum(),
                "adi": adi,
                "cv2": cv2,
                "seasonality_strength": seasonality_strength,
            }
        )

    features = pd.DataFrame(rows)

    features["zero_demand_share"] = (
        features["zero_months"]
        / features["months_observed"]
    )

    features["positive_demand_share"] = (
        features["positive_months"]
        / features["months_observed"]
    )

    features["cv"] = (
        features["demand_std"]
        / features["avg_demand"].replace(0, np.nan)
    )

    features["cv"] = (
        features["cv"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    features["profile"] = features.apply(
        lambda row: classify_demand_profile(
            row["adi"],
            row["cv2"],
        ),
        axis=1,
    )

    return features


def save_features(features):
    """Save product-level demand features to CSV."""

    output_path = "data/sample/demand_features.csv"

    features.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Saved demand features to %s",
        output_path,
    )


def main():
    """Run the demand analysis pipeline."""

    logger.info("Starting demand analysis...")

    engine = create_engine(DB_CONNECTION_STRING)

    sales = load_sales(engine)

    features = build_product_features(sales)

    logger.info(
        "Calculated features for %s products.",
        len(features),
    )

    logger.info("Demand profile distribution:")

    logger.info(
        "\n%s",
        features["profile"].value_counts().sort_index(),
    )

    save_features(features)

    logger.info(
        "Demand analysis completed successfully."
    )


if __name__ == "__main__":
    main()
