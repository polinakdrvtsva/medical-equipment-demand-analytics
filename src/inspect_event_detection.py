import pandas as pd


DETECTION_PATH = "data/sample/demand_detection_results.csv"
GROUND_TRUTH_PATH = "data/sample/ground_truth.csv"


def main():
    detection = pd.read_csv(DETECTION_PATH)
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)

    detection["date"] = pd.to_datetime(
        detection["date"]
    )

    ground_truth["DATE"] = pd.to_datetime(
        ground_truth["DATE"]
    )

    ground_truth = ground_truth.rename(
        columns={
            "PART_ID": "part_id",
            "DATE": "date",
            "EVENT_TYPE": "event_type",
        }
    )

    # --------------------------------------------------------
    # Merge ground truth with detection results
    # --------------------------------------------------------

    detection_columns = [
        "part_id",
        "date",
        "demand",
        "profile",
        "coverage",
        "history_before_current",
        "baseline_median",
        "baseline_q3",
        "baseline_iqr",
        "upper_bound",
        "anomaly_score",
        "is_quantity_outlier",
        "is_reactivation",
        "passes_yearly_validation",
        "yearly_ratio",
        "is_anomaly",
        "outlier_method",
    ]

    events = ground_truth.merge(
        detection[detection_columns],
        on=["part_id", "date"],
        how="left",
    )

    # --------------------------------------------------------
    # Event-level summary
    # --------------------------------------------------------

    print("\nEVENT-LEVEL DIAGNOSTICS")
    print("-" * 100)

    summary = (
        events
        .groupby("event_type")
        .agg(
            event_count=("part_id", "size"),
            avg_demand=("demand", "mean"),
            median_demand=("demand", "median"),
            max_demand=("demand", "max"),
            avg_baseline=("baseline_median", "mean"),
            avg_upper_bound=("upper_bound", "mean"),
            avg_anomaly_score=("anomaly_score", "mean"),
            detected_count=("is_anomaly", "sum"),
            reactivation_count=(
                "is_reactivation",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["detection_rate"] = (
        summary["detected_count"]
        / summary["event_count"]
    )

    summary["reactivation_rate"] = (
        summary["reactivation_count"]
        / summary["event_count"]
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # Top events by demand
    # --------------------------------------------------------

    print("\n\nTOP EVENTS BY DEMAND")
    print("-" * 100)

    top_events = (
        events[
            [
                "event_type",
                "part_id",
                "date",
                "demand",
                "profile",
                "coverage",
                "history_before_current",
                "baseline_median",
                "upper_bound",
                "anomaly_score",
                "is_quantity_outlier",
                "is_reactivation",
                "passes_yearly_validation",
                "yearly_ratio",
                "is_anomaly",
                "outlier_method",
            ]
        ]
        .sort_values(
            ["event_type", "demand"],
            ascending=[True, False],
        )
        .groupby("event_type")
        .head(10)
    )

    print(
        top_events.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # Missed events
    # --------------------------------------------------------

    print("\n\nMISSED EVENTS")
    print("-" * 100)

    missed = events[
        (~events["is_anomaly"])
        & (
            events["event_type"]
            != "NEW_PRODUCT"
        )
    ].copy()

    print(
        missed[
            [
                "event_type",
                "part_id",
                "date",
                "demand",
                "profile",
                "coverage",
                "history_before_current",
                "baseline_median",
                "upper_bound",
                "anomaly_score",
                "is_quantity_outlier",
                "is_reactivation",
                "passes_yearly_validation",
                "yearly_ratio",
                "outlier_method",
            ]
        ]
        .sort_values(
            ["event_type", "demand"],
            ascending=[True, False],
        )
        .groupby("event_type")
        .head(15)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # --------------------------------------------------------
    # Why were events not detected?
    # --------------------------------------------------------

    print("\n\nMISSED EVENTS — POSSIBLE REASONS")
    print("-" * 100)

    missed["reason"] = "OTHER"

    missed.loc[
        missed["history_before_current"]
        < 12,
        "reason",
    ] = "INSUFFICIENT_HISTORY"

    missed.loc[
        missed["upper_bound"].isna(),
        "reason",
    ] = "NO_VALID_THRESHOLD"

    missed.loc[
        (
            missed["upper_bound"].notna()
            & (
                missed["demand"]
                <= missed["upper_bound"]
            )
        ),
        "reason",
    ] = "BELOW_THRESHOLD"

    missed.loc[
        (
            missed["history_before_current"]
            >= 12
        )
        & missed["upper_bound"].notna()
        & (
            missed["demand"]
            > missed["upper_bound"]
        )
        & (~missed["is_quantity_outlier"])
    ] = "BLOCKED_BY_DETECTION_RULE"

    reason_summary = (
        missed
        .groupby(
            [
                "event_type",
                "reason",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    print(
        reason_summary.to_string(
            index=False,
        )
    )

    # --------------------------------------------------------
    # Strong missed events
    #
    # Events where demand is above the upper bound
    # but the final detector did not flag them.
    # --------------------------------------------------------

    print(
        "\n\nSTRONG MISSED EVENTS"
    )
    print("-" * 100)

    strong_missed = missed[
        missed["upper_bound"].notna()
        & (
            missed["demand"]
            > missed["upper_bound"]
        )
    ].copy()

    print(
        strong_missed[
            [
                "event_type",
                "part_id",
                "date",
                "demand",
                "profile",
                "coverage",
                "history_before_current",
                "upper_bound",
                "anomaly_score",
                "is_quantity_outlier",
                "is_anomaly",
                "outlier_method",
            ]
        ]
        .sort_values(
            ["event_type", "demand"],
            ascending=[True, False],
        )
        .groupby("event_type")
        .head(15)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )


if __name__ == "__main__":
    main()
