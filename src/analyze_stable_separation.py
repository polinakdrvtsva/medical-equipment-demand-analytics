import pandas as pd
from pathlib import Path


RESULTS_PATH = Path(
    "data/sample/demand_detection_results.csv"
)

GROUND_TRUTH_PATH = Path(
    "data/sample/ground_truth.csv"
)


def main():
    results = pd.read_csv(RESULTS_PATH)
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)

    results.columns = results.columns.str.strip().str.lower()
    ground_truth.columns = ground_truth.columns.str.strip().str.lower()

    results["date"] = pd.to_datetime(results["date"])
    ground_truth["date"] = pd.to_datetime(ground_truth["date"])

    df = results.merge(
        ground_truth,
        on=["part_id", "date"],
        how="left",
    )

    df["event_type"] = df["event_type"].fillna("NORMAL")

    # Only STABLE products
    stable = df[df["profile"] == "STABLE"].copy()

    # ---------------------------------------------------------
    # 1. Distribution by event type
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("STABLE DEMAND DISTRIBUTION")
    print("=" * 80)

    summary = (
        stable
        .groupby("event_type")
        .agg(
            observations=("demand", "size"),
            mean_demand=("demand", "mean"),
            median_demand=("demand", "median"),
            q75_demand=("demand", lambda x: x.quantile(0.75)),
            q90_demand=("demand", lambda x: x.quantile(0.90)),
            q95_demand=("demand", lambda x: x.quantile(0.95)),
            max_demand=("demand", "max"),
            mean_score=("anomaly_score", "mean"),
            median_score=("anomaly_score", "median"),
            q75_score=("anomaly_score", lambda x: x.quantile(0.75)),
            q90_score=("anomaly_score", lambda x: x.quantile(0.90)),
            max_score=("anomaly_score", "max"),
        )
        .sort_values("mean_demand", ascending=False)
    )

    print(summary.to_string())

    # ---------------------------------------------------------
    # 2. How many observations exceed different demand levels?
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("STABLE EVENT SHARE ABOVE DEMAND THRESHOLDS")
    print("=" * 80)

    demand_thresholds = [8, 10, 12, 15, 20, 25, 30]

    event_types = [
        "NORMAL",
        "LARGE_ORDER",
        "MAINTENANCE_EVENT",
        "REACTIVATION",
        "EMERGENCY_REPLACEMENT",
    ]

    for threshold in demand_thresholds:

        print(f"\nDemand >= {threshold}")

        for event_type in event_types:

            subset = stable[
                stable["event_type"] == event_type
            ]

            if subset.empty:
                continue

            share = (
                (subset["demand"] >= threshold).mean()
            )

            print(
                f"  {event_type:<25} "
                f"{share:.3f}"
            )

    # ---------------------------------------------------------
    # 3. How many observations exceed different score levels?
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("STABLE EVENT SHARE ABOVE SCORE THRESHOLDS")
    print("=" * 80)

    score_thresholds = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]

    for threshold in score_thresholds:

        print(f"\nScore >= {threshold}")

        for event_type in event_types:

            subset = stable[
                stable["event_type"] == event_type
            ]

            if subset.empty:
                continue

            share = (
                (subset["anomaly_score"] >= threshold)
                .mean()
            )

            print(
                f"  {event_type:<25} "
                f"{share:.3f}"
            )

    # ---------------------------------------------------------
    # 4. Normal observations that look most like events
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("HIGHEST NORMAL STABLE OBSERVATIONS")
    print("=" * 80)

    normal = stable[
        stable["event_type"] == "NORMAL"
    ].copy()

    cols = [
        "part_id",
        "date",
        "demand",
        "history_before_current",
        "baseline_median",
        "upper_bound",
        "anomaly_score",
    ]

    print(
        normal
        .sort_values(
            ["demand", "anomaly_score"],
            ascending=False,
        )
        [cols]
        .head(30)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # 5. Events that look most like normal observations
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("LOWEST EVENT STABLE OBSERVATIONS")
    print("=" * 80)

    events = stable[
        stable["event_type"] != "NORMAL"
    ].copy()

    print(
        events
        .sort_values(
            ["anomaly_score", "demand"],
            ascending=True,
        )
        [cols + ["event_type"]]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
