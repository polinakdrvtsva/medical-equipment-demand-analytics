import pandas as pd
from pathlib import Path


RESULTS_PATH = Path("data/sample/demand_detection_results.csv")
GROUND_TRUTH_PATH = Path("data/sample/ground_truth.csv")


def normalize_columns(df):
    df.columns = [
        column.strip().lower()
        for column in df.columns
    ]
    return df


def main():
    results = pd.read_csv(RESULTS_PATH)
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)

    results = normalize_columns(results)
    ground_truth = normalize_columns(ground_truth)

    print("Results columns:")
    print(results.columns.tolist())

    print("\nGround truth columns:")
    print(ground_truth.columns.tolist())

    # Check required columns explicitly
    required_results = {"part_id", "date", "demand", "profile", "is_anomaly"}
    required_gt = {"part_id", "date", "event_type"}

    missing_results = required_results - set(results.columns)
    missing_gt = required_gt - set(ground_truth.columns)

    if missing_results:
        raise ValueError(
            f"Missing columns in detection results: {missing_results}"
        )

    if missing_gt:
        raise ValueError(
            f"Missing columns in ground truth: {missing_gt}"
        )

    results["date"] = pd.to_datetime(results["date"])
    ground_truth["date"] = pd.to_datetime(ground_truth["date"])

    df = results.merge(
        ground_truth,
        on=["part_id", "date"],
        how="left",
    )

    df["event_type"] = df["event_type"].fillna("NORMAL")

    # ---------------------------------------------------------
    # 1. False positives by profile
    # ---------------------------------------------------------

    anomalies = df[df["is_anomaly"]].copy()
    fp = anomalies[anomalies["event_type"] == "NORMAL"].copy()

    print("\n" + "=" * 70)
    print("FALSE POSITIVES BY PROFILE")
    print("=" * 70)

    print(
        fp["profile"]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # 2. Anomaly rate by profile
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANOMALY RATE BY PROFILE")
    print("=" * 70)

    profile_stats = (
        df.groupby("profile")
        .agg(
            observations=("is_anomaly", "size"),
            anomalies=("is_anomaly", "sum"),
        )
    )

    profile_stats["anomaly_rate"] = (
        profile_stats["anomalies"]
        / profile_stats["observations"]
    )

    print(
        profile_stats
        .sort_values("anomaly_rate", ascending=False)
        .to_string()
    )

    # ---------------------------------------------------------
    # 3. STABLE false positives
    # ---------------------------------------------------------

    stable_fp = fp[fp["profile"] == "STABLE"].copy()

    print("\n" + "=" * 70)
    print("STABLE FALSE POSITIVES")
    print("=" * 70)

    if stable_fp.empty:
        print("No STABLE false positives.")
    else:
        cols = [
            "part_id",
            "date",
            "demand",
            "history_before_current",
            "baseline_median",
            "baseline_q3",
            "baseline_iqr",
            "upper_bound",
            "anomaly_score",
            "outlier_method",
        ]

        print(
            stable_fp
            .sort_values("anomaly_score", ascending=False)
            [cols]
            .head(30)
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # 4. STABLE true positives
    # ---------------------------------------------------------

    stable_tp = anomalies[
        (anomalies["profile"] == "STABLE")
        & (anomalies["event_type"] != "NORMAL")
    ].copy()

    print("\n" + "=" * 70)
    print("STABLE TRUE POSITIVES")
    print("=" * 70)

    if stable_tp.empty:
        print("No STABLE true positives.")
    else:
        cols = [
            "part_id",
            "date",
            "demand",
            "event_type",
            "history_before_current",
            "baseline_median",
            "baseline_q3",
            "baseline_iqr",
            "upper_bound",
            "anomaly_score",
            "outlier_method",
        ]

        print(
            stable_tp
            .sort_values("anomaly_score", ascending=False)
            [cols]
            .head(30)
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # 5. ERRATIC false positives
    # ---------------------------------------------------------

    erratic_fp = fp[fp["profile"] == "ERRATIC"].copy()

    print("\n" + "=" * 70)
    print("ERRATIC FALSE POSITIVES")
    print("=" * 70)

    if erratic_fp.empty:
        print("No ERRATIC false positives.")
    else:
        cols = [
            "part_id",
            "date",
            "demand",
            "history_before_current",
            "baseline_median",
            "baseline_q3",
            "baseline_iqr",
            "upper_bound",
            "anomaly_score",
            "outlier_method",
        ]

        print(
            erratic_fp
            .sort_values("anomaly_score", ascending=False)
            [cols]
            .head(30)
            .to_string(index=False)
        )

    # ---------------------------------------------------------
    # 6. STABLE threshold summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("STABLE THRESHOLD SUMMARY")
    print("=" * 70)

    stable = df[df["profile"] == "STABLE"].copy()

    if not stable.empty:
        print(
            stable[
                [
                    "demand",
                    "baseline_median",
                    "upper_bound",
                    "anomaly_score",
                ]
            ]
            .describe()
            .to_string()
        )

    # ---------------------------------------------------------
    # 7. STABLE event detection
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("STABLE EVENT DETECTION")
    print("=" * 70)

    stable_events = stable[stable["event_type"] != "NORMAL"]

    if stable_events.empty:
        print("No STABLE events in ground truth.")
    else:
        summary = (
            stable_events
            .groupby("event_type")
            .agg(
                events=("event_type", "size"),
                detected=("is_anomaly", "sum"),
                avg_demand=("demand", "mean"),
                avg_upper_bound=("upper_bound", "mean"),
            )
        )

        summary["recall"] = (
            summary["detected"] / summary["events"]
        )

        print(summary.to_string())

    # ---------------------------------------------------------
    # 8. Overall event detection
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL EVENT DETECTION")
    print("=" * 70)

    events = df[df["event_type"] != "NORMAL"]

    summary = (
        events
        .groupby("event_type")
        .agg(
            events=("event_type", "size"),
            detected=("is_anomaly", "sum"),
            avg_demand=("demand", "mean"),
            avg_upper_bound=("upper_bound", "mean"),
        )
    )

    summary["recall"] = (
        summary["detected"] / summary["events"]
    )

    print(summary.to_string())


if __name__ == "__main__":
    main()
