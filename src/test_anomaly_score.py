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

    thresholds = [
        1.0,
        1.1,
        1.2,
        1.3,
        1.5,
        2.0,
    ]

    print("\n" + "=" * 80)
    print("SCORE THRESHOLD BACKTEST")
    print("=" * 80)

    print(
        f"{'Threshold':>10} "
        f"{'Anomalies':>10} "
        f"{'TP':>8} "
        f"{'FP':>8} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10}"
    )

    print("-" * 80)

    for threshold in thresholds:

        detected = (
            df["is_quantity_outlier"]
            & (df["anomaly_score"] >= threshold)
        )

        actual_event = df["event_type"].isin(
            [
                "LARGE_ORDER",
                "REACTIVATION",
                "MAINTENANCE_EVENT",
                "EMERGENCY_REPLACEMENT",
            ]
        )

        tp = (detected & actual_event).sum()
        fp = (detected & ~actual_event).sum()
        fn = (~detected & actual_event).sum()

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )

        print(
            f"{threshold:>10.1f} "
            f"{detected.sum():>10} "
            f"{tp:>8} "
            f"{fp:>8} "
            f"{precision:>10.3f} "
            f"{recall:>10.3f} "
            f"{f1:>10.3f}"
        )

    print("\n" + "=" * 80)
    print("EVENT RECALL BY SCORE THRESHOLD")
    print("=" * 80)

    event_types = [
        "LARGE_ORDER",
        "REACTIVATION",
        "MAINTENANCE_EVENT",
        "EMERGENCY_REPLACEMENT",
    ]

    for threshold in thresholds:

        detected = (
            df["is_quantity_outlier"]
            & (df["anomaly_score"] >= threshold)
        )

        print(f"\nScore >= {threshold:.1f}")

        for event_type in event_types:

            event_rows = df[
                df["event_type"] == event_type
            ]

            if event_rows.empty:
                continue

            detected_count = detected[
                event_rows.index
            ].sum()

            recall = (
                detected_count
                / len(event_rows)
            )

            print(
                f"  {event_type:<25} "
                f"{detected_count:>3}/{len(event_rows):<3} "
                f"recall={recall:.3f}"
            )


if __name__ == "__main__":
    main()
