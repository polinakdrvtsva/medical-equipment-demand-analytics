import pandas as pd


PATH = "data/sample/demand_detection_results.csv"


def main():
    df = pd.read_csv(PATH)

    stable = df[
        (df["profile"] == "STABLE")
        & (df["is_anomaly"])
    ].copy()

    print("\nSTABLE ANOMALIES")
    print("-" * 80)

    print(
        stable[
            [
                "part_id",
                "date",
                "demand",
                "baseline_median",
                "upper_bound",
                "anomaly_score",
                "outlier_method",  
            ]
        ]
        .sort_values(["part_id", "date"])
        .head(50)
        .to_string(index=False)
    )

    print("\n\nSUMMARY")
    print("-" * 80)

    print(
        stable[
            [
                "demand",
                "baseline_median",
                "upper_bound",
                "anomaly_score",
            ]
        ].describe()
    )

    print("\n\nANOMALIES BY PART")
    print("-" * 80)

    print(
        stable["part_id"]
        .value_counts()
        .describe()
    )


if __name__ == "__main__":
    main()
