import pandas as pd


sales = pd.read_csv(
    "data/generated/sales.csv",
    parse_dates=["DATE"],
)

products = pd.read_csv(
    "data/generated/products.csv",
)

ground_truth = pd.read_csv(
    "data/generated/ground_truth.csv",
    parse_dates=["DATE"],
)


print("\n=== SAMPLE PRODUCTS ===")
print(products.head(10).to_string(index=False))


print("\n=== SAMPLE SALES ===")
print(sales.head(20).to_string(index=False))


print("\n=== DEMAND DISTRIBUTION ===")
print(
    sales["DEMAND"]
    .describe(percentiles=[0.5, 0.75, 0.90, 0.95, 0.99])
)


print("\n=== ZERO DEMAND BY CATEGORY ===")

category_stats = (
    sales
    .merge(
        products[["PART_ID", "CATEGORY"]],
        on="PART_ID",
        how="left",
    )
    .groupby("CATEGORY")["DEMAND"]
    .agg(
        TOTAL="count",
        ZERO_DEMAND=lambda x: (x == 0).mean(),
        MEAN="mean",
        MEDIAN="median",
        MAX="max",
    )
)

print(category_stats.round(2).to_string())


print("\n=== EVENT COUNTS ===")
print(
    ground_truth["EVENT_TYPE"]
    .value_counts()
    .to_string()
)


print("\n=== EXAMPLE LARGE ORDERS ===")

large_orders = (
    ground_truth[
        ground_truth["EVENT_TYPE"] == "LARGE_ORDER"
    ]
    .merge(
        sales,
        on=["PART_ID", "DATE"],
        how="left",
    )
    .merge(
        products,
        on="PART_ID",
        how="left",
    )
    .sort_values("DEMAND", ascending=False)
    .head(10)
)

print(
    large_orders[
        [
            "PART_ID",
            "DESCRIPTION",
            "CATEGORY",
            "DATE",
            "DEMAND",
            "MICROAREA",
            "EVENT_TYPE",
        ]
    ].to_string(index=False)
)


print("\n=== EXAMPLE REACTIVATIONS ===")

reactivations = (
    ground_truth[
        ground_truth["EVENT_TYPE"] == "REACTIVATION"
    ]
    .merge(
        sales,
        on=["PART_ID", "DATE"],
        how="left",
    )
    .merge(
        products,
        on="PART_ID",
        how="left",
    )
    .head(10)
)

print(
    reactivations[
        [
            "PART_ID",
            "DESCRIPTION",
            "CATEGORY",
            "DATE",
            "DEMAND",
            "MICROAREA",
            "EVENT_TYPE",
        ]
    ].to_string(index=False)
)
