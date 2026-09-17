from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import create_engine

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import DB_CONNECTION_STRING


INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "demand_detection_results.csv"
)

OUTPUT_TABLE = "detection_results"


def main():
    print(f"Loading detection results from: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Detection results file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded from CSV: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # Convert date columns explicitly
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Connect to PostgreSQL
    engine = create_engine(DB_CONNECTION_STRING)

    # Replace the table so that the database always reflects
    # the latest detection run.
    df.to_sql(
        OUTPUT_TABLE,
        engine,
        if_exists="replace",
        index=False,
        method="multi",
    )

    print(
        f"Table '{OUTPUT_TABLE}' successfully created "
        f"with {len(df):,} rows."
    )

    # Basic validation
    with engine.connect() as connection:
        result = connection.exec_driver_sql(
            f"SELECT COUNT(*) FROM {OUTPUT_TABLE}"
        )
        count = result.scalar()

    print(f"Rows in PostgreSQL: {count:,}")

    if count != len(df):
        raise ValueError(
            "Row count mismatch between CSV and PostgreSQL table."
        )

    print("Validation PASSED.")


if __name__ == "__main__":
    main()
