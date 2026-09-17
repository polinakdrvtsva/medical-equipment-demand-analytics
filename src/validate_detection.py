import logging

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sqlalchemy import create_engine

from config import DB_CONNECTION_STRING


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# DATA LOADING
# ============================================================

def load_detection_results():
    """Load anomaly detection results."""

    path = "data/sample/demand_detection_results.csv"

    logger.info(
        "Loading detection results from %s...",
        path,
    )

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    logger.info(
        "Loaded %s detection observations.",
        len(df),
    )

    return df


def load_ground_truth(engine):
    """Load synthetic ground-truth events from PostgreSQL."""

    query = """
        SELECT
            part_id,
            date,
            event_type
        FROM ground_truth
        ORDER BY part_id, date;
    """

    logger.info("Loading ground-truth events from PostgreSQL...")

    ground_truth = pd.read_sql(
        query,
        engine,
    )

    ground_truth["date"] = pd.to_datetime(
        ground_truth["date"]
    )

    logger.info(
        "Loaded %s ground-truth events.",
        len(ground_truth),
    )

    return ground_truth


# ============================================================
# PREPARE VALIDATION DATA
# ============================================================

def prepare_validation_data(
    detection,
    ground_truth,
):
    """
    Combine detection results with ground truth.

    NEW_PRODUCT events are excluded from the anomaly target
    because new-product status is treated as lifecycle context,
    not exceptional demand.
    """

    valid_event_types = {
        "LARGE_ORDER",
        "REACTIVATION",
        "MAINTENANCE_EVENT",
        "EMERGENCY_REPLACEMENT",
    }

    ground_truth = ground_truth.copy()

    ground_truth["is_ground_truth_anomaly"] = (
        ground_truth["event_type"].isin(valid_event_types)
    )

    validation = detection.merge(
        ground_truth[
            [
                "part_id",
                "date",
                "event_type",
                "is_ground_truth_anomaly",
            ]
        ],
        on=["part_id", "date"],
        how="left",
    )

    validation["is_ground_truth_anomaly"] = (
        validation["is_ground_truth_anomaly"]
        .fillna(False).astype(bool)
    )

    validation["is_detected_anomaly"] = (
        validation["is_anomaly"]
        .astype(bool)
    )

    validation["event_type"] = (
        validation["event_type"]
        .fillna("NO_EVENT")
    )

    return validation


# ============================================================
# OVERALL METRICS
# ============================================================

def calculate_overall_metrics(validation):
    """Calculate overall anomaly detection metrics."""

    y_true = validation["is_ground_truth_anomaly"]
    y_pred = validation["is_detected_anomaly"]

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    ).ravel()

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# EVENT-LEVEL METRICS
# ============================================================

def calculate_event_metrics(validation):
    """
    Calculate recall for each ground-truth event type.

    This shows which types of exceptional demand are detected
    well by the algorithm.
    """

    event_types = [
        "LARGE_ORDER",
        "REACTIVATION",
        "MAINTENANCE_EVENT",
        "EMERGENCY_REPLACEMENT",
    ]

    rows = []

    for event_type in event_types:
        actual = (
            validation["event_type"] == event_type
        )

        detected = validation["is_detected_anomaly"]

        event_count = actual.sum()

        detected_count = (
            actual & detected
        ).sum()

        recall = (
            detected_count / event_count
            if event_count > 0
            else 0.0
        )

        rows.append(
            {
                "event_type": event_type,
                "ground_truth_count": event_count,
                "detected_count": detected_count,
                "recall": recall,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# REACTIVATION ANALYSIS
# ============================================================

def calculate_reactivation_metrics(validation):
    """Evaluate the separate reactivation lifecycle signal."""

    actual_reactivation = (
        validation["event_type"] == "REACTIVATION"
    )

    detected_reactivation = (
        validation["is_reactivation"]
    )

    actual_count = actual_reactivation.sum()

    detected_count = (
        actual_reactivation
        & detected_reactivation
    ).sum()

    recall = (
        detected_count / actual_count
        if actual_count > 0
        else 0.0
    )

    return {
        "ground_truth_reactivations": actual_count,
        "detected_reactivations": detected_count,
        "reactivation_recall": recall,
    }


# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

def save_validation_results(
    validation,
    event_metrics,
):
    """Save validation datasets and metrics."""

    validation_path = (
        "data/sample/validation_results.csv"
    )

    event_metrics_path = (
        "data/sample/event_detection_metrics.csv"
    )

    validation.to_csv(
        validation_path,
        index=False,
    )

    event_metrics.to_csv(
        event_metrics_path,
        index=False,
    )

    logger.info(
        "Saved validation results to %s",
        validation_path,
    )

    logger.info(
        "Saved event metrics to %s",
        event_metrics_path,
    )


# ============================================================
# LOG RESULTS
# ============================================================

def log_results(
    overall_metrics,
    event_metrics,
    reactivation_metrics,
):
    """Log validation results."""

    logger.info("")
    logger.info("OVERALL ANOMALY DETECTION METRICS")
    logger.info("---------------------------------")

    logger.info(
        "True positives: %s",
        overall_metrics["true_positives"],
    )

    logger.info(
        "False positives: %s",
        overall_metrics["false_positives"],
    )

    logger.info(
        "False negatives: %s",
        overall_metrics["false_negatives"],
    )

    logger.info(
        "True negatives: %s",
        overall_metrics["true_negatives"],
    )

    logger.info(
        "Precision: %.3f",
        overall_metrics["precision"],
    )

    logger.info(
        "Recall: %.3f",
        overall_metrics["recall"],
    )

    logger.info(
        "F1-score: %.3f",
        overall_metrics["f1"],
    )

    logger.info("")
    logger.info("EVENT-LEVEL RECALL")
    logger.info("------------------")

    logger.info(
        "\n%s",
        event_metrics.to_string(index=False),
    )

    logger.info("")
    logger.info("REACTIVATION SIGNAL")
    logger.info("-------------------")

    logger.info(
        "Ground-truth reactivations: %s",
        reactivation_metrics[
            "ground_truth_reactivations"
        ],
    )

    logger.info(
        "Detected reactivations: %s",
        reactivation_metrics[
            "detected_reactivations"
        ],
    )

    logger.info(
        "Reactivation recall: %.3f",
        reactivation_metrics[
            "reactivation_recall"
        ],
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """Run anomaly detection validation."""

    logger.info(
        "Starting anomaly detection validation..."
    )

    engine = create_engine(
        DB_CONNECTION_STRING
    )

    detection = load_detection_results()

    ground_truth = load_ground_truth(engine)

    validation = prepare_validation_data(
        detection,
        ground_truth,
    )

    overall_metrics = calculate_overall_metrics(
        validation
    )

    event_metrics = calculate_event_metrics(
        validation
    )

    reactivation_metrics = (
        calculate_reactivation_metrics(
            validation
        )
    )

    log_results(
        overall_metrics,
        event_metrics,
        reactivation_metrics,
    )

    save_validation_results(
        validation,
        event_metrics,
    )

    logger.info(
        "Anomaly detection validation completed successfully."
    )


if __name__ == "__main__":
    main()
