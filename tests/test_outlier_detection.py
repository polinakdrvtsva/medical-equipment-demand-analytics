import numpy as np
import pandas as pd

from src.outlier_detection import (
    calculate_adi,
    calculate_cv2,
    classify_demand_profile,
)


def test_calculate_adi():
    demand = pd.Series([0, 10, 0, 5, 0, 0])

    adi = calculate_adi(demand)

    assert adi == 3.0


def test_calculate_adi_with_all_zero_demand():
    demand = pd.Series([0, 0, 0, 0])

    adi = calculate_adi(demand)

    assert np.isinf(adi)


def test_calculate_cv2_with_constant_positive_demand():
    demand = pd.Series([5, 5, 5, 5])

    cv2 = calculate_cv2(demand)

    assert cv2 == 0.0


def test_calculate_cv2_ignores_zero_demand():
    demand = pd.Series([0, 5, 0, 5, 0, 5])

    cv2 = calculate_cv2(demand)

    assert cv2 == 0.0


def test_calculate_cv2_with_variable_demand():
    demand = pd.Series([2, 4, 6, 8])

    cv2 = calculate_cv2(demand)

    assert cv2 > 0


def test_calculate_cv2_with_single_positive_observation():
    demand = pd.Series([0, 0, 10, 0])

    cv2 = calculate_cv2(demand)

    assert cv2 == 0.0


def test_classify_stable_demand():
    profile = classify_demand_profile(
        adi=2.0,
        cv2=0.5,
    )

    assert profile == "STABLE"


def test_classify_erratic_demand():
    profile = classify_demand_profile(
        adi=2.0,
        cv2=2.0,
    )

    assert profile == "ERRATIC"


def test_classify_intermittent_demand():
    profile = classify_demand_profile(
        adi=5.0,
        cv2=0.5,
    )

    assert profile == "INTERMITTENT"


def test_classify_lumpy_demand():
    profile = classify_demand_profile(
        adi=5.0,
        cv2=2.0,
    )

    assert profile == "LUMPY"


def test_classify_rare_demand():
    profile = classify_demand_profile(
        adi=15.0,
        cv2=2.0,
    )

    assert profile == "RARE"


def test_classify_lumpy_high_adi_low_cv2():
    profile = classify_demand_profile(
        adi=15.0,
        cv2=0.5,
    )

    assert profile == "LUMPY"
