from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

N_PRODUCTS = 700

START_DATE = "2022-01-01"
END_DATE = "2025-12-01"

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "sample"

PRODUCTS_FILE = OUTPUT_DIR / "products_sample.csv"
SALES_FILE = OUTPUT_DIR / "sales_sample.csv"
GROUND_TRUTH_FILE = OUTPUT_DIR / "ground_truth.csv"


# ============================================================
# PRODUCT CATALOG
# ============================================================

CATEGORIES = {
    "IMAGING": {
        "n": 120,
        "products": [
            ("Ultrasound Transducer", "TRANSDUCERS"),
            ("Ultrasound Control Module", "CONTROL MODULES"),
            ("X-Ray Detector Module", "DETECTOR COMPONENTS"),
            ("X-Ray Power Supply", "POWER COMPONENTS"),
            ("MRI Cooling Pump", "COOLING COMPONENTS"),
            ("MRI Gradient Amplifier", "POWER COMPONENTS"),
            ("CT Patient Table Motor", "MOTORS"),
            ("CT Detector Assembly", "DETECTOR COMPONENTS"),
            ("Imaging System Cable", "CABLES"),
            ("Imaging Console Keyboard", "USER INTERFACE"),
        ],
    },
    "PATIENT_MONITORING": {
        "n": 110,
        "products": [
            ("ECG Cable Assembly", "CABLES"),
            ("ECG Electrode Lead", "ELECTRODES"),
            ("Patient Monitor Display", "DISPLAY COMPONENTS"),
            ("Patient Monitor Battery", "POWER COMPONENTS"),
            ("SpO2 Sensor", "SENSORS"),
            ("Blood Pressure Module", "SENSOR MODULES"),
            ("Temperature Probe", "SENSORS"),
            ("Patient Monitor Main Board", "CONTROL MODULES"),
            ("Pulse Oximeter Cable", "CABLES"),
            ("Monitor Mounting Arm", "MECHANICAL COMPONENTS"),
        ],
    },
    "RESPIRATORY": {
        "n": 90,
        "products": [
            ("Ventilator Valve", "VALVES"),
            ("Ventilator Flow Sensor", "SENSORS"),
            ("Ventilator Oxygen Sensor", "SENSORS"),
            ("Ventilator Control Board", "CONTROL MODULES"),
            ("Ventilator Display Module", "DISPLAY COMPONENTS"),
            ("Ventilator Motor", "MOTORS"),
            ("Ventilator Air Filter", "FILTERS"),
            ("Ventilator Tubing Assembly", "TUBING"),
            ("Respiratory Humidifier Heater", "HEATING COMPONENTS"),
            ("Respiratory Pressure Sensor", "SENSORS"),
        ],
    },
    "INFUSION": {
        "n": 90,
        "products": [
            ("Infusion Pump Motor", "MOTORS"),
            ("Infusion Pump Battery", "POWER COMPONENTS"),
            ("Infusion Pump Display", "DISPLAY COMPONENTS"),
            ("Infusion Pump Control Board", "CONTROL MODULES"),
            ("Infusion Pump Door Assembly", "MECHANICAL COMPONENTS"),
            ("Infusion Pump Pressure Sensor", "SENSORS"),
            ("Infusion Pump Valve", "VALVES"),
            ("Infusion Pump Cable", "CABLES"),
            ("Syringe Pump Motor", "MOTORS"),
            ("Syringe Pump Control Module", "CONTROL MODULES"),
        ],
    },
    "SURGICAL": {
        "n": 85,
        "products": [
            ("Surgical Table Motor", "MOTORS"),
            ("Surgical Table Control Unit", "CONTROL MODULES"),
            ("Surgical Light LED Module", "LIGHTING COMPONENTS"),
            ("Surgical Light Power Supply", "POWER COMPONENTS"),
            ("Electrosurgical Generator Cable", "CABLES"),
            ("Electrosurgical Foot Switch", "USER INTERFACE"),
            ("Surgical Instrument Drive", "MECHANICAL COMPONENTS"),
            ("Operating Table Sensor", "SENSORS"),
            ("Surgical Pump Valve", "VALVES"),
            ("Surgical Equipment Battery", "POWER COMPONENTS"),
        ],
    },
    "LABORATORY": {
        "n": 80,
        "products": [
            ("Analyzer Pump", "PUMPS"),
            ("Analyzer Sample Probe", "SENSORS"),
            ("Analyzer Control Board", "CONTROL MODULES"),
            ("Analyzer Photometric Module", "OPTICAL COMPONENTS"),
            ("Laboratory Centrifuge Motor", "MOTORS"),
            ("Centrifuge Rotor Assembly", "MECHANICAL COMPONENTS"),
            ("Laboratory Refrigerator Compressor", "COOLING COMPONENTS"),
            ("Laboratory Printer Module", "USER INTERFACE"),
            ("Sample Handling Valve", "VALVES"),
            ("Laboratory Power Supply", "POWER COMPONENTS"),
        ],
    },
    "EMERGENCY": {
        "n": 65,
        "products": [
            ("Defibrillator Battery", "POWER COMPONENTS"),
            ("Defibrillator Paddle Cable", "CABLES"),
            ("Defibrillator Control Board", "CONTROL MODULES"),
            ("Defibrillator Display", "DISPLAY COMPONENTS"),
            ("Emergency Monitor Sensor", "SENSORS"),
            ("Emergency Ventilator Valve", "VALVES"),
            ("Emergency Equipment Power Supply", "POWER COMPONENTS"),
            ("AED Battery Pack", "POWER COMPONENTS"),
            ("AED Electrode Cable", "CABLES"),
            ("Emergency Device Charger", "POWER COMPONENTS"),
        ],
    },
    "STERILIZATION": {
        "n": 60,
        "products": [
            ("Autoclave Door Seal", "SEALS"),
            ("Autoclave Heating Element", "HEATING COMPONENTS"),
            ("Autoclave Control Board", "CONTROL MODULES"),
            ("Autoclave Pressure Sensor", "SENSORS"),
            ("Autoclave Water Pump", "PUMPS"),
            ("Sterilizer Temperature Probe", "SENSORS"),
            ("Sterilizer Valve Assembly", "VALVES"),
            ("Sterilizer Display Module", "DISPLAY COMPONENTS"),
            ("Sterilizer Power Supply", "POWER COMPONENTS"),
            ("Sterilizer Door Motor", "MOTORS"),
        ],
    },
}


# Exact profile distribution.
PROFILE_COUNTS = {
    "STABLE": 140,
    "SEASONAL": 70,
    "ERRATIC": 105,
    "INTERMITTENT": 175,
    "LUMPY": 140,
    "RARE": 70,
}


MICROAREAS = [
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "CENTRAL",
]


# Each category has a typical demand scale.
CATEGORY_BASE_DEMAND = {
    "IMAGING": 3.8,
    "PATIENT_MONITORING": 5.0,
    "RESPIRATORY": 4.4,
    "INFUSION": 4.8,
    "SURGICAL": 3.7,
    "LABORATORY": 4.5,
    "EMERGENCY": 5.2,
    "STERILIZATION": 4.0,
}


# Preferred profiles by subcategory.
# This is used to influence demand characteristics without
# breaking the exact global profile distribution.
PROFILE_PREFERENCE = {
    "TRANSDUCERS": ["INTERMITTENT", "LUMPY", "STABLE"],
    "CONTROL MODULES": ["STABLE", "ERRATIC", "SEASONAL"],
    "DETECTOR COMPONENTS": ["ERRATIC", "STABLE", "LUMPY"],
    "POWER COMPONENTS": ["STABLE", "ERRATIC", "SEASONAL"],
    "COOLING COMPONENTS": ["STABLE", "INTERMITTENT", "LUMPY"],
    "MOTORS": ["STABLE", "ERRATIC", "INTERMITTENT"],
    "CABLES": ["STABLE", "INTERMITTENT", "SEASONAL"],
    "SENSORS": ["STABLE", "ERRATIC", "INTERMITTENT"],
    "SENSOR MODULES": ["STABLE", "ERRATIC", "INTERMITTENT"],
    "DISPLAY COMPONENTS": ["STABLE", "SEASONAL", "ERRATIC"],
    "ELECTRODES": ["INTERMITTENT", "STABLE", "RARE"],
    "MECHANICAL COMPONENTS": ["STABLE", "LUMPY", "INTERMITTENT"],
    "VALVES": ["INTERMITTENT", "LUMPY", "ERRATIC"],
    "FILTERS": ["STABLE", "INTERMITTENT", "SEASONAL"],
    "TUBING": ["STABLE", "INTERMITTENT", "LUMPY"],
    "HEATING COMPONENTS": ["STABLE", "SEASONAL", "ERRATIC"],
    "USER INTERFACE": ["STABLE", "ERRATIC", "SEASONAL"],
    "LIGHTING COMPONENTS": ["STABLE", "SEASONAL", "ERRATIC"],
    "PUMPS": ["STABLE", "ERRATIC", "INTERMITTENT"],
    "OPTICAL COMPONENTS": ["ERRATIC", "STABLE", "LUMPY"],
    "SEALS": ["INTERMITTENT", "LUMPY", "RARE"],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def choose_launch_date(rng):
    """
    Assign a realistic product launch date.

    Most products have a full history, while some products
    are introduced during the observation period.
    """

    options = [
        pd.Timestamp("2022-01-01"),
        pd.Timestamp("2022-01-01"),
        pd.Timestamp("2022-01-01"),
        pd.Timestamp("2022-01-01"),
        pd.Timestamp("2022-07-01"),
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-07-01"),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-07-01"),
    ]

    return rng.choice(options)


def choose_preferred_profile(subcategory, rng):
    """
    Return a preferred profile for a subcategory.

    This preference is used only as a scoring mechanism.
    Exact global profile counts are enforced separately.
    """

    preferences = PROFILE_PREFERENCE.get(
        subcategory,
        list(PROFILE_COUNTS.keys()),
    )

    return rng.choice(preferences)


def assign_exact_profiles(products_df, rng):
    """
    Assign the exact requested number of products to each
    demand profile.

    Subcategory preference is incorporated through a scoring
    process, while the global profile counts remain exact.
    """

    profiles = []

    for profile, count in PROFILE_COUNTS.items():
        profiles.extend([profile] * count)

    profiles = np.array(profiles)

    if len(profiles) != len(products_df):
        raise ValueError(
            "PROFILE_COUNTS must sum to N_PRODUCTS."
        )

    rng.shuffle(profiles)

    products_df = products_df.copy()

    # Initial random assignment.
    products_df["PROFILE"] = profiles

    # Improve profile/subcategory compatibility through
    # a limited number of swaps.
    for _ in range(5000):
        idx_a, idx_b = rng.integers(
            0,
            len(products_df),
            size=2,
        )

        if idx_a == idx_b:
            continue

        profile_a = products_df.iloc[idx_a]["PROFILE"]
        profile_b = products_df.iloc[idx_b]["PROFILE"]

        subcat_a = products_df.iloc[idx_a]["SUBCATEGORY"]
        subcat_b = products_df.iloc[idx_b]["SUBCATEGORY"]

        pref_a = PROFILE_PREFERENCE.get(subcat_a, [])
        pref_b = PROFILE_PREFERENCE.get(subcat_b, [])

        before = (
            int(profile_a in pref_a)
            + int(profile_b in pref_b)
        )

        after = (
            int(profile_b in pref_a)
            + int(profile_a in pref_b)
        )

        if after > before:
            products_df.iloc[idx_a, products_df.columns.get_loc("PROFILE")] = profile_b
            products_df.iloc[idx_b, products_df.columns.get_loc("PROFILE")] = profile_a

    return products_df


def choose_microarea(preferred_area, rng):
    """
    Select the sales microarea.

    The product usually sells in its preferred area,
    but cross-regional sales are possible.
    """

    if rng.random() < 0.70:
        return preferred_area

    other_areas = [
        area for area in MICROAREAS
        if area != preferred_area
    ]

    return rng.choice(other_areas)


def get_profile_parameters(profile, category, subcategory, rng):
    """
    Return the base demand parameters for a product.

    Category controls the overall demand scale.
    Profile controls demand behavior.
    """

    category_base = CATEGORY_BASE_DEMAND[category]

    # Small product-specific variation.
    base = category_base * rng.uniform(0.65, 1.35)

    # Some subcategories naturally have lower demand.
    if subcategory in {
        "CONTROL MODULES",
        "POWER COMPONENTS",
        "OPTICAL COMPONENTS",
        "DETECTOR COMPONENTS",
    }:
        base *= rng.uniform(0.75, 1.00)

    if subcategory in {
        "CABLES",
        "FILTERS",
        "SEALS",
        "TUBING",
        "ELECTRODES",
    }:
        base *= rng.uniform(0.85, 1.20)

    return max(base, 0.5)


def seasonal_factor(month):
    """
    Realistic recurring demand seasonality.

    Peaks are intentionally moderate rather than extreme.
    """

    factors = {
        1: 0.78,
        2: 0.82,
        3: 0.98,
        4: 1.12,
        5: 1.28,
        6: 1.22,
        7: 1.08,
        8: 0.98,
        9: 0.96,
        10: 1.12,
        11: 0.90,
        12: 0.80,
    }

    return factors[month]


def generate_base_demand(
    profile,
    base_demand,
    date,
    rng,
):
    """
    Generate monthly demand according to the assigned profile.
    """

    if profile == "STABLE":
        demand = rng.poisson(base_demand)

        # Small probability of zero demand.
        if rng.random() < 0.01:
            demand = 0

    elif profile == "SEASONAL":
        seasonal = seasonal_factor(date.month)

        demand = rng.poisson(
            base_demand * seasonal
        )

        if rng.random() < 0.02:
            demand = 0

    elif profile == "ERRATIC":
        # Frequent positive demand with high variability.
        variable_mean = base_demand * rng.lognormal(
            mean=0.0,
            sigma=0.45,
        )

        demand = rng.poisson(
            max(variable_mean, 0.2)
        )

        if rng.random() < 0.05:
            demand = 0

    elif profile == "INTERMITTENT":
        if rng.random() < 0.28:
            demand = rng.poisson(
                max(base_demand, 0.5)
            )
        else:
            demand = 0

    elif profile == "LUMPY":
        if rng.random() < 0.18:
            multiplier = rng.uniform(1.5, 4.0)

            demand = rng.poisson(
                max(base_demand * multiplier, 1.0)
            )
        else:
            demand = 0

    elif profile == "RARE":
        if rng.random() < 0.07:
            multiplier = rng.uniform(1.0, 3.0)

            demand = rng.poisson(
                max(base_demand * multiplier, 1.0)
            )
        else:
            demand = 0

    else:
        raise ValueError(
            f"Unknown demand profile: {profile}"
        )

    return int(max(demand, 0))


def choose_event(
    profile,
    date,
    demand,
    history,
    reactivation_count,
    maintenance_count,
    rng,
):
    """
    Generate non-new-product events.

    Event rules are intentionally generic and synthetic.
    """

    # --------------------------------------------------------
    # REACTIVATION
    # --------------------------------------------------------

    if (
        profile in {"INTERMITTENT", "LUMPY", "RARE"}
        and len(history) >= 6
        and all(value == 0 for value in history[-6:])
        and reactivation_count < 2
        and rng.random() < 0.08
    ):
        reactivation_demand = max(
            demand,
            int(
                round(
                    rng.uniform(2.0, 5.0)
                    * max(
                        1.0,
                        np.mean([
                            x for x in history
                            if x > 0
                        ]) if any(
                            x > 0 for x in history
                        ) else 1.0,
                    )
                )
            ),
        )

        return reactivation_demand, "REACTIVATION"

    # --------------------------------------------------------
    # MAINTENANCE EVENT
    # --------------------------------------------------------

    maintenance_months = {
        3,
        4,
        5,
        9,
        10,
    }

    if (
        date.month in maintenance_months
        and maintenance_count < 2
        and demand > 0
        and rng.random() < 0.012
    ):
        maintenance_demand = max(
            demand,
            int(
                round(
                    demand
                    * rng.uniform(1.5, 3.0)
                )
            ),
        )

        return maintenance_demand, "MAINTENANCE_EVENT"

    # --------------------------------------------------------
    # LARGE ORDER
    # --------------------------------------------------------

    if (
        demand > 0
        and rng.random() < 0.012
    ):
        large_order_demand = max(
            demand,
            int(
                round(
                    demand
                    * rng.uniform(2.5, 6.0)
                )
            ),
        )

        return large_order_demand, "LARGE_ORDER"

    return demand, None


# ============================================================
# GENERATE PRODUCTS
# ============================================================

def generate_products(rng):
    """
    Generate the synthetic product master table.
    """

    rows = []

    part_counter = 1

    for category, category_data in CATEGORIES.items():

        n_products = category_data["n"]

        for _ in range(n_products):

            description, subcategory = rng.choice(
                category_data["products"]
            )

            launch_date = choose_launch_date(rng)

            preferred_area = rng.choice(
                MICROAREAS
            )

            rows.append(
                {
                    "PART_ID": f"MED-{part_counter:04d}",
                    "DESCRIPTION": description,
                    "CATEGORY": category,
                    "SUBCATEGORY": subcategory,
                    "LAUNCH_DATE": launch_date,
                    "PREFERRED_MICROAREA": preferred_area,
                }
            )

            part_counter += 1

    products_df = pd.DataFrame(rows)

    if len(products_df) != N_PRODUCTS:
        raise ValueError(
            f"Expected {N_PRODUCTS} products, "
            f"generated {len(products_df)}."
        )

    products_df = assign_exact_profiles(
        products_df,
        rng,
    )

    return products_df


# ============================================================
# GENERATE SALES
# ============================================================

def generate_sales(products_df, rng):
    """
    Generate monthly demand for each product.

    The internal PROFILE field is used only during generation
    and is never written to the public sales table.
    """

    dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="MS",
    )

    sales_rows = []
    event_rows = []

    for _, product in products_df.iterrows():

        part_id = product["PART_ID"]
        category = product["CATEGORY"]
        subcategory = product["SUBCATEGORY"]
        profile = product["PROFILE"]

        launch_date = product["LAUNCH_DATE"]
        preferred_area = product["PREFERRED_MICROAREA"]

        base_demand = get_profile_parameters(
            profile=profile,
            category=category,
            subcategory=subcategory,
            rng=rng,
        )

        history = []

        reactivation_count = 0
        maintenance_count = 0

        for date in dates:

            # ------------------------------------------------
            # PRODUCT LIFECYCLE
            # ------------------------------------------------

            if date < launch_date:
                continue

            demand = generate_base_demand(
                profile=profile,
                base_demand=base_demand,
                date=date,
                rng=rng,
            )

            # ------------------------------------------------
            # NEW PRODUCT
            # ------------------------------------------------

            is_first_month = date == launch_date

            if (
                is_first_month
                and demand > 0
            ):
                event_type = "NEW_PRODUCT"

            else:
                demand, event_type = choose_event(
                    profile=profile,
                    date=date,
                    demand=demand,
                    history=history,
                    reactivation_count=reactivation_count,
                    maintenance_count=maintenance_count,
                    rng=rng,
                )

            # ------------------------------------------------
            # TRACK EVENT COUNTS
            # ------------------------------------------------

            if event_type == "REACTIVATION":
                reactivation_count += 1

            elif event_type == "MAINTENANCE_EVENT":
                maintenance_count += 1

            # ------------------------------------------------
            # MICROAREA
            # ------------------------------------------------

            microarea = choose_microarea(
                preferred_area,
                rng,
            )

            sales_rows.append(
                {
                    "PART_ID": part_id,
                    "DATE": date,
                    "DEMAND": int(max(demand, 0)),
                    "MICROAREA": microarea,
                }
            )

            if event_type is not None:
                event_rows.append(
                    {
                        "PART_ID": part_id,
                        "DATE": date,
                        "EVENT_TYPE": event_type,
                    }
                )

            history.append(
                int(max(demand, 0))
            )

    sales_df = pd.DataFrame(
        sales_rows
    )

    ground_truth_df = pd.DataFrame(
        event_rows
    )

    return sales_df, ground_truth_df


# ============================================================
# EMERGENCY REPLACEMENT EVENTS
# ============================================================

def inject_emergency_events(
    sales_df,
    ground_truth_df,
    products_df,
    rng,
    target_events=40,
):
    """
    Add a controlled number of emergency replacement events.

    Emergency events are selected from sparse-demand products
    and from rows that do not already contain another event.
    """

    product_profiles = products_df[
        ["PART_ID", "PROFILE"]
    ].copy()

    merged = sales_df.merge(
        product_profiles,
        on="PART_ID",
        how="left",
    )

    existing_events = ground_truth_df[
        ["PART_ID", "DATE"]
    ].drop_duplicates()

    merged = merged.merge(
        existing_events.assign(
            HAS_EVENT=True
        ),
        on=["PART_ID", "DATE"],
        how="left",
    )

    # Prefer sparse products because emergency replacement
    # is more plausible for intermittent / lumpy / rare items.
    candidates = merged[
        merged["PROFILE"].isin(
            ["INTERMITTENT", "LUMPY", "RARE"]
        )
        & merged["HAS_EVENT"].isna()
        & (merged["DATE"] > merged.groupby("PART_ID")["DATE"].transform("min"))
    ].copy()

    if len(candidates) < target_events:
        target_events = len(candidates)

    if target_events == 0:
        return sales_df, ground_truth_df

    selected = candidates.sample(
        n=target_events,
        random_state=RANDOM_SEED,
    )

    sales_df = sales_df.copy()
    ground_truth_df = ground_truth_df.copy()

    for _, selected_row in selected.iterrows():

        mask = (
            (sales_df["PART_ID"] == selected_row["PART_ID"])
            & (sales_df["DATE"] == selected_row["DATE"])
        )

        current_demand = int(
            sales_df.loc[mask, "DEMAND"].iloc[0]
        )

        emergency_demand = max(
            current_demand,
            int(
                round(
                    max(
                        2.0,
                        current_demand
                    )
                    * rng.uniform(4.0, 8.0)
                )
            ),
        )

        sales_df.loc[
            mask,
            "DEMAND"
        ] = emergency_demand

        ground_truth_df.loc[
            len(ground_truth_df)
        ] = {
            "PART_ID": selected_row["PART_ID"],
            "DATE": selected_row["DATE"],
            "EVENT_TYPE": "EMERGENCY_REPLACEMENT",
        }

    return sales_df, ground_truth_df


# ============================================================
# VALIDATION
# ============================================================

def validate_data(
    products_df,
    sales_df,
    ground_truth_df,
):
    """
    Run basic integrity checks.
    """

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    # Product count.
    assert len(products_df) == N_PRODUCTS

    # Unique product IDs.
    assert products_df["PART_ID"].is_unique

    # Valid demand.
    assert sales_df["DEMAND"].notna().all()
    assert (sales_df["DEMAND"] >= 0).all()

    # No duplicate monthly sales rows.
    duplicate_sales = sales_df.duplicated(
        subset=[
            "PART_ID",
            "DATE",
            "MICROAREA",
        ]
    ).sum()

    assert duplicate_sales == 0

    # Date range.
    assert sales_df["DATE"].min() >= pd.Timestamp(
        START_DATE
    )

    assert sales_df["DATE"].max() <= pd.Timestamp(
        END_DATE
    )

    # Ground truth dates must exist in sales.
    sales_keys = set(
        zip(
            sales_df["PART_ID"],
            sales_df["DATE"],
        )
    )

    ground_truth_keys = set(
        zip(
            ground_truth_df["PART_ID"],
            ground_truth_df["DATE"],
        )
    )

    assert ground_truth_keys.issubset(
        sales_keys
    )

    # No duplicate ground-truth events.
    assert not ground_truth_df.duplicated(
        subset=[
            "PART_ID",
            "DATE",
        ]
    ).any()

    # Exact profile distribution.
    actual_profiles = (
        products_df["PROFILE"]
        .value_counts()
        .to_dict()
    )

    assert actual_profiles == PROFILE_COUNTS

    print(
        f"Products: {len(products_df):,}"
    )

    print(
        f"Sales rows: {len(sales_df):,}"
    )

    print(
        f"Ground-truth events: "
        f"{len(ground_truth_df):,}"
    )

    print(
        "Validation: PASSED"
    )


# ============================================================
# PROFILE BEHAVIOR VALIDATION
# ============================================================

def validate_profile_behavior(
    products_df,
    sales_df,
):
    """
    Check whether the generated demand profiles actually
    behave differently.
    """

    print("\n" + "=" * 70)
    print("DEMAND BEHAVIOR BY PROFILE")
    print("=" * 70)

    profile_sales = sales_df.merge(
        products_df[
            ["PART_ID", "PROFILE"]
        ],
        on="PART_ID",
        how="left",
    )

    summary = (
        profile_sales
        .groupby("PROFILE")["DEMAND"]
        .agg(
            rows="count",
            mean="mean",
            median="median",
            max="max",
            zero_share=lambda x: (
                x.eq(0).mean()
            ),
        )
        .round(2)
    )

    print(summary)

    print("\nExpected behavior:")
    print(
        "- STABLE: low zero share, relatively stable demand"
    )
    print(
        "- SEASONAL: recurring monthly demand pattern"
    )
    print(
        "- ERRATIC: frequent demand with high variability"
    )
    print(
        "- INTERMITTENT: many zero-demand months"
    )
    print(
        "- LUMPY: many zeros with larger positive orders"
    )
    print(
        "- RARE: very sparse positive demand"
    )


# ============================================================
# REPORT
# ============================================================

def print_report(
    products_df,
    sales_df,
    ground_truth_df,
):
    """
    Print a compact summary of the generated dataset.
    """

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Products: {len(products_df):,}"
    )

    print(
        f"Sales rows: {len(sales_df):,}"
    )

    print(
        f"Date range: "
        f"{sales_df['DATE'].min().date()} "
        f"to "
        f"{sales_df['DATE'].max().date()}"
    )

    print(
        f"Demand mean: "
        f"{sales_df['DEMAND'].mean():.2f}"
    )

    print(
        f"Demand median: "
        f"{sales_df['DEMAND'].median():.2f}"
    )

    print(
        f"Demand std: "
        f"{sales_df['DEMAND'].std():.2f}"
    )

    print(
        f"Demand max: "
        f"{sales_df['DEMAND'].max():,}"
    )

    print(
        f"Zero-demand share: "
        f"{(sales_df['DEMAND'] == 0).mean():.1%}"
    )

    print("\nPROFILE DISTRIBUTION")

    print(
        products_df["PROFILE"]
        .value_counts()
        .sort_index()
    )

    print("\nEVENT DISTRIBUTION")

    print(
        ground_truth_df["EVENT_TYPE"]
        .value_counts()
        .sort_index()
    )

    print("\nDEMAND DISTRIBUTION")

    print(
        sales_df["DEMAND"]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(2)
    )

    print("\nDEMAND BY CATEGORY")

    category_sales = sales_df.merge(
        products_df[
            [
                "PART_ID",
                "CATEGORY",
            ]
        ],
        on="PART_ID",
        how="left",
    )

    category_summary = (
        category_sales
        .groupby("CATEGORY")["DEMAND"]
        .agg(
            rows="count",
            mean="mean",
            median="median",
            zero_share=lambda x: (
                x.eq(0).mean()
            ),
            max="max",
        )
        .sort_values(
            "mean",
            ascending=False,
        )
        .round(2)
    )

    print(category_summary)


# ============================================================
# SAVE DATA
# ============================================================

def save_data(
    products_df,
    sales_df,
    ground_truth_df,
):
    """
    Save public dataset files.

    Internal generation fields are removed from products.csv.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # PUBLIC PRODUCTS TABLE
    # --------------------------------------------------------

    public_products = products_df[
        [
            "PART_ID",
            "DESCRIPTION",
            "CATEGORY",
            "SUBCATEGORY",
        ]
    ].copy()

    public_products.to_csv(
        PRODUCTS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # PUBLIC SALES TABLE
    # --------------------------------------------------------

    public_sales = sales_df[
        [
            "PART_ID",
            "DATE",
            "DEMAND",
            "MICROAREA",
        ]
    ].copy()

    public_sales.to_csv(
        SALES_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    ground_truth_df = (
        ground_truth_df
        .sort_values(
            [
                "PART_ID",
                "DATE",
            ]
        )
        .reset_index(drop=True)
    )

    ground_truth_df.to_csv(
        GROUND_TRUTH_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Products:      {PRODUCTS_FILE}"
    )

    print(
        f"Sales:         {SALES_FILE}"
    )

    print(
        f"Ground truth:  {GROUND_TRUTH_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main data-generation pipeline.
    """

    print("=" * 70)
    print("MEDICAL EQUIPMENT DEMAND DATA GENERATOR")
    print("=" * 70)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print("\nGenerating products...")

    products_df = generate_products(
        rng
    )

    print(
        f"Generated {len(products_df):,} products."
    )

    print("\nGenerating monthly sales...")

    sales_df, ground_truth_df = generate_sales(
        products_df,
        rng,
    )

    print(
        f"Generated {len(sales_df):,} sales rows."
    )

    print("\nInjecting emergency replacement events...")

    sales_df, ground_truth_df = inject_emergency_events(
        sales_df=sales_df,
        ground_truth_df=ground_truth_df,
        products_df=products_df,
        rng=rng,
        target_events=40,
    )

    print(
        "Emergency replacement events added."
    )

    validate_data(
        products_df,
        sales_df,
        ground_truth_df,
    )

    validate_profile_behavior(
        products_df,
        sales_df,
    )

    print_report(
        products_df,
        sales_df,
        ground_truth_df,
    )

    save_data(
        products_df,
        sales_df,
        ground_truth_df,
    )

    print("\nGeneration completed successfully.")


if __name__ == "__main__":
    main()
