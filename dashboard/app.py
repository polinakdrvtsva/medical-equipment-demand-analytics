from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import DB_CONNECTION_STRING


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Medical Equipment Demand Analytics",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 2rem;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Database connection
# ============================================================

@st.cache_resource
def get_engine():
    return create_engine(DB_CONNECTION_STRING)


@st.cache_data
def load_data():
    query = """
        SELECT
            part_id,
            date,
            description,
            category,
            subcategory,
            microarea,
            demand,
            profile,
            upper_bound,
            outlier_method,
            anomaly_score,
            is_reactivation,
            is_anomaly,
            anomaly_type
        FROM analytics_demand_output
        ORDER BY date, part_id;
    """

    engine = get_engine()

    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])

    return df


df = load_data()


# ============================================================
# Header
# ============================================================

st.title("Medical Equipment Demand Analytics")

st.markdown(
    """
    **Demand monitoring and exceptional demand detection**

    Explore demand patterns across products, categories and service areas,
    and identify demand that is unusually high compared with a product's
    historical behaviour.

    Built as a reproducible analytics pipeline using **PostgreSQL, SQL,
    Python and Streamlit**.
    """
)


# ============================================================
# About this analysis
# ============================================================

with st.expander("About this analysis"):

    st.markdown(
        """
        This dashboard uses a **synthetic dataset** representing monthly
        demand for medical equipment spare parts from 2022 to 2025.

        Spare parts can have very different demand patterns. Some products
        have **stable demand**, while others are **intermittent, lumpy,
        or highly variable**. Because of this, unusual demand cannot be
        identified using one common threshold for all products.

        Each product is assigned a demand profile based on how frequently
        it is requested and how much its demand varies over time. A
        **profile-specific historical baseline** is then used to identify
        demand that is unusually high compared with the product's own
        past behaviour.

        The analysis focuses on **exceptional demand** — cases where
        observed demand is significantly above the level expected for
        that particular product. Reactivation of products after a period
        of no demand is tracked separately as an additional signal.

        **Demand profiles**

        - **Stable** — demand is relatively consistent over time.
        - **Intermittent** — demand occurs with frequent periods of zero demand.
        - **Lumpy** — demand is infrequent and varies considerably in size.
        - **Erratic** — demand is relatively frequent but highly variable.

        The dataset is synthetic and was created for portfolio demonstration
        purposes. Ground-truth events were generated during data creation
        and are used separately to validate the detection approach.
        """
    )


# ============================================================
# Sidebar filters
# ============================================================

st.sidebar.header("Filters")

st.sidebar.caption(
    "Use the filters below to explore demand patterns "
    "and exceptional events."
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

categories = st.sidebar.multiselect(
    "Category",
    options=sorted(df["category"].dropna().unique()),
)

microareas = st.sidebar.multiselect(
    "Microarea",
    options=sorted(df["microarea"].dropna().unique()),
)

profiles = st.sidebar.multiselect(
    "Demand profile",
    options=sorted(df["profile"].dropna().unique()),
)


# ============================================================
# Apply filters
# ============================================================

filtered_df = df.copy()

if len(date_range) == 2:

    start_date, end_date = date_range

    filtered_df = filtered_df[
        (filtered_df["date"].dt.date >= start_date)
        & (filtered_df["date"].dt.date <= end_date)
    ]

if categories:

    filtered_df = filtered_df[
        filtered_df["category"].isin(categories)
    ]

if microareas:

    filtered_df = filtered_df[
        filtered_df["microarea"].isin(microareas)
    ]

if profiles:

    filtered_df = filtered_df[
        filtered_df["profile"].isin(profiles)
    ]


# ============================================================
# Product selector
# ============================================================

st.sidebar.divider()

st.sidebar.header("Product Analysis")

product_options = (
    filtered_df[
        ["part_id", "description"]
    ]
    .drop_duplicates()
    .sort_values("part_id")
)

product_labels = {
    row["part_id"]: f'{row["part_id"]} — {row["description"]}'
    for _, row in product_options.iterrows()
}

selected_product = st.sidebar.selectbox(
    "Select a product",
    options=["None"] + list(product_labels.keys()),
    format_func=lambda x: (
        "Select a product..."
        if x == "None"
        else product_labels[x]
    ),
)


# ============================================================
# KPI calculations
# ============================================================

total_demand = filtered_df["demand"].sum()

active_parts = filtered_df.loc[
    filtered_df["demand"] > 0,
    "part_id",
].nunique()

exceptional_events = filtered_df["is_anomaly"].sum()

anomaly_rate = (
    exceptional_events / len(filtered_df) * 100
    if len(filtered_df) > 0
    else 0
)


# ============================================================
# KPI cards
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Demand",
    f"{total_demand:,.0f}",
)

col2.metric(
    "Active Parts",
    f"{active_parts:,}",
)

col3.metric(
    "Exceptional Events",
    f"{exceptional_events:,}",
)

col4.metric(
    "Exceptional Demand Rate",
    f"{anomaly_rate:.2f}%",
)


# ============================================================
# Product demand history
# ============================================================

st.divider()

st.subheader("Product Demand History")

if selected_product == "None":

    st.info(
        "Select a product in the sidebar to view its demand history "
        "and exceptional demand events."
    )

else:

    product_df = filtered_df[
        filtered_df["part_id"] == selected_product
    ].copy()

    product_df = product_df.sort_values("date")

    product_info = product_df.iloc[0]

    st.markdown(
        f"**{product_info['part_id']} — "
        f"{product_info['description']}**  \n"
        f"Category: {product_info['category']} · "
        f"Subcategory: {product_info['subcategory']} · "
        f"Demand profile: {product_info['profile']}"
    )

    # Normal demand
    normal_df = product_df[
        ~product_df["is_anomaly"]
    ].copy()

    # Exceptional demand
    exceptional_df = product_df[
        product_df["is_anomaly"]
    ].copy()

    fig_product = px.line(
        product_df,
        x="date",
        y="demand",
        markers=True,
        labels={
            "date": "Date",
            "demand": "Demand",
        },
        hover_data={
            "upper_bound": ":.1f",
            "anomaly_score": ":.2f",
            "anomaly_type": True,
        },
    )

    fig_product.update_traces(
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "Demand: %{y}<br>"
            "<extra></extra>"
        ),
        selector=dict(mode="lines+markers"),
    )

    fig_product.add_scatter(
        x=product_df["date"],
        y=product_df["upper_bound"],
        mode="lines",
        name="Expected upper level",
        line={
            "dash": "dash",
        },
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "Expected upper level: %{y:.1f}"
            "<extra></extra>"
        ),
    )

    # Highlight exceptional observations
    if not exceptional_df.empty:

        fig_product.add_scatter(
            x=exceptional_df["date"],
            y=exceptional_df["demand"],
            mode="markers",
            marker={
                "size": 11,
                "symbol": "circle",
            },
            name="Exceptional demand",
            customdata=exceptional_df[
                [
                    "upper_bound",
                    "anomaly_score",
                    "anomaly_type",
                ]
            ],
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>"
                "Demand: %{y}<br>"
                "Expected upper level: %{customdata[0]:.1f}<br>"
                "Anomaly score: %{customdata[1]:.2f}<br>"
                "Type: %{customdata[2]}"
                "<extra></extra>"
            ),
        )

    fig_product.update_layout(
        legend_title_text="",
    )

    st.plotly_chart(
        fig_product,
        width="stretch",
    )

    if not exceptional_df.empty:

        st.caption(
            f"{len(exceptional_df)} exceptional demand event(s) "
            f"detected for this product in the selected period."
        )

    else:

        st.caption(
            "No exceptional demand events were detected for this "
            "product in the selected period."
        )


# ============================================================
# Business overview
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Demand by category
# ------------------------------------------------------------

with col1:

    st.subheader("Demand by Category")

    category_df = (
        filtered_df
        .groupby("category", as_index=False)
        .agg(
            total_demand=("demand", "sum"),
            exceptional_events=("is_anomaly", "sum"),
        )
        .sort_values(
            "total_demand",
            ascending=True,
        )
    )

    fig_category = px.bar(
        category_df,
        x="total_demand",
        y="category",
        orientation="h",
        labels={
            "total_demand": "Total Demand",
            "category": "",
        },
        hover_data={
            "exceptional_events": True,
            "total_demand": ":,.0f",
        },
    )

    st.plotly_chart(
        fig_category,
        width="stretch",
    )


# ------------------------------------------------------------
# Exceptional events by microarea
# ------------------------------------------------------------

with col2:

    st.subheader("Exceptional Events by Microarea")

    microarea_df = (
        filtered_df
        .groupby("microarea", as_index=False)
        .agg(
            exceptional_events=("is_anomaly", "sum"),
        )
        .sort_values(
            "exceptional_events",
            ascending=True,
        )
    )

    fig_microarea = px.bar(
        microarea_df,
        x="exceptional_events",
        y="microarea",
        orientation="h",
        labels={
            "exceptional_events": "Exceptional Events",
            "microarea": "",
        },
    )

    st.plotly_chart(
        fig_microarea,
        width="stretch",
    )


# ------------------------------------------------------------
# Exceptional events by category
# ------------------------------------------------------------

col1, col2 = st.columns(2)


with col1:

    st.subheader("Exceptional Events by Category")

    exceptional_category_df = (
        filtered_df
        .groupby("category", as_index=False)
        .agg(
            exceptional_events=("is_anomaly", "sum"),
        )
        .sort_values(
            "exceptional_events",
            ascending=True,
        )
    )

    fig_exceptional_category = px.bar(
        exceptional_category_df,
        x="exceptional_events",
        y="category",
        orientation="h",
        labels={
            "exceptional_events": "Exceptional Events",
            "category": "",
        },
    )

    st.plotly_chart(
        fig_exceptional_category,
        width="stretch",
    )


# ------------------------------------------------------------
# Demand profile distribution
# ------------------------------------------------------------

with col2:

    st.subheader("Demand Profile Distribution")

    profile_df = (
        filtered_df
        .groupby("profile", as_index=False)
        .agg(
            observations=("part_id", "size"),
            products=("part_id", "nunique"),
        )
        .sort_values(
            "observations",
            ascending=True,
        )
    )

    fig_profile = px.bar(
        profile_df,
        x="observations",
        y="profile",
        orientation="h",
        labels={
            "observations": "Observations",
            "profile": "",
        },
        hover_data={
            "products": True,
            "observations": ":,.0f",
        },
    )

    st.plotly_chart(
        fig_profile,
        width="stretch",
    )


# ============================================================
# Exceptional demand explorer
# ============================================================

st.divider()

st.subheader("Exceptional Demand Explorer")

anomalies_df = filtered_df[
    filtered_df["is_anomaly"]
].copy()

anomalies_df = anomalies_df.sort_values(
    "anomaly_score",
    ascending=False,
)

display_columns = [
    "date",
    "part_id",
    "category",
    "subcategory",
    "microarea",
    "demand",
    "upper_bound",
    "anomaly_score",
    "anomaly_type",
]

if anomalies_df.empty:

    st.info(
        "No exceptional demand events match the selected filters."
    )

else:

    display_df = anomalies_df[display_columns].rename(
        columns={
            "date": "Date",
            "part_id": "Part ID",
            "category": "Category",
            "subcategory": "Subcategory",
            "microarea": "Microarea",
            "demand": "Demand",
            "upper_bound": "Expected Upper Level",
            "anomaly_score": "Anomaly Score",
            "anomaly_type": "Anomaly Type",
        }
    )

    display_df["Date"] = display_df["Date"].dt.strftime(
        "%b %Y"
    )

    display_df["Expected Upper Level"] = (
        display_df["Expected Upper Level"].round(1)
    )

    display_df["Anomaly Score"] = (
        display_df["Anomaly Score"].round(2)
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )
