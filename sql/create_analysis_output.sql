-- ============================================================
-- Final analytics output
-- ============================================================

DROP TABLE IF EXISTS analytics_demand_output;

CREATE TABLE analytics_demand_output AS
SELECT
    d.part_id,
    d.date,
    p.description,
    p.category,
    p.subcategory,
    d.microarea,
    d.demand,

    -- Demand profile
    d.profile,

    -- Historical baseline
    d.history_months,
    d.history_before_current,
    d.previous_demand,
    d.baseline_median,
    d.baseline_q1,
    d.baseline_q3,
    d.baseline_iqr,
    d.upper_bound,

    -- Detection results
    d.outlier_method,
    d.is_quantity_outlier,
    d.anomaly_score,
    d.is_anomaly,
    d.anomaly_type,

    -- Reactivation signal
    d.is_reactivation,
    d.unusual_gap,
    d.months_since_previous_demand,

    -- Yearly validation
    d.previous_yearly_demand,
    d.yearly_ratio,
    d.passes_yearly_validation

FROM detection_results d
LEFT JOIN products p
    ON d.part_id = p.part_id;


-- ============================================================
-- Indexes
-- ============================================================

CREATE INDEX idx_analytics_output_date
    ON analytics_demand_output (date);

CREATE INDEX idx_analytics_output_part
    ON analytics_demand_output (part_id);

CREATE INDEX idx_analytics_output_category
    ON analytics_demand_output (category);

CREATE INDEX idx_analytics_output_microarea
    ON analytics_demand_output (microarea);

CREATE INDEX idx_analytics_output_anomaly
    ON analytics_demand_output (is_anomaly);


-- ============================================================
-- Basic validation
-- ============================================================

SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT part_id) AS unique_parts,
    MIN(date) AS min_date,
    MAX(date) AS max_date,
    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) AS anomalies,
    SUM(CASE WHEN is_reactivation THEN 1 ELSE 0 END)
        AS reactivation_signals
FROM analytics_demand_output;