-- ============================================================
-- MEDICAL EQUIPMENT DEMAND ANALYTICS
-- ANALYTICAL SQL QUERIES
-- ============================================================


-- ============================================================
-- 1. MONTHLY DEMAND TREND
-- ============================================================

SELECT
    date,
    SUM(demand) AS total_demand,
    COUNT(DISTINCT part_id) AS active_products,
    ROUND(AVG(demand), 2) AS avg_demand
FROM sales
GROUP BY date
ORDER BY date;


-- ============================================================
-- 2. DEMAND BY CATEGORY
-- ============================================================

SELECT
    p.category,
    SUM(s.demand) AS total_demand,
    ROUND(AVG(s.demand), 2) AS avg_monthly_demand,
    COUNT(DISTINCT s.part_id) AS products
FROM sales s
JOIN products p
    ON s.part_id = p.part_id
GROUP BY p.category
ORDER BY total_demand DESC;


-- ============================================================
-- 3. DEMAND BY MICROAREA
-- ============================================================

SELECT
    microarea,
    SUM(demand) AS total_demand,
    ROUND(AVG(demand), 2) AS avg_demand,
    COUNT(DISTINCT part_id) AS products
FROM sales
GROUP BY microarea
ORDER BY total_demand DESC;


-- ============================================================
-- 4. CATEGORY × MICROAREA
-- ============================================================

SELECT
    p.category,
    s.microarea,
    SUM(s.demand) AS total_demand,
    ROUND(AVG(s.demand), 2) AS avg_demand
FROM sales s
JOIN products p
    ON s.part_id = p.part_id
GROUP BY
    p.category,
    s.microarea
ORDER BY
    p.category,
    total_demand DESC;


-- ============================================================
-- 5. TOP 20 PRODUCTS BY TOTAL DEMAND
-- ============================================================

SELECT
    p.part_id,
    p.description,
    p.category,
    p.subcategory,
    SUM(s.demand) AS total_demand,
    ROUND(AVG(s.demand), 2) AS avg_monthly_demand
FROM sales s
JOIN products p
    ON s.part_id = p.part_id
GROUP BY
    p.part_id,
    p.description,
    p.category,
    p.subcategory
ORDER BY total_demand DESC
LIMIT 20;


-- ============================================================
-- 6. PRODUCTS WITH HIGHEST ZERO-DEMAND SHARE
-- ============================================================

SELECT
    p.part_id,
    p.description,
    p.category,
    COUNT(*) AS months_observed,
    SUM(
        CASE
            WHEN s.demand = 0 THEN 1
            ELSE 0
        END
    ) AS zero_demand_months,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN s.demand = 0 THEN 1.0
                ELSE 0.0
            END
        ),
        2
    ) AS zero_demand_pct,
    SUM(s.demand) AS total_demand
FROM sales s
JOIN products p
    ON s.part_id = p.part_id
GROUP BY
    p.part_id,
    p.description,
    p.category
HAVING COUNT(*) >= 12
ORDER BY zero_demand_pct DESC
LIMIT 20;


-- ============================================================
-- 7. DEMAND VARIABILITY BY PRODUCT
-- ============================================================

SELECT
    p.part_id,
    p.description,
    p.category,
    ROUND(AVG(s.demand), 2) AS avg_demand,
    ROUND(STDDEV(s.demand), 2) AS demand_std,
    ROUND(
        STDDEV(s.demand)
        / NULLIF(AVG(s.demand), 0),
        2
    ) AS coefficient_of_variation,
    MAX(s.demand) AS max_demand
FROM sales s
JOIN products p
    ON s.part_id = p.part_id
GROUP BY
    p.part_id,
    p.description,
    p.category
HAVING AVG(s.demand) > 0
ORDER BY coefficient_of_variation DESC
LIMIT 20;


-- ============================================================
-- 8. MONTHLY SEASONALITY
-- ============================================================

SELECT
    EXTRACT(
        MONTH FROM date
    ) AS month_number,
    TO_CHAR(
        date,
        'Month'
    ) AS month_name,
    SUM(demand) AS total_demand,
    ROUND(
        AVG(demand),
        2
    ) AS avg_demand
FROM sales
GROUP BY
    EXTRACT(MONTH FROM date),
    TO_CHAR(date, 'Month')
ORDER BY month_number;


-- ============================================================
-- 9. PRODUCT LIFECYCLE
-- ============================================================

SELECT
    p.part_id,
    p.description,
    p.category,
    MIN(s.date) AS first_observed_month,
    MAX(s.date) AS last_observed_month,
    COUNT(*) AS months_observed,
    SUM(s.demand) AS total_demand
FROM sales s
JOIN products p
    ON s.part_id = p.part_id
GROUP BY
    p.part_id,
    p.description,
    p.category
ORDER BY first_observed_month, p.part_id;


-- ============================================================
-- 10. EVENT IMPACT
-- ============================================================

SELECT
    g.event_type,
    COUNT(*) AS event_count,
    ROUND(
        AVG(s.demand),
        2
    ) AS avg_event_demand,
    MAX(s.demand) AS max_event_demand,
    SUM(s.demand) AS total_event_demand
FROM ground_truth g
JOIN sales s
    ON g.part_id = s.part_id
    AND g.date = s.date
GROUP BY g.event_type
ORDER BY event_count DESC;


-- ============================================================
-- 11. DEMAND DURING EVENT VS NORMAL MONTHS
-- ============================================================

SELECT
    CASE
        WHEN g.event_type IS NOT NULL
            THEN 'EVENT'
        ELSE 'NORMAL'
    END AS demand_type,
    COUNT(*) AS rows_count,
    ROUND(
        AVG(s.demand),
        2
    ) AS avg_demand,
    PERCENTILE_CONT(0.50)
        WITHIN GROUP (
            ORDER BY s.demand
        ) AS median_demand,
    MAX(s.demand) AS max_demand
FROM sales s
LEFT JOIN ground_truth g
    ON s.part_id = g.part_id
    AND s.date = g.date
GROUP BY
    CASE
        WHEN g.event_type IS NOT NULL
            THEN 'EVENT'
        ELSE 'NORMAL'
    END
ORDER BY demand_type;


-- ============================================================
-- 12. TOP PRODUCTS WITH EVENT ACTIVITY
-- ============================================================

SELECT
    p.part_id,
    p.description,
    p.category,
    COUNT(g.event_type) AS event_count,
    SUM(s.demand) AS total_demand,
    ROUND(
        AVG(s.demand),
        2
    ) AS avg_demand
FROM products p
JOIN sales s
    ON p.part_id = s.part_id
LEFT JOIN ground_truth g
    ON s.part_id = g.part_id
    AND s.date = g.date
GROUP BY
    p.part_id,
    p.description,
    p.category
HAVING COUNT(g.event_type) > 0
ORDER BY
    event_count DESC,
    total_demand DESC
LIMIT 20;