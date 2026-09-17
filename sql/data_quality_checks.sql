-- ============================================================
-- MEDICAL EQUIPMENT DEMAND ANALYTICS
-- DATA QUALITY CHECKS
-- ============================================================


-- ============================================================
-- 1. BASIC TABLE COUNTS
-- ============================================================

SELECT
    'products' AS table_name,
    COUNT(*) AS row_count
FROM products

UNION ALL

SELECT
    'sales',
    COUNT(*)
FROM sales

UNION ALL

SELECT
    'ground_truth',
    COUNT(*)
FROM ground_truth;


-- ============================================================
-- 2. NULL VALUES
-- ============================================================

SELECT
    'products.part_id' AS field,
    COUNT(*) AS null_count
FROM products
WHERE part_id IS NULL

UNION ALL

SELECT
    'products.description',
    COUNT(*)
FROM products
WHERE description IS NULL

UNION ALL

SELECT
    'products.category',
    COUNT(*)
FROM products
WHERE category IS NULL

UNION ALL

SELECT
    'products.subcategory',
    COUNT(*)
FROM products
WHERE subcategory IS NULL

UNION ALL

SELECT
    'sales.part_id',
    COUNT(*)
FROM sales
WHERE part_id IS NULL

UNION ALL

SELECT
    'sales.date',
    COUNT(*)
FROM sales
WHERE date IS NULL

UNION ALL

SELECT
    'sales.demand',
    COUNT(*)
FROM sales
WHERE demand IS NULL

UNION ALL

SELECT
    'sales.microarea',
    COUNT(*)
FROM sales
WHERE microarea IS NULL;


-- ============================================================
-- 3. DUPLICATE PRODUCTS
-- ============================================================

SELECT
    part_id,
    COUNT(*) AS occurrences
FROM products
GROUP BY part_id
HAVING COUNT(*) > 1
ORDER BY occurrences DESC;


-- ============================================================
-- 4. DUPLICATE PRODUCT / MONTH RECORDS
-- ============================================================

SELECT
    part_id,
    date,
    COUNT(*) AS occurrences
FROM sales
GROUP BY
    part_id,
    date
HAVING COUNT(*) > 1
ORDER BY occurrences DESC;


-- ============================================================
-- 5. NEGATIVE DEMAND
-- ============================================================

SELECT
    COUNT(*) AS negative_demand_rows
FROM sales
WHERE demand < 0;


-- ============================================================
-- 6. INVALID MICROAREAS
-- ============================================================

SELECT DISTINCT
    microarea
FROM sales
WHERE microarea NOT IN (
    'NORTH',
    'SOUTH',
    'EAST',
    'WEST',
    'CENTRAL'
);


-- ============================================================
-- 7. SALES WITHOUT MATCHING PRODUCT
-- ============================================================

SELECT
    COUNT(*) AS unmatched_sales_rows
FROM sales s
LEFT JOIN products p
    ON s.part_id = p.part_id
WHERE p.part_id IS NULL;


-- ============================================================
-- 8. GROUND-TRUTH EVENTS WITHOUT MATCHING PRODUCT
-- ============================================================

SELECT
    COUNT(*) AS unmatched_ground_truth_products
FROM ground_truth g
LEFT JOIN products p
    ON g.part_id = p.part_id
WHERE p.part_id IS NULL;


-- ============================================================
-- 9. GROUND-TRUTH EVENTS WITHOUT MATCHING SALES
-- ============================================================

SELECT
    COUNT(*) AS unmatched_ground_truth_sales
FROM ground_truth g
LEFT JOIN sales s
    ON g.part_id = s.part_id
    AND g.date = s.date
WHERE s.part_id IS NULL;


-- ============================================================
-- 10. SALES DATE RANGE
-- ============================================================

SELECT
    MIN(date) AS min_date,
    MAX(date) AS max_date,
    COUNT(DISTINCT date) AS distinct_months
FROM sales;


-- ============================================================
-- 11. PRODUCTS WITH SALES
-- ============================================================

SELECT
    COUNT(DISTINCT part_id) AS products_with_sales
FROM sales;


-- ============================================================
-- 12. PRODUCTS WITHOUT SALES
-- ============================================================

SELECT
    COUNT(*) AS products_without_sales
FROM products p
LEFT JOIN (
    SELECT DISTINCT part_id
    FROM sales
) s
    ON p.part_id = s.part_id
WHERE s.part_id IS NULL;


-- ============================================================
-- 13. DEMAND SUMMARY
-- ============================================================

SELECT
    COUNT(*) AS sales_rows,
    SUM(demand) AS total_demand,
    ROUND(AVG(demand), 2) AS avg_demand,
    PERCENTILE_CONT(0.50)
        WITHIN GROUP (ORDER BY demand) AS median_demand,
    MAX(demand) AS max_demand,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN demand = 0 THEN 1.0
                ELSE 0.0
            END
        ),
        2
    ) AS zero_demand_pct
FROM sales;


-- ============================================================
-- 14. GROUND-TRUTH EVENT TYPES
-- ============================================================

SELECT
    event_type,
    COUNT(*) AS event_count
FROM ground_truth
GROUP BY event_type
ORDER BY event_count DESC;