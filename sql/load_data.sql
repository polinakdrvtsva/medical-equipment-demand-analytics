-- ============================================================
-- MEDICAL EQUIPMENT DEMAND ANALYTICS
-- LOAD SAMPLE DATA
-- ============================================================


-- ------------------------------------------------------------
-- LOAD PRODUCTS
-- ------------------------------------------------------------

\copy products(part_id, description, category, subcategory) FROM 'data/sample/products_sample.csv' WITH (FORMAT csv, HEADER true);


-- ------------------------------------------------------------
-- LOAD SALES
-- ------------------------------------------------------------

\copy sales(part_id, date, demand, microarea) FROM 'data/sample/sales_sample.csv' WITH (FORMAT csv, HEADER true);


-- ------------------------------------------------------------
-- LOAD GROUND TRUTH
-- ------------------------------------------------------------

\copy ground_truth(part_id, date, event_type) FROM 'data/sample/ground_truth.csv' WITH (FORMAT csv, HEADER true);


-- ------------------------------------------------------------
-- BASIC ROW COUNTS
-- ------------------------------------------------------------

SELECT
    'products' AS table_name,
    COUNT(*) AS row_count
FROM products

UNION ALL

SELECT
    'sales' AS table_name,
    COUNT(*) AS row_count
FROM sales

UNION ALL

SELECT
    'ground_truth' AS table_name,
    COUNT(*) AS row_count
FROM ground_truth;