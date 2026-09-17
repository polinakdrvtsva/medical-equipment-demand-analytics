-- ============================================================
-- MEDICAL EQUIPMENT DEMAND ANALYTICS
-- DATABASE SCHEMA
-- ============================================================

-- ------------------------------------------------------------
-- PRODUCTS
-- ------------------------------------------------------------

DROP TABLE IF EXISTS ground_truth;
DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS products;


CREATE TABLE products (
    part_id VARCHAR(20) PRIMARY KEY,
    description VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100) NOT NULL
);


-- ------------------------------------------------------------
-- SALES
-- ------------------------------------------------------------

CREATE TABLE sales (
    part_id VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    demand INTEGER NOT NULL,
    microarea VARCHAR(50) NOT NULL,

    CONSTRAINT fk_sales_product
        FOREIGN KEY (part_id)
        REFERENCES products(part_id),

    CONSTRAINT chk_sales_demand_non_negative
        CHECK (demand >= 0),

    CONSTRAINT chk_sales_microarea
        CHECK (
            microarea IN (
                'NORTH',
                'SOUTH',
                'EAST',
                'WEST',
                'CENTRAL'
            )
        ),

    CONSTRAINT pk_sales
        PRIMARY KEY (part_id, date)
);


-- ------------------------------------------------------------
-- GROUND TRUTH
-- Used only for validating the anomaly detection logic.
-- It should not be used as an input feature in the analysis.
-- ------------------------------------------------------------

CREATE TABLE ground_truth (
    part_id VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    event_type VARCHAR(50) NOT NULL,

    CONSTRAINT fk_ground_truth_product
        FOREIGN KEY (part_id)
        REFERENCES products(part_id),

    CONSTRAINT chk_ground_truth_event
        CHECK (
            event_type IN (
                'NEW_PRODUCT',
                'LARGE_ORDER',
                'REACTIVATION',
                'MAINTENANCE_EVENT',
                'EMERGENCY_REPLACEMENT'
            )
        ),

    CONSTRAINT pk_ground_truth
        PRIMARY KEY (part_id, date)
);


-- ------------------------------------------------------------
-- INDEXES
-- ------------------------------------------------------------

CREATE INDEX idx_sales_date
    ON sales(date);

CREATE INDEX idx_sales_microarea
    ON sales(microarea);

CREATE INDEX idx_sales_part_date
    ON sales(part_id, date);

CREATE INDEX idx_ground_truth_event_type
    ON ground_truth(event_type);


-- ------------------------------------------------------------
-- SCHEMA SUMMARY
-- ------------------------------------------------------------

-- products:
--   Product master data.
--
-- sales:
--   Monthly demand by product and microarea.
--
-- ground_truth:
--   Synthetic event labels used only for validating
--   anomaly detection performance.