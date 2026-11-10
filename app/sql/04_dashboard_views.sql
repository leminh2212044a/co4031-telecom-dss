-- ======================================================================
-- CO4031 - TELECOM DATA WAREHOUSE & DECISION SUPPORT SYSTEM (DSS)
-- Script: app/sql/04_dashboard_views.sql
-- Description: Các View tổng hợp (Aggregate Data Marts) phục vụ Dashboard OLAP
--              Chạy SAU sql/ddl/03_indexes_and_views.sql
-- Chạy:   psql -U postgres -d telecom_dw -f app/sql/04_dashboard_views.sql
-- Author: Backend & Dashboard Agent (CO4031 - HCMUT)
-- ======================================================================

-- 1. KPI tổng quan (1 dòng / snapshot)
CREATE OR REPLACE VIEW edw.vw_dash_kpi_summary AS
SELECT
    snapshot_date,
    COUNT(*)                                   AS customers,
    SUM(churn_flag)                            AS churned,
    ROUND(AVG(churn_flag::numeric), 4)         AS churn_rate,
    SUM(monthly_charges)                       AS mrr,
    SUM(monthly_charges * churn_flag)          AS revenue_at_risk,
    ROUND(AVG(monthly_charges), 2)             AS avg_monthly_charges,
    SUM(total_charges)                         AS total_revenue,
    ROUND(AVG(tenure_months), 1)               AS avg_tenure,
    ROUND(AVG(clv_proxy), 2)                   AS avg_clv
FROM edw.vw_bi_churn_analytics
GROUP BY snapshot_date;

-- 2. Khối địa lý có tổng con (ROLLUP) : region -> city -> zip_code
CREATE OR REPLACE VIEW edw.vw_dash_churn_geo_rollup AS
SELECT
    COALESCE(region, '∑ ALL')                  AS region,
    COALESCE(city, '∑ ALL')                    AS city,
    COALESCE(zip_code, '∑ ALL')                AS zip_code,
    GROUPING(region, city, zip_code)           AS grouping_id,
    COUNT(*)                                   AS customers,
    SUM(churn_flag)                            AS churned,
    ROUND(AVG(churn_flag::numeric), 4)         AS churn_rate,
    SUM(monthly_charges)                       AS mrr,
    SUM(monthly_charges * churn_flag)          AS revenue_at_risk
FROM edw.vw_bi_churn_analytics
GROUP BY ROLLUP (region, city, zip_code);

-- 3. Khối CUBE Hợp đồng × Internet × Thanh toán (mọi tổ hợp Slice/Dice)
CREATE OR REPLACE VIEW edw.vw_dash_churn_contract_service_cube AS
SELECT
    COALESCE(contract_type, '∑ ALL')           AS contract_type,
    COALESCE(internet_service_type, '∑ ALL')   AS internet_service_type,
    COALESCE(payment_category, '∑ ALL')        AS payment_category,
    GROUPING(contract_type, internet_service_type, payment_category) AS grouping_id,
    COUNT(*)                                   AS customers,
    SUM(churn_flag)                            AS churned,
    ROUND(AVG(churn_flag::numeric), 4)         AS churn_rate,
    SUM(monthly_charges)                       AS mrr,
    SUM(monthly_charges * churn_flag)          AS revenue_at_risk
FROM edw.vw_bi_churn_analytics
GROUP BY CUBE (contract_type, internet_service_type, payment_category);

-- 4. Cohort thâm niên × Số add-on
CREATE OR REPLACE VIEW edw.vw_dash_tenure_addon AS
SELECT
    tenure_group,
    total_active_addons,
    COUNT(*)                                   AS customers,
    ROUND(AVG(churn_flag::numeric), 4)         AS churn_rate,
    ROUND(AVG(monthly_charges), 2)             AS avg_monthly_charges,
    ROUND(AVG(clv_proxy), 2)                   AS avg_clv
FROM edw.vw_bi_churn_analytics
GROUP BY tenure_group, total_active_addons;

-- 5. Index hỗ trợ lọc Slice/Dice trên bảng Dimension
CREATE INDEX IF NOT EXISTS idx_dim_loc_region_city ON edw.dim_location(region, city);
CREATE INDEX IF NOT EXISTS idx_dim_ctr_type_payment ON edw.dim_contract(contract_type, payment_category);
CREATE INDEX IF NOT EXISTS idx_dim_svc_internet ON edw.dim_service(internet_service_type);
