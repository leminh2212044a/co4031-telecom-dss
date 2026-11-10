-- ======================================================================
-- CO4031 - TELECOM DATA WAREHOUSE & DECISION SUPPORT SYSTEM (DSS)
-- Script: app/sql/05_ml_customer_scores.sql
-- Description: Bảng lưu kết quả chấm điểm của MBMS ghi ngược vào DW,
--              để Power BI (và các công cụ BI khác) đọc như một bảng bình thường.
-- Nạp dữ liệu: python -m app.export_scores --to-db   (script tự chạy DDL này)
-- ======================================================================

CREATE TABLE IF NOT EXISTS edw.ml_customer_scores (
    customer_id         VARCHAR(20)  PRIMARY KEY,
    churn_probability   NUMERIC(6,4) NOT NULL,
    churn_prediction    SMALLINT     NOT NULL,
    risk_tier           VARCHAR(10)  NOT NULL,
    cluster_id          SMALLINT,
    cluster_name        VARCHAR(60),
    priority_score      NUMERIC(6,2) NOT NULL,
    priority_rank       INTEGER      NOT NULL,
    expected_mrr_at_risk NUMERIC(10,2) NOT NULL,
    model_name          VARCHAR(40)  NOT NULL,
    scored_at           TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ml_scores_risk ON edw.ml_customer_scores(risk_tier);
CREATE INDEX IF NOT EXISTS idx_ml_scores_cluster ON edw.ml_customer_scores(cluster_id);

-- View gộp Data Mart BI + điểm ML: Power BI chỉ cần nạp 1 bảng này
CREATE OR REPLACE VIEW edw.vw_bi_churn_scored AS
SELECT b.*,
       s.churn_probability,
       s.churn_prediction,
       s.risk_tier,
       s.cluster_id,
       s.cluster_name,
       s.priority_score,
       s.priority_rank,
       s.expected_mrr_at_risk
FROM edw.vw_bi_churn_analytics b
LEFT JOIN edw.ml_customer_scores s ON s.customer_id = b.customer_id;
