"""
CO4031 - TELECOM DSS | Phase 3
Module: Cấu hình chung cho Backend API & Dashboard (app/config.py)

Biến môi trường (đọc từ .env nếu có):
- DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD : kết nối PostgreSQL telecom_dw
- DSS_DATA_SOURCE : auto | postgres | csv   (auto = thử PostgreSQL, lỗi thì dùng CSV Data Marts)
- DSS_API_URL     : địa chỉ FastAPI cho Dashboard (mặc định http://localhost:8000)
- DSS_BACKEND     : auto | api | local      (Dashboard gọi qua API hay gọi trực tiếp service)
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv không bắt buộc
    pass

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

# --- Data Marts xuất từ DW (fallback khi không kết nối được PostgreSQL) ---
DATA_DIR = ROOT_DIR / "data" / "processed"
BI_CSV_PATH = DATA_DIR / "telecom_bi_churn_analytics.csv"
ML_CSV_PATH = DATA_DIR / "telecom_ml_feature_store.csv"

# --- Artifacts của MBMS (Phase 2) ---
MODEL_DIR = ROOT_DIR / "models" / "saved"
EVAL_DIR = ROOT_DIR / "models" / "evaluation"

# --- SQL Views của DW ---
BI_VIEW = "edw.vw_bi_churn_analytics"
ML_VIEW = "edw.vw_ml_feature_store"
KPI_VIEW = "edw.vw_dash_kpi_summary"
SCORES_TABLE = "edw.ml_customer_scores"

# --- Đầu ra cho Power BI (nằm trong app/) ---
POWERBI_DIR = APP_DIR / "powerbi"
POWERBI_DATA_DIR = POWERBI_DIR / "data"

DATA_SOURCE = os.getenv("DSS_DATA_SOURCE", "auto").lower()
API_URL = os.getenv("DSS_API_URL", "http://localhost:8000").rstrip("/")
DASHBOARD_BACKEND = os.getenv("DSS_BACKEND", "auto").lower()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "telecom_dw"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}
