from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config

try:
    import psycopg2
except ImportError:  # cho phép chạy chế độ CSV khi chưa cài psycopg2
    psycopg2 = None


# ======================================================================
# METADATA OLAP: CHIỀU, PHÂN CẤP, ĐỘ ĐO (whitelist chống SQL injection)
# ======================================================================
DIMENSIONS: Dict[str, str] = {
    "country": "Quốc gia",
    "state": "Bang",
    "region": "Vùng thị trường",
    "city": "Thành phố",
    "zip_code": "Mã bưu chính",
    "contract_type": "Loại hợp đồng",
    "payment_category": "Nhóm thanh toán",
    "payment_method": "Phương thức thanh toán",
    "is_auto_payment": "Thanh toán tự động",
    "is_paperless_billing": "Hóa đơn điện tử",
    "internet_service_type": "Loại Internet",
    "multiple_lines": "Nhiều đường thoại",
    "has_phone_service": "Dịch vụ thoại",
    "total_active_addons": "Số add-on",
    "has_online_security": "Online Security",
    "has_online_backup": "Online Backup",
    "has_device_protection": "Device Protection",
    "has_tech_support": "Tech Support",
    "has_streaming_tv": "Streaming TV",
    "has_streaming_movies": "Streaming Movies",
    "tenure_group": "Nhóm thâm niên",
    "gender": "Giới tính",
    "is_senior_citizen": "Người cao tuổi",
    "has_partner": "Có bạn đời",
    "has_dependents": "Có người phụ thuộc",
    "churn_label": "Trạng thái churn",
}

BOOL_DIMENSIONS = {
    "is_auto_payment", "is_paperless_billing", "has_phone_service",
    "has_online_security", "has_online_backup", "has_device_protection",
    "has_tech_support", "has_streaming_tv", "has_streaming_movies",
    "is_senior_citizen", "has_partner", "has_dependents",
}
INT_DIMENSIONS = {"total_active_addons"}

HIERARCHIES: Dict[str, List[str]] = {
    "Địa lý (Geography)": ["country", "state", "region", "city", "zip_code"],
    "Hợp đồng & Thanh toán": ["contract_type", "payment_category", "payment_method"],
    "Dịch vụ (Service)": ["internet_service_type", "total_active_addons", "has_tech_support"],
    "Khách hàng (Customer)": ["tenure_group", "is_senior_citizen", "has_dependents"],
}

# Độ đo: tên -> (biểu thức SQL, nhãn hiển thị)
MEASURES: Dict[str, tuple] = {
    "customers": ("COUNT(*)", "Số khách hàng"),
    "churned": ("SUM(churn_flag)", "Số khách rời mạng"),
    "churn_rate": ("AVG(churn_flag::numeric)", "Tỷ lệ churn"),
    "mrr": ("SUM(monthly_charges)", "Doanh thu tháng (MRR)"),
    "revenue_at_risk": ("SUM(monthly_charges * churn_flag)", "MRR mất do churn"),
    "avg_monthly_charges": ("AVG(monthly_charges)", "Cước TB/tháng"),
    "total_revenue": ("SUM(total_charges)", "Tổng doanh thu tích lũy"),
    "avg_tenure": ("AVG(tenure_months)", "Thâm niên TB (tháng)"),
    "avg_clv": ("AVG(clv_proxy)", "CLV proxy TB"),
}

TENURE_ORDER = ["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]
CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]


def _to_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "t"}


def normalize_filters(filters: Optional[Dict[str, Sequence]]) -> Dict[str, list]:
    """Kiểm tra whitelist & chuẩn hóa kiểu dữ liệu giá trị lọc."""
    clean: Dict[str, list] = {}
    for col, values in (filters or {}).items():
        if col not in DIMENSIONS:
            raise ValueError(f"Chiều không hợp lệ: {col}")
        if values is None:
            continue
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        values = list(values)
        if not values:
            continue
        if col in BOOL_DIMENSIONS:
            clean[col] = sorted({_to_bool(v) for v in values})
        elif col in INT_DIMENSIONS:
            clean[col] = sorted({int(v) for v in values})
        else:
            clean[col] = sorted({str(v) for v in values})
    return clean


def _validate_dims(dims: Sequence[str]) -> List[str]:
    dims = list(dims or [])
    for d in dims:
        if d not in DIMENSIONS:
            raise ValueError(f"Chiều không hợp lệ: {d}")
    return dims


# ======================================================================
# DATA WAREHOUSE ACCESS LAYER
# ======================================================================
class DataWarehouse:
    """Truy cập DW qua SQL Views (PostgreSQL) hoặc Data Marts CSV."""

    def __init__(self, source: str = config.DATA_SOURCE):
        self.source = self._resolve_source(source)
        self._bi_cache: Optional[pd.DataFrame] = None
        self._ml_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Kết nối
    # ------------------------------------------------------------------
    def _connect(self):
        return psycopg2.connect(connect_timeout=3, **config.DB_CONFIG)

    def _resolve_source(self, source: str) -> str:
        if source == "csv":
            return "csv"
        if psycopg2 is not None:
            try:
                conn = self._connect()
                cur = conn.cursor()
                cur.execute(f"SELECT 1 FROM {config.BI_VIEW} LIMIT 1;")
                conn.close()
                return "postgres"
            except Exception as exc:  # noqa: BLE001
                if source == "postgres":
                    raise ConnectionError(f"Không kết nối được PostgreSQL: {exc}") from exc
        elif source == "postgres":
            raise ImportError("Cần cài psycopg2-binary để dùng chế độ postgres.")
        if not config.BI_CSV_PATH.exists():
            raise FileNotFoundError(
                "Không kết nối được PostgreSQL và không tìm thấy Data Mart CSV tại "
                f"{config.BI_CSV_PATH}. Hãy chạy etl/etl_pipeline.py và scripts/export_data_marts.py."
            )
        return "csv"

    def _query(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        conn = self._connect()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # pandas cảnh báo khi dùng DBAPI thuần
                return pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()

    @property
    def source_label(self) -> str:
        if self.source == "postgres":
            return f"PostgreSQL · {config.DB_CONFIG['dbname']} ({config.BI_VIEW})"
        return f"CSV Data Mart · {config.BI_CSV_PATH.name}"

    # ------------------------------------------------------------------
    # Đọc Data Marts
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_mart(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "zip_code" in df.columns:
            df["zip_code"] = df["zip_code"].astype(str)
        for col in BOOL_DIMENSIONS & set(df.columns):
            df[col] = df[col].map(_to_bool)
        for col in ["monthly_charges", "total_charges", "estimated_annual_charges",
                    "avg_charges_per_tenure", "clv_proxy", "latitude", "longitude"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        for col in ["tenure_months", "churn_flag", "total_active_addons"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df

    def bi_mart(self) -> pd.DataFrame:
        """Data Mart BI đầy đủ (7,043 dòng) - dùng cho bản đồ, bảng chi tiết."""
        if self._bi_cache is None:
            if self.source == "postgres":
                raw = self._query(f"SELECT * FROM {config.BI_VIEW};")
            else:
                raw = pd.read_csv(config.BI_CSV_PATH)
            self._bi_cache = self._clean_mart(raw)
        return self._bi_cache

    def feature_store(self) -> pd.DataFrame:
        """Feature Store ML - đầu vào chấm điểm toàn bộ khách hàng bằng model."""
        if self._ml_cache is None:
            if self.source == "postgres":
                raw = self._query(f"SELECT * FROM {config.ML_VIEW};")
            else:
                raw = pd.read_csv(config.ML_CSV_PATH)
            self._ml_cache = self._clean_mart(raw)
        return self._ml_cache

    def filtered_mart(self, filters: Optional[Dict[str, Sequence]] = None) -> pd.DataFrame:
        df = self.bi_mart()
        for col, values in normalize_filters(filters).items():
            df = df[df[col].isin(values)]
        return df

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def distinct_values(self, column: str) -> list:
        _validate_dims([column])
        if self.source == "postgres":
            df = self._query(
                f"SELECT DISTINCT {column} AS v FROM {config.BI_VIEW} ORDER BY 1;"
            )
            values = df["v"].tolist()
        else:
            values = sorted(self.bi_mart()[column].dropna().unique().tolist())
        if column == "zip_code":
            values = [str(v) for v in values]
        if column == "tenure_group":
            values = [v for v in TENURE_ORDER if v in values] + [v for v in values if v not in TENURE_ORDER]
        if column == "contract_type":
            values = [v for v in CONTRACT_ORDER if v in values] + [v for v in values if v not in CONTRACT_ORDER]
        return [v.item() if hasattr(v, "item") else v for v in values]

    def metadata(self) -> dict:
        return {
            "source": self.source,
            "source_label": self.source_label,
            "dimensions": DIMENSIONS,
            "hierarchies": HIERARCHIES,
            "measures": {k: v[1] for k, v in MEASURES.items()},
        }

    # ------------------------------------------------------------------
    # SQL builder (chế độ postgres)
    # ------------------------------------------------------------------
    @staticmethod
    def _where_clause(filters: Dict[str, list]):
        clauses, params = [], []
        for col, values in filters.items():
            if col in BOOL_DIMENSIONS:
                clauses.append(f"{col} = ANY(%s::boolean[])")
            elif col in INT_DIMENSIONS:
                clauses.append(f"{col} = ANY(%s::int[])")
            else:
                clauses.append(f"{col}::text = ANY(%s::text[])")
            params.append(values)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    @staticmethod
    def _measure_sql() -> str:
        return ",\n       ".join(f"{expr} AS {name}" for name, (expr, _) in MEASURES.items())

    # ------------------------------------------------------------------
    # Pandas aggregator (chế độ csv)
    # ------------------------------------------------------------------
    @staticmethod
    def _pandas_measures(g) -> pd.DataFrame:
        out = pd.DataFrame({
            "customers": g.size(),
            "churned": g["churn_flag"].sum(),
            "churn_rate": g["churn_flag"].mean(),
            "mrr": g["monthly_charges"].sum(),
            "revenue_at_risk": g["_rev_risk"].sum(),
            "avg_monthly_charges": g["monthly_charges"].mean(),
            "total_revenue": g["total_charges"].sum(),
            "avg_tenure": g["tenure_months"].mean(),
            "avg_clv": g["clv_proxy"].mean(),
        })
        return out

    @staticmethod
    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        for col in MEASURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        for col in ["customers", "churned"]:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        return df

    # ------------------------------------------------------------------
    # OLAP: Aggregate (Slice / Dice / Drill)
    # ------------------------------------------------------------------
    def aggregate(self, group_by: Sequence[str] = (),
                  filters: Optional[Dict[str, Sequence]] = None) -> pd.DataFrame:
        """Gộp các độ đo theo danh sách chiều, sau khi áp dụng Slice/Dice."""
        group_by = _validate_dims(group_by)
        clean = normalize_filters(filters)

        if self.source == "postgres":
            where, params = self._where_clause(clean)
            select_dims = "".join(f"{d}, " for d in group_by)
            group_sql = ("GROUP BY " + ", ".join(group_by) + " ORDER BY " + ", ".join(group_by)) if group_by else ""
            sql = f"""
                SELECT {select_dims}{self._measure_sql()}
                FROM {config.BI_VIEW}
                {where}
                {group_sql};
            """
            df = self._query(sql, params)
            if "zip_code" in df.columns:
                df["zip_code"] = df["zip_code"].astype(str)
            return self._finalize(df)

        mart = self.filtered_mart(clean).copy()
        mart["_rev_risk"] = mart["monthly_charges"] * mart["churn_flag"]
        if group_by:
            df = self._pandas_measures(mart.groupby(group_by, dropna=False)).reset_index()
        else:
            df = self._pandas_measures(mart.assign(_all=1).groupby("_all")).reset_index(drop=True)
            if mart.empty:
                df = pd.DataFrame([{m: 0 for m in MEASURES}])
        return self._finalize(df)

    def kpis(self, filters: Optional[Dict[str, Sequence]] = None) -> dict:
        row = self.aggregate([], filters)
        if row.empty:
            return {m: 0.0 for m in MEASURES}
        return {k: (float(v) if pd.notna(v) else 0.0) for k, v in row.iloc[0].to_dict().items()}

    # ------------------------------------------------------------------
    # OLAP: Drill-down theo phân cấp
    # ------------------------------------------------------------------
    def drill(self, hierarchy: str, path: Sequence[str] = (),
              filters: Optional[Dict[str, Sequence]] = None) -> dict:
        """
        path = các giá trị đã chọn từ cấp cao xuống. Trả về các phần tử con ở cấp kế tiếp.
        Ví dụ: hierarchy="Địa lý (Geography)", path=["United States", "California", "Bay Area"]
               -> gộp theo city trong Bay Area.
        """
        levels = HIERARCHIES.get(hierarchy)
        if levels is None:
            raise ValueError(f"Phân cấp không tồn tại: {hierarchy}")
        path = list(path or [])[: len(levels) - 1]
        drill_filters = dict(filters or {})
        for level, value in zip(levels, path):
            drill_filters[level] = [value]
        level = levels[len(path)]
        data = self.aggregate([level], drill_filters)
        return {
            "hierarchy": hierarchy,
            "levels": levels,
            "path": path,
            "level": level,
            "depth": len(path),
            "can_drill_down": len(path) + 1 < len(levels),
            "data": data,
        }

    # ------------------------------------------------------------------
    # OLAP: Roll-up có dòng tổng con (GROUP BY ROLLUP)
    # ------------------------------------------------------------------
    def rollup(self, levels: Sequence[str],
               filters: Optional[Dict[str, Sequence]] = None) -> pd.DataFrame:
        levels = _validate_dims(levels)
        if not levels:
            return self.aggregate([], filters)
        clean = normalize_filters(filters)

        if self.source == "postgres":
            where, params = self._where_clause(clean)
            dims_sql = ", ".join(levels)
            sql = f"""
                SELECT {dims_sql}, GROUPING({dims_sql}) AS grouping_id,
                       {self._measure_sql()}
                FROM {config.BI_VIEW}
                {where}
                GROUP BY ROLLUP({dims_sql})
                ORDER BY {", ".join(f"{d} NULLS LAST" for d in levels)};
            """
            df = self._query(sql, params)
            # grouping_id: bit=1 nghĩa là cột đó được gộp -> cấp = số cột không gộp
            n = len(levels)
            df["rollup_level"] = df["grouping_id"].apply(
                lambda g: n - bin(int(g)).count("1"))
            df = df.drop(columns=["grouping_id"])
        else:
            mart = self.filtered_mart(clean).copy()
            mart["_rev_risk"] = mart["monthly_charges"] * mart["churn_flag"]
            parts = []
            for depth in range(len(levels), -1, -1):
                keys = levels[:depth]
                if keys:
                    part = self._pandas_measures(mart.groupby(keys, dropna=False)).reset_index()
                else:
                    part = self._pandas_measures(mart.assign(_all=1).groupby("_all")).reset_index(drop=True)
                for missing in levels[depth:]:
                    part[missing] = None
                part["rollup_level"] = depth
                parts.append(part)
            df = pd.concat(parts, ignore_index=True)
            df = df.sort_values(levels, na_position="last", kind="stable")

        for col in levels:
            df[col] = df[col].astype(object).where(df[col].notna(), "∑ Tổng")
            df[col] = df[col].astype(str)
        return self._finalize(df[levels + ["rollup_level"] + list(MEASURES)].reset_index(drop=True))

    # ------------------------------------------------------------------
    # OLAP: Pivot (bảng chéo)
    # ------------------------------------------------------------------
    def pivot(self, rows: str, columns: str, measure: str = "churn_rate",
              filters: Optional[Dict[str, Sequence]] = None) -> pd.DataFrame:
        if measure not in MEASURES:
            raise ValueError(f"Độ đo không hợp lệ: {measure}")
        if rows == columns:
            raise ValueError("Chiều dòng và chiều cột phải khác nhau.")
        data = self.aggregate([rows, columns], filters)
        table = data.pivot(index=rows, columns=columns, values=measure)
        table.index = table.index.astype(str)
        table.columns = table.columns.astype(str)
        return table

    # ------------------------------------------------------------------
    # Tiện ích
    # ------------------------------------------------------------------
    def kpi_view(self) -> Optional[pd.DataFrame]:
        """Đọc view KPI tổng hợp sẵn (app/sql/04_dashboard_views.sql) nếu có."""
        if self.source != "postgres":
            return None
        try:
            return self._query(f"SELECT * FROM {config.KPI_VIEW};")
        except Exception:  # noqa: BLE001 - view chưa được tạo
            return None

    def customer_record(self, customer_id: str) -> Optional[dict]:
        df = self.feature_store()
        hit = df[df["customer_id"].astype(str) == str(customer_id)]
        if hit.empty:
            return None
        return hit.iloc[0].to_dict()
