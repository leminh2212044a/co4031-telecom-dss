"""
CO4031 - TELECOM DSS | Phase 3
Module: RESTful Backend API (FastAPI) tích hợp Kho dữ liệu + Mô hình ML (app/api.py)

Chạy:   uvicorn app.api:app --reload --port 8000     (từ thư mục gốc dự án)
Docs:   http://localhost:8000/docs  (Swagger UI tự sinh)

Nhóm endpoint:
  /api/v1/meta ...           metadata chiều / phân cấp / độ đo
  /api/v1/olap/...           Endpoint 1 - truy vấn chỉ số tổng hợp từ Data Marts (Slice/Dice/Drill/Roll-up/Pivot)
  /api/v1/predict ...        Endpoint 2 - nạp model .pkl, trả về Churn % + gợi ý giữ chân
  /api/v1/whatif/...         What-If cấp khách hàng và cấp chính sách
  /api/v1/retention/...      Danh sách ưu tiên giữ chân (Model 3)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ml_service import json_safe
from .services import get_service

app = FastAPI(
    title="CO4031 Telecom DSS API",
    description="Backend tích hợp Data Warehouse (PostgreSQL Star Schema) và MBMS (Churn, K-Means, Priority).",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

Filters = Optional[Dict[str, List[str]]]


# ======================================================================
# SCHEMAS
# ======================================================================
class FilterRequest(BaseModel):
    filters: Filters = Field(default=None, examples=[{"contract_type": ["Month-to-month"]}])


class AggregateRequest(FilterRequest):
    group_by: List[str] = Field(default_factory=list, examples=[["region", "contract_type"]])


class DrillRequest(FilterRequest):
    hierarchy: str = Field(examples=["Địa lý (Geography)"])
    path: List[str] = Field(default_factory=list, examples=[["United States", "California"]])


class RollupRequest(FilterRequest):
    levels: List[str] = Field(examples=[["region", "city"]])


class PivotRequest(FilterRequest):
    rows: str = Field(examples=["contract_type"])
    columns: str = Field(examples=["internet_service_type"])
    measure: str = Field(default="churn_rate")


class CustomerProfile(BaseModel):
    """Hồ sơ khách hàng cho dự báo. Các trường suy diễn (total_active_addons, is_auto_payment,
    avg_charges_per_tenure, clv_proxy, tenure_group) được backend tự tính."""
    gender: str = "Female"
    is_senior_citizen: bool = False
    has_partner: bool = False
    has_dependents: bool = False
    region: str = "Bay Area"
    city: str = ""
    tenure_months: int = Field(default=2, ge=0, le=120)
    has_phone_service: bool = True
    multiple_lines: str = "No"
    internet_service_type: str = "Fiber optic"
    has_online_security: bool = False
    has_online_backup: bool = False
    has_device_protection: bool = False
    has_tech_support: bool = False
    has_streaming_tv: bool = False
    has_streaming_movies: bool = False
    contract_type: str = "Month-to-month"
    is_paperless_billing: bool = True
    payment_method: str = "Electronic check"
    monthly_charges: float = Field(default=85.5, ge=0)
    total_charges: Optional[float] = Field(default=None, ge=0)


class CompareRequest(BaseModel):
    baseline: CustomerProfile
    scenario: CustomerProfile


class PolicyScenario(BaseModel):
    contract_migration_pct: float = Field(default=0, ge=0, le=100)
    autopay_pct: float = Field(default=0, ge=0, le=100)
    security_bundle_pct: float = Field(default=0, ge=0, le=100)
    price_change_pct: float = Field(default=0, ge=-50, le=50)
    only_active: bool = True
    targeting: str = Field(default="risk", pattern="^(risk|random)$")


class PolicyRequest(FilterRequest):
    scenario: PolicyScenario = PolicyScenario()
    segment_by: str = "contract_type"


class RetentionRequest(FilterRequest):
    k: int = Field(default=100, ge=1, le=7043)
    risk_tiers: Optional[List[str]] = None
    only_active: bool = True


def _run(fn, *args, **kwargs):
    try:
        return json_safe(fn(*args, **kwargs))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ======================================================================
# HEALTH & METADATA
# ======================================================================
@app.get("/health", tags=["System"])
def health():
    svc = get_service()
    return {"status": "ok", "data_source": svc.dw.source_label, "model": svc.models.info}


@app.get("/api/v1/meta", tags=["Metadata"])
def metadata():
    return _run(get_service().metadata)


@app.get("/api/v1/meta/values/{column}", tags=["Metadata"])
def distinct_values(column: str):
    return _run(get_service().distinct_values, column)


# ======================================================================
# ENDPOINT 1: OLAP / DATA MARTS
# ======================================================================
@app.post("/api/v1/olap/kpis", tags=["OLAP"])
def kpis(req: FilterRequest):
    return _run(get_service().kpis, req.filters)


@app.post("/api/v1/olap/aggregate", tags=["OLAP"])
def aggregate(req: AggregateRequest):
    return _run(get_service().aggregate, req.group_by, req.filters)


@app.post("/api/v1/olap/drill", tags=["OLAP"])
def drill(req: DrillRequest):
    return _run(get_service().drill, req.hierarchy, req.path, req.filters)


@app.post("/api/v1/olap/rollup", tags=["OLAP"])
def rollup(req: RollupRequest):
    return _run(get_service().rollup, req.levels, req.filters)


@app.post("/api/v1/olap/pivot", tags=["OLAP"])
def pivot(req: PivotRequest):
    svc = get_service()
    try:
        table = svc.pivot(req.rows, req.columns, req.measure, req.filters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return json_safe({"index": list(table.index), "columns": list(table.columns),
                      "values": table.to_numpy().tolist()})


@app.post("/api/v1/olap/geo", tags=["OLAP"])
def geo(req: FilterRequest):
    return _run(get_service().geo_points, req.filters)


# ======================================================================
# ENDPOINT 2: ML PREDICTION & WHAT-IF
# ======================================================================
@app.post("/api/v1/predict", tags=["ML"])
def predict(profile: CustomerProfile):
    """Dự báo Churn %, risk tier, phân khúc K-Means, priority score và gợi ý hành động giữ chân."""
    return _run(get_service().score_customer, profile.model_dump())


@app.get("/api/v1/customers", tags=["ML"])
def customer_ids(limit: int = Query(300, ge=1, le=7043)):
    return _run(get_service().customer_ids, limit)


@app.get("/api/v1/customers/{customer_id}", tags=["ML"])
def customer(customer_id: str):
    rec = get_service().customer_record(customer_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy khách hàng")
    return json_safe(rec)


@app.post("/api/v1/whatif/compare", tags=["What-If"])
def whatif_compare(req: CompareRequest):
    return _run(get_service().compare_scenarios, req.baseline.model_dump(), req.scenario.model_dump())


@app.post("/api/v1/whatif/policy", tags=["What-If"])
def whatif_policy(req: PolicyRequest):
    return _run(get_service().simulate_policy, req.scenario.model_dump(), req.filters, req.segment_by)


# ======================================================================
# MODEL 2 & 3: PHÂN KHÚC & ƯU TIÊN GIỮ CHÂN
# ======================================================================
@app.post("/api/v1/segments/profile", tags=["Segmentation"])
def segment_profile(req: FilterRequest):
    return _run(get_service().segment_profile, req.filters)


@app.post("/api/v1/segments/points", tags=["Segmentation"])
def segment_points(req: FilterRequest):
    return _run(get_service().segment_points, req.filters)


@app.post("/api/v1/retention/top", tags=["Retention"])
def retention_top(req: RetentionRequest):
    return _run(get_service().retention_top, req.k, req.filters, req.risk_tiers, req.only_active)


@app.get("/api/v1/retention/metrics", tags=["Retention"])
def retention_metrics():
    df = get_service().priority_metrics()
    return json_safe(df) if df is not None else []
