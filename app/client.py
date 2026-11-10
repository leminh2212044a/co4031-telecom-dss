"""
CO4031 - TELECOM DSS | Phase 3
Module: Client cho Dashboard (app/client.py)

Dashboard không phụ thuộc vào việc backend chạy ở đâu:
- ApiClient   : gọi FastAPI qua HTTP (kiến trúc client-server, đúng mô hình DSS 3 thành phần)
- LocalClient : gọi trực tiếp DSSService trong cùng tiến trình (demo nhanh, không cần bật API)
Cả hai trả về cùng kiểu dữ liệu (dict / pandas.DataFrame).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import config

try:
    import requests
except ImportError:
    requests = None


class LocalClient:
    mode = "local"

    def __init__(self):
        from .services import get_service
        self.svc = get_service()

    @property
    def label(self):
        return f"In-process · {self.svc.dw.source_label}"

    def metadata(self): return self.svc.metadata()
    def distinct_values(self, col): return self.svc.distinct_values(col)
    def kpis(self, filters=None): return self.svc.kpis(filters)
    def aggregate(self, group_by, filters=None): return self.svc.aggregate(group_by, filters)
    def drill(self, hierarchy, path, filters=None): return self.svc.drill(hierarchy, path, filters)
    def rollup(self, levels, filters=None): return self.svc.rollup(levels, filters)
    def pivot(self, rows, columns, measure, filters=None): return self.svc.pivot(rows, columns, measure, filters)
    def geo_points(self, filters=None): return self.svc.geo_points(filters)
    def predict(self, profile): return self.svc.score_customer(profile)
    def compare(self, baseline, scenario): return self.svc.compare_scenarios(baseline, scenario)
    def policy(self, scenario, filters=None, segment_by="contract_type"):
        return self.svc.simulate_policy(scenario, filters, segment_by)
    def segment_profile(self, filters=None): return self.svc.segment_profile(filters)
    def segment_points(self, filters=None): return self.svc.segment_points(filters)
    def retention_top(self, k=100, filters=None, risk_tiers=None, only_active=True):
        return self.svc.retention_top(k, filters, risk_tiers, only_active)
    def retention_metrics(self): return self.svc.priority_metrics()
    def customer_ids(self, limit=300): return self.svc.customer_ids(limit)
    def customer(self, cid): return self.svc.customer_record(cid)


class ApiClient:
    mode = "api"

    def __init__(self, base_url: str = config.API_URL, timeout: float = 30):
        if requests is None:
            raise ImportError("Cần cài 'requests' để Dashboard gọi API.")
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self._health = self._get("/health")

    @property
    def label(self):
        return f"FastAPI {self.base} · {self._health.get('data_source', '')}"

    def _get(self, path, **params):
        r = requests.get(self.base + path, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path, payload):
        r = requests.post(self.base + path, json=payload, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"API {path} lỗi {r.status_code}: {r.text}")
        return r.json()

    @staticmethod
    def _df(rows) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def metadata(self): return self._get("/api/v1/meta")
    def distinct_values(self, col): return self._get(f"/api/v1/meta/values/{col}")
    def kpis(self, filters=None): return self._post("/api/v1/olap/kpis", {"filters": filters})
    def aggregate(self, group_by, filters=None):
        return self._df(self._post("/api/v1/olap/aggregate", {"group_by": list(group_by), "filters": filters}))
    def drill(self, hierarchy, path, filters=None):
        res = self._post("/api/v1/olap/drill", {"hierarchy": hierarchy, "path": list(path), "filters": filters})
        res["data"] = self._df(res["data"])
        return res
    def rollup(self, levels, filters=None):
        return self._df(self._post("/api/v1/olap/rollup", {"levels": list(levels), "filters": filters}))
    def pivot(self, rows, columns, measure, filters=None):
        res = self._post("/api/v1/olap/pivot",
                         {"rows": rows, "columns": columns, "measure": measure, "filters": filters})
        return pd.DataFrame(res["values"], index=res["index"], columns=res["columns"])
    def geo_points(self, filters=None): return self._df(self._post("/api/v1/olap/geo", {"filters": filters}))
    def predict(self, profile): return self._post("/api/v1/predict", profile)
    def compare(self, baseline, scenario):
        return self._post("/api/v1/whatif/compare", {"baseline": baseline, "scenario": scenario})
    def policy(self, scenario, filters=None, segment_by="contract_type"):
        res = self._post("/api/v1/whatif/policy",
                         {"scenario": scenario, "filters": filters, "segment_by": segment_by})
        res["breakdown"] = self._df(res["breakdown"])
        return res
    def segment_profile(self, filters=None): return self._df(self._post("/api/v1/segments/profile", {"filters": filters}))
    def segment_points(self, filters=None): return self._df(self._post("/api/v1/segments/points", {"filters": filters}))
    def retention_top(self, k=100, filters=None, risk_tiers=None, only_active=True):
        return self._df(self._post("/api/v1/retention/top", {
            "k": int(k), "filters": filters, "risk_tiers": risk_tiers, "only_active": only_active}))
    def retention_metrics(self):
        rows = self._get("/api/v1/retention/metrics")
        return self._df(rows) if rows else None
    def customer_ids(self, limit=300): return self._get("/api/v1/customers", limit=limit)
    def customer(self, cid):
        try:
            return self._get(f"/api/v1/customers/{cid}")
        except Exception:  # noqa: BLE001
            return None


def make_client(mode: Optional[str] = None):
    """auto: dùng API nếu đang chạy, nếu không thì gọi trực tiếp service."""
    mode = (mode or config.DASHBOARD_BACKEND).lower()
    if mode in ("api", "auto"):
        try:
            return ApiClient()
        except Exception:  # noqa: BLE001
            if mode == "api":
                raise
    return LocalClient()
