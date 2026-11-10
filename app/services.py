"""
CO4031 - TELECOM DSS | Phase 3
Module: Facade tích hợp DBMS (Data Warehouse) + MBMS (Models) (app/services.py)

"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Sequence

import pandas as pd

from . import config
from .data_access import DataWarehouse, normalize_filters
from .ml_service import ModelService


class DSSService:
    def __init__(self, source: str = config.DATA_SOURCE):
        self.dw = DataWarehouse(source)
        self.models = ModelService()
        self.models.fit_population_stats(self.dw.feature_store())
        self._scored: Optional[pd.DataFrame] = None

    # ---------------- DBMS ----------------
    def metadata(self) -> dict:
        meta = self.dw.metadata()
        meta["model"] = self.models.info
        return meta

    def distinct_values(self, column: str) -> list:
        return self.dw.distinct_values(column)

    def kpis(self, filters=None) -> dict:
        return self.dw.kpis(filters)

    def aggregate(self, group_by: Sequence[str], filters=None) -> pd.DataFrame:
        return self.dw.aggregate(group_by, filters)

    def drill(self, hierarchy: str, path: Sequence[str], filters=None) -> dict:
        return self.dw.drill(hierarchy, path, filters)

    def rollup(self, levels: Sequence[str], filters=None) -> pd.DataFrame:
        return self.dw.rollup(levels, filters)

    def pivot(self, rows: str, columns: str, measure: str, filters=None) -> pd.DataFrame:
        return self.dw.pivot(rows, columns, measure, filters)

    def geo_points(self, filters=None) -> pd.DataFrame:
        cols = ["customer_id", "region", "city", "zip_code", "latitude", "longitude",
                "monthly_charges", "churn_flag", "churn_label", "contract_type"]
        return self.dw.filtered_mart(filters)[cols]

    # ---------------- MBMS ----------------
    def scored_population(self) -> pd.DataFrame:
        """Feature Store + churn probability + cụm + priority (cache vì 7k dòng)."""
        if self._scored is None:
            self._scored = self.models.score_population(self.dw.feature_store())
        return self._scored

    def _apply_filters_to_scored(self, filters) -> pd.DataFrame:
        df = self.scored_population()
        clean = normalize_filters(filters)
        if clean:
            # Feature store không có mọi chiều BI -> join theo customer_id với BI mart
            ids = set(self.dw.filtered_mart(clean)["customer_id"].astype(str))
            df = df[df["customer_id"].astype(str).isin(ids)]
        return df

    def retention_top(self, k: int = 100, filters=None, risk_tiers=None,
                      only_active: bool = True) -> pd.DataFrame:
        df = self._apply_filters_to_scored(filters)
        if only_active:
            df = df[df["churn_flag"] == 0]
        if risk_tiers:
            df = df[df["risk_tier"].isin(risk_tiers)]
        cols = ["priority_rank", "customer_id", "priority_score", "churn_probability", "risk_tier",
                "cluster_name", "contract_type", "internet_service_type", "payment_method",
                "tenure_months", "monthly_charges", "total_active_addons", "region", "city", "churn_flag"]
        cols = [c for c in cols if c in df.columns]
        return df.sort_values("priority_score", ascending=False)[cols].head(int(k)).reset_index(drop=True)

    def segment_profile(self, filters=None) -> pd.DataFrame:
        df = self._apply_filters_to_scored(filters)
        if "cluster_id" not in df.columns:
            return pd.DataFrame()
        prof = df.groupby(["cluster_id", "cluster_name"]).agg(
            customers=("customer_id", "size"),
            avg_tenure=("tenure_months", "mean"),
            avg_monthly_charges=("monthly_charges", "mean"),
            avg_total_charges=("total_charges", "mean"),
            avg_addons=("total_active_addons", "mean"),
            actual_churn_rate=("churn_flag", "mean"),
            avg_churn_probability=("churn_probability", "mean"),
            mrr=("monthly_charges", "sum"),
        ).reset_index()
        prof["share"] = prof["customers"] / max(prof["customers"].sum(), 1)
        return prof

    def segment_points(self, filters=None, sample: int = 2500) -> pd.DataFrame:
        df = self._apply_filters_to_scored(filters)
        cols = ["customer_id", "tenure_months", "monthly_charges", "total_active_addons",
                "cluster_name", "churn_probability", "churn_flag"]
        cols = [c for c in cols if c in df.columns]
        if len(df) > sample:
            df = df.sample(sample, random_state=42)
        return df[cols]

    def score_customer(self, profile: dict) -> dict:
        return self.models.score_customer(profile)

    def compare_scenarios(self, baseline: dict, scenario: dict) -> dict:
        return self.models.compare_scenarios(baseline, scenario)

    def customer_record(self, customer_id: str) -> Optional[dict]:
        return self.dw.customer_record(customer_id)

    def customer_ids(self, limit: int = 300) -> list:
        """Khách hàng mẫu cho Form What-If (ưu tiên khách đang hoạt động, rủi ro cao)."""
        df = self.scored_population()
        df = df[df["churn_flag"] == 0].sort_values("churn_probability", ascending=False)
        return df["customer_id"].astype(str).head(limit).tolist()

    def simulate_policy(self, scenario: dict, filters=None, segment_by: str = "contract_type") -> dict:
        ids = None
        if normalize_filters(filters):
            ids = set(self.dw.filtered_mart(filters)["customer_id"].astype(str))
        fs = self.dw.feature_store()
        if ids is not None:
            fs = fs[fs["customer_id"].astype(str).isin(ids)]
        return self.models.simulate_policy(fs, scenario, segment_by=segment_by)

    def priority_metrics(self) -> Optional[pd.DataFrame]:
        path = config.EVAL_DIR / "churn_priority_metrics.csv"
        return pd.read_csv(path) if path.exists() else None


@lru_cache(maxsize=1)
def get_service() -> DSSService:
    return DSSService()
