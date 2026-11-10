"""
CO4031 - TELECOM DSS | Phase 3
Module: Dịch vụ Mô hình (MBMS Service) cho API & Dashboard (app/ml_service.py)

"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from . import config


def _sklearn_pickle_compat():
    """
    Các file .pkl trong models/saved/ được lưu bằng scikit-learn 1.6.1. Từ bản 1.7, lớp nội bộ
    `_RemainderColsList` của ColumnTransformer bị xóa nên preprocessor.pkl không unpickle được.
    Shim này khai báo lại lớp đó để nạp được trên bản mới. Cách chắc chắn nhất vẫn là cài
    đúng scikit-learn==1.6.1 (xem app/requirements.txt).
    """
    import warnings
    try:
        from sklearn.exceptions import InconsistentVersionWarning
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    except ImportError:
        pass
    try:
        from collections import UserList
        from sklearn.compose import _column_transformer as ct
        if not hasattr(ct, "_RemainderColsList"):
            class _RemainderColsList(UserList):
                def __init__(self, data=None, *, future_dtype=None,
                             warning_was_emitted=False, warning_enabled=True):
                    super().__init__(data or [])
                    self.future_dtype = future_dtype
                    self.warning_was_emitted = warning_was_emitted
                    self.warning_enabled = warning_enabled
            ct._RemainderColsList = _RemainderColsList
    except Exception:  # noqa: BLE001
        pass


_sklearn_pickle_compat()

# ======================================================================
# HẰNG SỐ NGHIỆP VỤ (khớp với Phase 1 ETL và Phase 2 ML)
# ======================================================================
ADDON_COLS = [
    "has_online_security", "has_online_backup", "has_device_protection",
    "has_tech_support", "has_streaming_tv", "has_streaming_movies",
]
BOOL_FEATURES = [
    "is_senior_citizen", "has_partner", "has_dependents", "has_phone_service",
    *ADDON_COLS, "is_paperless_billing", "is_auto_payment",
]
CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]
INTERNET_TYPES = ["DSL", "Fiber optic", "No"]
MULTIPLE_LINES = ["No", "Yes", "No phone service"]
DEFAULT_KMEANS_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges",
    "avg_charges_per_tenure", "clv_proxy", "total_active_addons",
    "is_senior_citizen", "has_partner", "has_dependents",
    "is_paperless_billing", "is_auto_payment",
]
# Tên cụm theo hồ sơ trong docs/model_02_customer_segmentation.md (k=4, random_state=42)
CLUSTER_NAMES = {
    0: "Khách hàng Mới - Cước cơ bản",
    1: "Khách hàng Rủi ro cao",
    2: "Khách hàng Tiết kiệm",
    3: "Khách hàng Cao cấp - Trung thành",
}
CLUSTER_STRATEGY = {
    0: "Upsell dùng thử add-on (Security, Backup) giá ưu đãi để tăng ARPU.",
    1: "Can thiệp gấp: gọi chăm sóc, voucher chuyển hợp đồng 1-2 năm.",
    2: "Gói combo tiết kiệm, khuyến khích thanh toán tự động.",
    3: "Chương trình tri ân VIP, hỗ trợ kỹ thuật ưu tiên 24/7.",
}
DEFAULT_PRIORITY_WEIGHTS = {
    "churn_risk": 0.80, "monthly_value": 0.12,
    "contract_opportunity": 0.04, "service_opportunity": 0.04,
}
DEFAULT_CONTRACT_OPPORTUNITY = {"Month-to-month": 1.0, "One year": 0.6, "Two year": 0.3}
RISK_BINS = [-np.inf, 0.30, 0.50, 0.70, np.inf]
RISK_LABELS = ["Low", "Medium", "High", "Critical"]
RISK_LABELS_VI = {"Low": "Thấp", "Medium": "Trung bình", "High": "Cao", "Critical": "Rất cao"}


def risk_tier(prob: float) -> str:
    for upper, label in zip(RISK_BINS[1:], RISK_LABELS):
        if prob <= upper:
            return label
    return "Critical"


def tenure_group(t: int) -> str:
    if t <= 12:
        return "0-12 Months"
    if t <= 24:
        return "13-24 Months"
    if t <= 48:
        return "25-48 Months"
    return "49-72 Months"


def _as_bool(v) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in {"true", "1", "yes", "t"}


# ======================================================================
# CHUẨN HÓA HỒ SƠ KHÁCH HÀNG (giống quy tắc Transform trong ETL)
# ======================================================================
def build_customer_profile(inp: dict) -> dict:
    """
    Nhận input thô (từ Form / API) và suy ra đầy đủ thuộc tính như trong Feature Store:
    - Ràng buộc nghiệp vụ: không có Internet -> mọi add-on = False;
      không có thoại -> multiple_lines = 'No phone service'.
    - total_active_addons, is_auto_payment, tenure_group, avg_charges_per_tenure, clv_proxy.
    - total_charges mặc định = tenure * monthly nếu không nhập.
    """
    p = dict(inp)
    p["gender"] = str(p.get("gender", "Female"))
    p["region"] = str(p.get("region", "Bay Area"))
    p["city"] = str(p.get("city", ""))
    p["customer_id"] = str(p.get("customer_id", "WHAT-IF"))
    p["contract_type"] = str(p.get("contract_type", "Month-to-month"))
    p["payment_method"] = str(p.get("payment_method", "Electronic check"))
    p["internet_service_type"] = str(p.get("internet_service_type", "Fiber optic"))

    for col in BOOL_FEATURES:
        p[col] = _as_bool(p.get(col, False))

    if p["internet_service_type"] == "No":
        for col in ADDON_COLS:
            p[col] = False
    if not p["has_phone_service"]:
        p["multiple_lines"] = "No phone service"
    else:
        ml = str(p.get("multiple_lines", "No"))
        p["multiple_lines"] = "No" if ml == "No phone service" else ml

    p["is_auto_payment"] = "automatic" in p["payment_method"].lower()
    p["total_active_addons"] = int(sum(p[c] for c in ADDON_COLS))

    tenure = max(int(float(p.get("tenure_months", 0) or 0)), 0)
    monthly = max(float(p.get("monthly_charges", 0) or 0), 0.0)
    total = p.get("total_charges")
    total = float(total) if total not in (None, "") else tenure * monthly
    p["tenure_months"] = tenure
    p["monthly_charges"] = round(monthly, 2)
    p["total_charges"] = round(max(total, 0.0), 2)
    p["avg_charges_per_tenure"] = round(total / tenure, 2) if tenure > 0 else round(monthly, 2)
    p["clv_proxy"] = round(tenure * monthly, 2)
    p["tenure_group"] = tenure_group(tenure)
    return p


# ======================================================================
# MODEL SERVICE
# ======================================================================
class ModelService:
    def __init__(self, model_dir=config.MODEL_DIR):
        self.model_dir = model_dir
        self.churn_model = self._load("churn_model.pkl", required=True)
        self.preprocessor = self._load("preprocessor.pkl", required=True)
        self.threshold = float(self._load("churn_threshold.pkl", default=0.5))
        self.kmeans = self._load("kmeans_model.pkl")
        self.kmeans_scaler = self._load("kmeans_scaler.pkl")
        self.kmeans_features = list(self._load("kmeans_features.pkl", default=DEFAULT_KMEANS_FEATURES))
        self.priority_weights, self.contract_opportunity = self._load_priority_config()
        # Thống kê quần thể để chuẩn hóa priority score cho 1 khách hàng lẻ
        self.monthly_min, self.monthly_max, self.max_addons = 18.25, 118.75, 6.0

    # ------------------------------------------------------------------
    def _load(self, name, required=False, default=None):
        path = self.model_dir / name
        if path.exists():
            return joblib.load(path)
        if required:
            raise FileNotFoundError(
                f"Thiếu artifact {path}. Hãy chạy models/churn_classification.py trước.")
        return default

    def _load_priority_config(self):
        path = self.model_dir / "churn_priority_config.json"
        weights, contract = dict(DEFAULT_PRIORITY_WEIGHTS), dict(DEFAULT_CONTRACT_OPPORTUNITY)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            weights.update(cfg.get("weights", {}))
            contract.update(cfg.get("contract_opportunity", {}))
        return weights, contract

    def fit_population_stats(self, feature_store: pd.DataFrame):
        self.monthly_min = float(feature_store["monthly_charges"].min())
        self.monthly_max = float(feature_store["monthly_charges"].max())
        self.max_addons = max(float(feature_store["total_active_addons"].max()), 1.0)

    @property
    def info(self) -> dict:
        return {
            "churn_model": type(self.churn_model).__name__,
            "threshold": self.threshold,
            "kmeans_available": self.kmeans is not None and self.kmeans_scaler is not None,
            "n_clusters": int(getattr(self.kmeans, "n_clusters", 0) or 0),
            "priority_weights": self.priority_weights,
        }

    # ------------------------------------------------------------------
    # Ma trận đầu vào
    # ------------------------------------------------------------------
    @staticmethod
    def _model_frame(profiles) -> pd.DataFrame:
        df = pd.DataFrame(profiles if isinstance(profiles, list) else profiles).copy()
        for col in BOOL_FEATURES:
            if col in df.columns:
                df[col] = df[col].map(_as_bool).astype(int)
        return df

    def predict_proba(self, profiles) -> np.ndarray:
        X = self._model_frame(profiles)
        return self.churn_model.predict_proba(self.preprocessor.transform(X))[:, 1]

    def predict_cluster(self, profiles) -> Optional[np.ndarray]:
        if self.kmeans is None or self.kmeans_scaler is None:
            return None
        X = self._model_frame(profiles)[self.kmeans_features].astype(float)
        return self.kmeans.predict(self.kmeans_scaler.transform(X))

    def priority_score(self, prob, monthly, contract, addons) -> np.ndarray:
        prob, monthly, addons = map(lambda a: np.asarray(a, dtype=float), (prob, monthly, addons))
        rng = self.monthly_max - self.monthly_min
        monthly_value = np.clip((monthly - self.monthly_min) / rng, 0, 1) if rng > 0 else np.zeros_like(monthly)
        contract_opp = np.asarray(pd.Series(contract).map(self.contract_opportunity).fillna(0.5), dtype=float)
        service_opp = 1.0 - addons / self.max_addons
        w = self.priority_weights
        return 100 * (w["churn_risk"] * prob + w["monthly_value"] * monthly_value
                      + w["contract_opportunity"] * contract_opp
                      + w["service_opportunity"] * service_opp)

    # ------------------------------------------------------------------
    # 1. Chấm điểm 1 khách hàng
    # ------------------------------------------------------------------
    def score_customer(self, inp: dict, with_recommendations: bool = True) -> dict:
        p = build_customer_profile(inp)
        prob = float(self.predict_proba([p])[0])
        clusters = self.predict_cluster([p])
        cluster_id = int(clusters[0]) if clusters is not None else None
        tier = risk_tier(prob)
        result = {
            "profile": p,
            "churn_probability": prob,
            "churn_prediction": int(prob >= self.threshold),
            "threshold": self.threshold,
            "risk_tier": tier,
            "risk_tier_vi": RISK_LABELS_VI[tier],
            "cluster_id": cluster_id,
            "cluster_name": CLUSTER_NAMES.get(cluster_id, f"Cụm {cluster_id}") if cluster_id is not None else None,
            "cluster_strategy": CLUSTER_STRATEGY.get(cluster_id) if cluster_id is not None else None,
            "priority_score": float(self.priority_score(
                [prob], [p["monthly_charges"]], [p["contract_type"]], [p["total_active_addons"]])[0]),
            "expected_annual_revenue_at_risk": prob * p["monthly_charges"] * 12,
        }
        if with_recommendations:
            result["recommendations"] = self.recommend_actions(p, prob)
        return result

    # ------------------------------------------------------------------
    # 2. Gợi ý hành động giữ chân (counterfactual có định lượng)
    # ------------------------------------------------------------------
    def _candidate_actions(self, p: dict) -> List[dict]:
        actions = []
        if p["contract_type"] == "Month-to-month":
            actions.append({
                "action": "Ưu đãi chuyển sang hợp đồng 1 năm",
                "detail": "Tặng 1 tháng cước / giảm giá khi ký hợp đồng 12 tháng.",
                "changes": {"contract_type": "One year"},
            })
            actions.append({
                "action": "Ưu đãi chuyển sang hợp đồng 2 năm",
                "detail": "Khóa giá cước 24 tháng kèm quà tặng thiết bị.",
                "changes": {"contract_type": "Two year"},
            })
        elif p["contract_type"] == "One year":
            actions.append({
                "action": "Gia hạn lên hợp đồng 2 năm",
                "detail": "Ưu đãi gia hạn sớm cho khách hợp đồng 1 năm.",
                "changes": {"contract_type": "Two year"},
            })
        if not p["is_auto_payment"]:
            actions.append({
                "action": "Chuyển sang thanh toán tự động (thẻ tín dụng)",
                "detail": "Giảm 2-5% cước khi đăng ký auto-pay; giảm ma sát thanh toán.",
                "changes": {"payment_method": "Credit card (automatic)"},
            })
        if p["internet_service_type"] != "No":
            if not p["has_online_security"]:
                actions.append({
                    "action": "Tặng gói Online Security (3-6 tháng)",
                    "detail": "Add-on bảo mật miễn phí giai đoạn đầu, giữ nguyên cước.",
                    "changes": {"has_online_security": True},
                })
            if not p["has_tech_support"]:
                actions.append({
                    "action": "Tặng gói Tech Support ưu tiên",
                    "detail": "Hỗ trợ kỹ thuật ưu tiên, giảm sự cố gây rời mạng.",
                    "changes": {"has_tech_support": True},
                })
            if not p["has_online_security"] and not p["has_tech_support"]:
                actions.append({
                    "action": "Combo Security + Tech Support",
                    "detail": "Bundle 2 add-on bảo vệ, giữ nguyên cước tháng.",
                    "changes": {"has_online_security": True, "has_tech_support": True},
                })
        if p["monthly_charges"] > 70:
            actions.append({
                "action": "Giảm 10% cước hàng tháng",
                "detail": "Ưu đãi giá cho khách cước cao có nguy cơ rời mạng.",
                "changes": {"monthly_charges": round(p["monthly_charges"] * 0.9, 2)},
            })
        return actions

    def recommend_actions(self, p: dict, base_prob: Optional[float] = None, top_n: int = 5) -> List[dict]:
        if base_prob is None:
            base_prob = float(self.predict_proba([p])[0])
        candidates = self._candidate_actions(p)
        if not candidates:
            return []
        profiles = []
        for c in candidates:
            changed = dict(p)
            changed.update(c["changes"])
            # Giữ nguyên tổng cước lịch sử khi đổi giá; chỉ thay đổi cước tương lai
            changed["total_charges"] = p["total_charges"]
            profiles.append(build_customer_profile(changed))
        new_probs = self.predict_proba(profiles)
        recs = []
        for c, prof, new_p in zip(candidates, profiles, new_probs):
            delta = float(new_p) - base_prob
            monthly_cost = p["monthly_charges"] - prof["monthly_charges"]
            recs.append({
                "action": c["action"],
                "detail": c["detail"],
                "changes": c["changes"],
                "new_churn_probability": float(new_p),
                "delta": delta,
                "new_risk_tier": risk_tier(float(new_p)),
                # Doanh thu năm kỳ vọng giữ được: (giảm xác suất churn) x doanh thu năm sau ưu đãi - chi phí giảm giá
                "expected_annual_value": (-delta) * prof["monthly_charges"] * 12 - monthly_cost * 12 * (1 - float(new_p)),
            })
        recs.sort(key=lambda r: r["delta"])
        effective = [r for r in recs if r["delta"] < -0.005]
        return (effective or recs[:1])[:top_n]

    # ------------------------------------------------------------------
    # 3. What-If 1 khách hàng: baseline vs scenario
    # ------------------------------------------------------------------
    def compare_scenarios(self, baseline: dict, scenario: dict) -> dict:
        base = self.score_customer(baseline, with_recommendations=False)
        scen = self.score_customer(scenario, with_recommendations=True)
        changed = {
            k: {"baseline": base["profile"].get(k), "scenario": scen["profile"].get(k)}
            for k in scen["profile"]
            if k not in {"customer_id"} and base["profile"].get(k) != scen["profile"].get(k)
        }
        return {
            "baseline": base,
            "scenario": scen,
            "changed_fields": changed,
            "delta_probability": scen["churn_probability"] - base["churn_probability"],
            "delta_priority": scen["priority_score"] - base["priority_score"],
        }

    # ------------------------------------------------------------------
    # 4. Chấm điểm toàn bộ quần thể
    # ------------------------------------------------------------------
    def score_population(self, feature_store: pd.DataFrame) -> pd.DataFrame:
        df = feature_store.copy()
        df["churn_probability"] = self.predict_proba(df)
        clusters = self.predict_cluster(df)
        if clusters is not None:
            df["cluster_id"] = clusters.astype(int)
            df["cluster_name"] = df["cluster_id"].map(CLUSTER_NAMES).fillna("Cụm khác")
        df["risk_tier"] = pd.cut(df["churn_probability"], bins=RISK_BINS, labels=RISK_LABELS).astype(str)
        df["priority_score"] = self.priority_score(
            df["churn_probability"], df["monthly_charges"], df["contract_type"], df["total_active_addons"])
        df["priority_rank"] = df["priority_score"].rank(ascending=False, method="first").astype(int)
        return df.sort_values("priority_rank")

    # ------------------------------------------------------------------
    # 5. What-If cấp chính sách
    # ------------------------------------------------------------------
    def simulate_policy(self, feature_store: pd.DataFrame, scenario: dict,
                        segment_by: str = "contract_type", seed: int = 42) -> dict:
        """
        scenario keys (tỷ lệ 0-100):
          contract_migration_pct : % khách Month-to-month chuyển sang One year
          autopay_pct            : % khách thanh toán thủ công chuyển sang auto-pay
          security_bundle_pct    : % khách có Internet nhưng thiếu Security/TechSupport được tặng combo
          price_change_pct       : thay đổi cước tháng (%) cho toàn tập (âm = giảm giá)
          only_active            : chỉ mô phỏng trên khách chưa rời mạng (churn_flag=0)
        Lựa chọn khách được áp dụng: ưu tiên khách có rủi ro cao nhất (targeted campaign).
        """
        rng = np.random.default_rng(seed)
        base = feature_store.copy()
        if scenario.get("only_active"):
            base = base[base["churn_flag"] == 0]
        if base.empty:
            raise ValueError("Tập khách hàng mô phỏng rỗng.")
        base = base.reset_index(drop=True)
        base_prob = self.predict_proba(base)
        new = base.copy()
        targeting = scenario.get("targeting", "risk")  # risk | random

        def pick(mask: np.ndarray, pct: float) -> np.ndarray:
            idx = np.flatnonzero(mask)
            n = int(round(len(idx) * max(min(pct, 100), 0) / 100))
            if n == 0:
                return np.array([], dtype=int)
            if targeting == "random":
                return rng.choice(idx, size=n, replace=False)
            return idx[np.argsort(-base_prob[idx])][:n]

        applied = {}
        sel = pick((new["contract_type"] == "Month-to-month").to_numpy(), scenario.get("contract_migration_pct", 0))
        new.loc[sel, "contract_type"] = "One year"
        applied["contract_migration"] = int(len(sel))

        sel = pick((~new["is_auto_payment"].map(_as_bool)).to_numpy(), scenario.get("autopay_pct", 0))
        new.loc[sel, "payment_method"] = "Credit card (automatic)"
        new.loc[sel, "is_auto_payment"] = True
        applied["autopay"] = int(len(sel))

        mask = ((new["internet_service_type"] != "No")
                & ~(new["has_online_security"].map(_as_bool) & new["has_tech_support"].map(_as_bool))).to_numpy()
        sel = pick(mask, scenario.get("security_bundle_pct", 0))
        new.loc[sel, "has_online_security"] = True
        new.loc[sel, "has_tech_support"] = True
        addon_bool = new[ADDON_COLS].apply(lambda s: s.map(_as_bool))
        new["total_active_addons"] = addon_bool.sum(axis=1).astype(int)
        applied["security_bundle"] = int(len(sel))

        price = float(scenario.get("price_change_pct", 0) or 0)
        if price:
            new["monthly_charges"] = (new["monthly_charges"] * (1 + price / 100)).round(2)
        applied["price_change_pct"] = price

        new_prob = self.predict_proba(new)

        def summary(df, prob):
            return {
                "customers": int(len(df)),
                "avg_churn_probability": float(prob.mean()),
                "expected_churners": float(prob.sum()),
                "mrr": float(df["monthly_charges"].sum()),
                "expected_mrr_at_risk": float((prob * df["monthly_charges"]).sum()),
                "expected_retained_mrr": float(((1 - prob) * df["monthly_charges"]).sum()),
            }

        before, after = summary(base, base_prob), summary(new, new_prob)
        seg = segment_by if segment_by in base.columns else "contract_type"
        breakdown = pd.DataFrame({
            seg: base[seg].astype(str),
            "baseline_prob": base_prob,
            "scenario_prob": new_prob,
            "baseline_mrr_risk": base_prob * base["monthly_charges"],
            "scenario_mrr_risk": new_prob * new["monthly_charges"],
        }).groupby(seg).agg(
            customers=("baseline_prob", "size"),
            baseline_churn=("baseline_prob", "mean"),
            scenario_churn=("scenario_prob", "mean"),
            baseline_mrr_risk=("baseline_mrr_risk", "sum"),
            scenario_mrr_risk=("scenario_mrr_risk", "sum"),
        ).reset_index()
        breakdown["delta_churn"] = breakdown["scenario_churn"] - breakdown["baseline_churn"]

        return {
            "applied": applied,
            "baseline": before,
            "scenario": after,
            "delta": {k: after[k] - before[k] for k in before if k != "customers"},
            "breakdown": breakdown,
            "segment_by": seg,
        }


def json_safe(obj):
    """Chuyển numpy / pandas sang kiểu JSON thuần (dùng cho API)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return json_safe(obj.replace({np.nan: None}).to_dict(orient="records"))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
