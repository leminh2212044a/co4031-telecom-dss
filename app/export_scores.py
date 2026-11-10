"""
CO4031 - TELECOM DSS | Phase 3
Module: Xuất kết quả MBMS cho Power BI (app/export_scores.py)

Power BI không tự chạy được model .pkl khi người dùng tương tác, nên script này chạy model
một lần và lưu kết quả thành bảng để Power BI đọc:

1. ml_customer_scores   : điểm của từng khách hàng (churn %, risk tier, cụm K-Means, priority)
2. whatif_policy_grid   : kết quả mô phỏng chính sách tính sẵn bằng model cho từng mức 0-100%
                          (Power BI dùng What-If Parameter + LOOKUPVALUE để tra)
3. whatif_policy_segment: như trên nhưng chia theo loại hợp đồng

Chạy từ thư mục gốc dự án:
    python -m app.export_scores              # ghi CSV vào app/powerbi/data/
    python -m app.export_scores --to-db      # ghi thêm vào PostgreSQL (edw.ml_customer_scores,
                                             #   edw.ml_whatif_policy_grid, edw.ml_whatif_policy_segment)
Chạy lại script mỗi khi ETL nạp dữ liệu mới hoặc model được huấn luyện lại.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import config
from .services import DSSService

LEVERS = {
    "contract_migration": ("contract_migration_pct", "Chuyển Month-to-month sang hợp đồng 1 năm"),
    "autopay": ("autopay_pct", "Chuyển thanh toán thủ công sang auto-pay"),
    "security_bundle": ("security_bundle_pct", "Tặng combo Online Security + Tech Support"),
}
STEPS = list(range(0, 101, 10))


def build_scores(svc: DSSService) -> pd.DataFrame:
    df = svc.scored_population()
    out = pd.DataFrame({
        "customer_id": df["customer_id"].astype(str),
        "churn_probability": df["churn_probability"].round(4),
        "churn_prediction": (df["churn_probability"] >= svc.models.threshold).astype(int),
        "risk_tier": df["risk_tier"],
        "cluster_id": df.get("cluster_id"),
        "cluster_name": df.get("cluster_name"),
        "priority_score": df["priority_score"].round(2),
        "priority_rank": df["priority_rank"].astype(int),
        "expected_mrr_at_risk": (df["churn_probability"] * df["monthly_charges"]).round(2),
        "model_name": type(svc.models.churn_model).__name__,
    })
    return out.sort_values("priority_rank").reset_index(drop=True)


def build_whatif(svc: DSSService):
    grid_rows, seg_rows = [], []
    for lever, (param, label) in LEVERS.items():
        for pct in STEPS:
            sim = svc.simulate_policy({param: pct, "only_active": True, "targeting": "risk"})
            sc, base = sim["scenario"], sim["baseline"]
            grid_rows.append({
                "lever": lever, "lever_label": label, "pct": pct,
                "customers": sc["customers"],
                "avg_churn_probability": round(sc["avg_churn_probability"], 5),
                "expected_churners": round(sc["expected_churners"], 2),
                "expected_mrr_at_risk": round(sc["expected_mrr_at_risk"], 2),
                "expected_retained_mrr": round(sc["expected_retained_mrr"], 2),
                "delta_churners": round(sc["expected_churners"] - base["expected_churners"], 2),
                "delta_retained_mrr": round(sc["expected_retained_mrr"] - base["expected_retained_mrr"], 2),
                "customers_changed": sim["applied"][lever],
            })
            bd = sim["breakdown"]
            for _, r in bd.iterrows():
                seg_rows.append({
                    "lever": lever, "pct": pct, "contract_type": r["contract_type"],
                    "customers": int(r["customers"]),
                    "baseline_churn": round(float(r["baseline_churn"]), 5),
                    "scenario_churn": round(float(r["scenario_churn"]), 5),
                    "baseline_mrr_risk": round(float(r["baseline_mrr_risk"]), 2),
                    "scenario_mrr_risk": round(float(r["scenario_mrr_risk"]), 2),
                })
    return pd.DataFrame(grid_rows), pd.DataFrame(seg_rows)


def write_csv(scores, grid, seg):
    config.POWERBI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "ml_customer_scores.csv": scores,
        "whatif_policy_grid.csv": grid,
        "whatif_policy_segment.csv": seg,
    }
    for name, df in paths.items():
        df.to_csv(config.POWERBI_DATA_DIR / name, index=False, encoding="utf-8-sig")
        print(f"[OK] {config.POWERBI_DATA_DIR / name}  ({len(df):,} dòng)")


def write_db(scores, grid, seg):
    import psycopg2
    from psycopg2.extras import execute_values

    ddl = (config.APP_DIR / "sql" / "05_ml_customer_scores.sql").read_text(encoding="utf-8")
    conn = psycopg2.connect(**config.DB_CONFIG)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute(f"TRUNCATE {config.SCORES_TABLE};")
            cols = list(scores.columns)
            rows = [tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in r)
                    for r in scores.itertuples(index=False)]
            execute_values(cur, f"INSERT INTO {config.SCORES_TABLE} ({', '.join(cols)}) VALUES %s", rows)

            for table, df in (("edw.ml_whatif_policy_grid", grid), ("edw.ml_whatif_policy_segment", seg)):
                col_defs = ", ".join(
                    f"{c} {'TEXT' if df[c].dtype == object else 'DOUBLE PRECISION'}" for c in df.columns)
                cur.execute(f"DROP TABLE IF EXISTS {table}; CREATE TABLE {table} ({col_defs});")
                rows = [tuple(v.item() if hasattr(v, "item") else v for v in r) for r in df.itertuples(index=False)]
                execute_values(cur, f"INSERT INTO {table} ({', '.join(df.columns)}) VALUES %s", rows)
        print(f"[OK] Đã ghi {len(scores):,} dòng vào {config.SCORES_TABLE}, "
              "tạo view edw.vw_bi_churn_scored và 2 bảng What-If.")
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Xuất điểm ML và lưới What-If cho Power BI")
    parser.add_argument("--to-db", action="store_true", help="Ghi thêm vào PostgreSQL schema edw")
    parser.add_argument("--source", default=config.DATA_SOURCE, choices=["auto", "postgres", "csv"],
                        help="Nguồn đọc Feature Store (mặc định theo DSS_DATA_SOURCE)")
    args = parser.parse_args(argv)

    svc = DSSService(args.source)
    print(f"[INFO] Nguồn dữ liệu: {svc.dw.source_label}")
    scores = build_scores(svc)
    print("[INFO] Đang mô phỏng lưới What-If (3 đòn bẩy × 11 mức)...")
    grid, seg = build_whatif(svc)
    write_csv(scores, grid, seg)
    if args.to_db:
        write_db(scores, grid, seg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
