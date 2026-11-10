"""
CO4031 - TELECOM DSS | Phase 3
Module: Streamlit DSS Dashboard (app/main.py)

Chạy (từ thư mục gốc dự án):
    streamlit run app/main.py

Thành phần User Interface của DSS:
  1. Tổng quan      : KPI + biểu đồ churn/doanh thu theo chiều, bản đồ GIS
  2. OLAP Explorer  : Drill-down / Roll-up theo phân cấp, Slice/Dice (sidebar), Pivot, ROLLUP subtotals
  3. Phân khúc      : hồ sơ 4 cụm K-Means + chiến lược theo cụm
  4. Ưu tiên giữ chân: danh sách Top-K theo priority score (Model 3)
  5. What-If khách hàng: dự báo Churn % real-time, so sánh kịch bản, gợi ý hành động có định lượng
  6. Mô phỏng chính sách: What-If cấp tập khách hàng (hợp đồng, auto-pay, bundle, giá)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.client import make_client
from app.data_access import DIMENSIONS, HIERARCHIES, MEASURES
from app.ml_service import (CLUSTER_STRATEGY, CONTRACT_TYPES, INTERNET_TYPES,
                            MULTIPLE_LINES, PAYMENT_METHODS, RISK_LABELS_VI)

st.set_page_config(page_title="Telecom DSS · CO4031", page_icon="📡", layout="wide")

# ----------------------------------------------------------------------
# Giao diện chung
# ----------------------------------------------------------------------
CHURN_COLOR = "#C8423A"
STAY_COLOR = "#2E7D8C"
ACCENT = "#E0A526"
NEUTRAL = "#8A94A6"
TIER_COLORS = {"Low": "#2E7D8C", "Medium": "#E0A526", "High": "#E07B39", "Critical": "#C8423A"}
CLUSTER_PALETTE = ["#2E7D8C", "#C8423A", "#E0A526", "#5B5EA6"]
TEMPLATE = "plotly_white"

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
  div[data-testid="stMetric"] {background: rgba(46,125,140,0.06); border: 1px solid rgba(46,125,140,0.18);
      border-radius: 10px; padding: 10px 14px;}
  div[data-testid="stMetricLabel"] p {font-size: 0.82rem; opacity: 0.8;}
  .crumb {font-family: ui-monospace, Menlo, monospace; font-size: 0.9rem; padding: 6px 10px;
      border-radius: 6px; background: rgba(138,148,166,0.12); display: inline-block;}
  .note {font-size: 0.85rem; opacity: 0.75;}
</style>
""", unsafe_allow_html=True)


def fmt_money(v):
    return f"${v:,.0f}"


def fmt_pct(v):
    return f"{v * 100:.1f}%"


def dim_label(col):
    return DIMENSIONS.get(col, col)


def measure_label(m):
    return MEASURES[m][1]


MEASURE_FORMAT = {
    "customers": "{:,.0f}", "churned": "{:,.0f}", "churn_rate": "{:.1%}",
    "mrr": "${:,.0f}", "revenue_at_risk": "${:,.0f}", "avg_monthly_charges": "${:,.2f}",
    "total_revenue": "${:,.0f}", "avg_tenure": "{:.1f}", "avg_clv": "${:,.0f}",
}


def style_measures(df):
    fmt = {k: v for k, v in MEASURE_FORMAT.items() if k in df.columns}
    return df.style.format(fmt, na_rep="–")


# ----------------------------------------------------------------------
# Client & cache
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang kết nối Kho dữ liệu & nạp mô hình...")
def get_client(mode: str):
    return make_client(mode)


def _key(obj):
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)


@st.cache_data(show_spinner=False, ttl=900)
def _cached(method: str, payload: str, mode: str):
    args = json.loads(payload)
    return getattr(get_client(mode), method)(*args["a"], **args["k"])


def call(method, *args, **kwargs):
    return _cached(method, _key({"a": list(args), "k": kwargs}), st.session_state["backend_mode"])


def call_nocache(method, *args, **kwargs):
    return getattr(get_client(st.session_state["backend_mode"]), method)(*args, **kwargs)


# ----------------------------------------------------------------------
# SIDEBAR: nguồn dữ liệu + SLICE & DICE
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📡 Telecom DSS")
    st.caption("CO4031 · Kho dữ liệu & Hệ hỗ trợ quyết định")
    st.session_state.setdefault("backend_mode", "auto")
    st.selectbox("Backend", ["auto", "api", "local"], key="backend_mode",
                 help="auto: dùng FastAPI nếu đang chạy, nếu không gọi trực tiếp DW + model.")
    try:
        client = get_client(st.session_state["backend_mode"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Không khởi tạo được backend: {exc}")
        st.stop()
    st.caption(f"🔌 {client.label}")

    st.markdown("### 🔪 Slice & Dice")
    st.caption("Bộ lọc áp dụng cho mọi tab (trừ What-If khách hàng). Để trống = tất cả.")

    def ms(col, options=None):
        opts = options if options is not None else call("distinct_values", col)
        return st.multiselect(dim_label(col), opts, key=f"f_{col}")

    f_region = ms("region")
    geo = call("aggregate", ["region", "city"])
    city_opts = sorted(geo[geo["region"].isin(f_region)]["city"].unique()) if f_region else \
        sorted(geo["city"].unique())
    ms("city", city_opts)
    ms("contract_type")
    ms("internet_service_type")
    ms("payment_category")
    ms("tenure_group")
    with st.expander("Thêm chiều lọc"):
        ms("gender")
        ms("is_senior_citizen")
        ms("has_tech_support")
        ms("total_active_addons")

    if st.button("↺ Xóa bộ lọc", use_container_width=True):
        for k in list(st.session_state):
            if k.startswith("f_"):
                st.session_state[k] = []
        st.rerun()

FILTERS = {k[2:]: v for k, v in st.session_state.items() if k.startswith("f_") and v}
if FILTERS:
    with st.sidebar:
        st.info("Đang lọc: " + "; ".join(f"{dim_label(k)} ∈ {v}" for k, v in FILTERS.items()))

st.title("Hệ hỗ trợ ra quyết định giữ chân khách hàng Viễn thông")
st.caption("DBMS: PostgreSQL Star Schema (edw) · MBMS: Random Forest Churn, K-Means k=4, Retention Priority · "
           "UI: Streamlit + FastAPI")

tabs = st.tabs(["📊 Tổng quan", "🧭 OLAP Explorer", "🧩 Phân khúc", "🎯 Ưu tiên giữ chân",
                "🔮 What-If khách hàng", "🏛️ Mô phỏng chính sách"])


def bar_churn(dim, title, order=None):
    df = call("aggregate", [dim], FILTERS)
    if df.empty:
        return None
    df[dim] = df[dim].astype(str)
    fig = go.Figure()
    fig.add_bar(x=df[dim], y=df["customers"] - df["churned"], name="Ở lại", marker_color=STAY_COLOR)
    fig.add_bar(x=df[dim], y=df["churned"], name="Rời mạng", marker_color=CHURN_COLOR)
    fig.add_scatter(x=df[dim], y=df["churn_rate"] * 100, name="Tỷ lệ churn (%)", yaxis="y2",
                    mode="lines+markers+text", text=[f"{v:.1%}" for v in df["churn_rate"]],
                    textposition="top center", line=dict(color=ACCENT, width=2))
    fig.update_layout(
        title=title, barmode="stack", template=TEMPLATE, height=360, margin=dict(t=50, b=10, l=10, r=10),
        yaxis=dict(title="Khách hàng"),
        yaxis2=dict(title="Churn %", overlaying="y", side="right", rangemode="tozero", showgrid=False),
        legend=dict(orientation="h", y=-0.18),
        xaxis=dict(categoryorder="array", categoryarray=order) if order else {},
    )
    return fig


# ======================================================================
# TAB 1: TỔNG QUAN
# ======================================================================
with tabs[0]:
    k = call("kpis", FILTERS)
    c = st.columns(6)
    c[0].metric("Khách hàng", f"{k['customers']:,.0f}")
    c[1].metric("Tỷ lệ churn", fmt_pct(k["churn_rate"]))
    c[2].metric("Khách rời mạng", f"{k['churned']:,.0f}")
    c[3].metric("MRR", fmt_money(k["mrr"]))
    c[4].metric("MRR mất do churn", fmt_money(k["revenue_at_risk"]),
                f"{(k['revenue_at_risk'] / k['mrr'] * 100 if k['mrr'] else 0):.1f}% MRR", delta_color="inverse")
    c[5].metric("Thâm niên TB", f"{k['avg_tenure']:.1f} tháng")

    if k["customers"] == 0:
        st.warning("Không có khách hàng nào khớp bộ lọc.")
    else:
        r1 = st.columns(2)
        r1[0].plotly_chart(bar_churn("contract_type", "Churn theo loại hợp đồng", CONTRACT_TYPES),
                           use_container_width=True)
        r1[1].plotly_chart(bar_churn("tenure_group", "Churn theo nhóm thâm niên"), use_container_width=True)
        r2 = st.columns(2)
        r2[0].plotly_chart(bar_churn("internet_service_type", "Churn theo loại Internet"),
                           use_container_width=True)
        r2[1].plotly_chart(bar_churn("payment_category", "Churn theo hình thức thanh toán"),
                           use_container_width=True)

        r3 = st.columns([3, 2])
        with r3[0]:
            pts = call("geo_points", FILTERS)
            pts = pts[(pts["latitude"] != 0) & (pts["longitude"] != 0)]
            if not pts.empty:
                fig = px.scatter_map(
                    pts, lat="latitude", lon="longitude", color="churn_label",
                    color_discrete_map={"Yes": CHURN_COLOR, "No": STAY_COLOR},
                    hover_data={"customer_id": True, "city": True, "monthly_charges": ":.2f",
                                "latitude": False, "longitude": False},
                    zoom=5, height=420, opacity=0.6, title="Bản đồ khách hàng (GIS)")
                fig.update_layout(map_style="open-street-map", margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
        with r3[1]:
            reg = call("aggregate", ["region"], FILTERS).sort_values("revenue_at_risk")
            fig = px.bar(reg, x="revenue_at_risk", y="region", orientation="h", template=TEMPLATE,
                         text=[fmt_money(v) for v in reg["revenue_at_risk"]], height=420,
                         title="MRR mất do churn theo vùng", color_discrete_sequence=[CHURN_COLOR],
                         labels={"revenue_at_risk": "MRR mất ($)", "region": ""})
            fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# TAB 2: OLAP EXPLORER
# ======================================================================
with tabs[1]:
    st.subheader("Drill-down / Roll-up theo phân cấp")
    oc = st.columns([2, 2, 3])
    hierarchy = oc[0].selectbox("Phân cấp (Hierarchy)", list(HIERARCHIES))
    measure = oc[1].selectbox("Độ đo", list(MEASURES), index=list(MEASURES).index("churn_rate"),
                              format_func=measure_label)
    path_key = f"drill_{hierarchy}"
    st.session_state.setdefault(path_key, [])
    path = st.session_state[path_key]
    levels = HIERARCHIES[hierarchy]
    oc[2].markdown("**Đường drill hiện tại**")
    crumbs = ["ALL"] + [f"{dim_label(l)}={v}" for l, v in zip(levels, path)]
    oc[2].markdown(f"<span class='crumb'>{' ▸ '.join(crumbs)}</span>", unsafe_allow_html=True)

    res = call("drill", hierarchy, path, FILTERS)
    data = res["data"].copy()
    level = res["level"]
    data[level] = data[level].astype(str)

    if data.empty:
        st.warning("Không có dữ liệu ở cấp này với bộ lọc hiện tại.")
        if path and st.button("⟲ Về gốc phân cấp"):
            st.session_state[path_key] = []
            st.rerun()
    else:
        bc = st.columns([3, 2])
        fig = px.bar(data.sort_values(measure, ascending=False), x=level, y=measure, template=TEMPLATE,
                     color="churn_rate", color_continuous_scale=[[0, STAY_COLOR], [0.5, ACCENT], [1, CHURN_COLOR]],
                     hover_data=["customers", "churned", "mrr"], height=380,
                     labels={level: dim_label(level), measure: measure_label(measure), "churn_rate": "Churn"},
                     title=f"{measure_label(measure)} theo {dim_label(level)} (cấp {res['depth'] + 1}/{len(levels)})")
        if measure == "churn_rate":
            fig.update_yaxes(tickformat=".0%")
        fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), coloraxis_colorbar=dict(tickformat=".0%"))
        bc[0].plotly_chart(fig, use_container_width=True)
        with bc[1]:
            st.markdown(f"**Cấp hiện tại:** {dim_label(level)}")
            if res["can_drill_down"]:
                member = st.selectbox("Chọn phần tử để drill-down", data[level].tolist(),
                                      key=f"member_{hierarchy}_{len(path)}")
                if st.button(f"⬇ Drill-down xuống {dim_label(levels[len(path) + 1])}", use_container_width=True):
                    st.session_state[path_key] = path + [member]
                    st.rerun()
            else:
                st.caption("Đã ở cấp chi tiết nhất của phân cấp.")
            b2 = st.columns(2)
            if b2[0].button("⬆ Roll-up", disabled=not path, use_container_width=True):
                st.session_state[path_key] = path[:-1]
                st.rerun()
            if b2[1].button("⟲ Về gốc", disabled=not path, use_container_width=True):
                st.session_state[path_key] = []
                st.rerun()
            st.markdown("<p class='note'>Drill-down thêm điều kiện của cấp cha vào WHERE; Roll-up bỏ điều kiện "
                        "cấp cuối và gộp lên cấp cao hơn.</p>", unsafe_allow_html=True)
        st.dataframe(style_measures(data), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Pivot (Dice 2 chiều)")
    pc = st.columns(3)
    dims = list(DIMENSIONS)
    prow = pc[0].selectbox("Chiều dòng", dims, index=dims.index("contract_type"), format_func=dim_label)
    pcol = pc[1].selectbox("Chiều cột", dims, index=dims.index("internet_service_type"), format_func=dim_label)
    pmeasure = pc[2].selectbox("Độ đo pivot", list(MEASURES), index=list(MEASURES).index("churn_rate"),
                               format_func=measure_label, key="pivot_measure")
    if prow == pcol:
        st.warning("Chọn hai chiều khác nhau.")
    else:
        table = call("pivot", prow, pcol, pmeasure, FILTERS)
        if table.empty:
            st.warning("Không có dữ liệu.")
        else:
            is_rate = pmeasure == "churn_rate"
            fig = px.imshow(table * (100 if is_rate else 1), text_auto=".1f" if is_rate else ",.0f",
                            aspect="auto", color_continuous_scale=[[0, "#EAF4F5"], [0.5, ACCENT], [1, CHURN_COLOR]],
                            labels=dict(x=dim_label(pcol), y=dim_label(prow),
                                        color=measure_label(pmeasure) + (" (%)" if is_rate else "")),
                            height=max(300, 60 * len(table) + 120))
            fig.update_layout(template=TEMPLATE, margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Roll-up với dòng tổng con (GROUP BY ROLLUP)")
    ru_levels = st.multiselect("Các cấp gộp (theo thứ tự)", dims, default=["region", "contract_type"],
                               format_func=dim_label, max_selections=3)
    if ru_levels:
        ru = call("rollup", ru_levels, FILTERS)

        def highlight(row):
            if row["rollup_level"] < len(ru_levels):
                weight = "font-weight: 700;" + (" background-color: rgba(224,165,38,0.15);"
                                                 if row["rollup_level"] > 0 else
                                                 " background-color: rgba(200,66,58,0.12);")
                return [weight] * len(row)
            return [""] * len(row)

        st.dataframe(style_measures(ru).apply(highlight, axis=1), use_container_width=True, hide_index=True,
                     height=min(600, 38 * (len(ru) + 1)))
        st.caption("Dòng tô vàng = tổng con của cấp cha; dòng tô đỏ = tổng toàn bộ (grand total).")

# ======================================================================
# TAB 3: PHÂN KHÚC K-MEANS
# ======================================================================
with tabs[2]:
    prof = call("segment_profile", FILTERS)
    if prof is None or prof.empty:
        st.info("Chưa có model K-Means (models/saved/kmeans_model.pkl) hoặc không có dữ liệu.")
    else:
        prof = prof.sort_values("cluster_id")
        cols = st.columns(len(prof))
        for col, (_, row) in zip(cols, prof.iterrows()):
            col.metric(f"Cụm {int(row['cluster_id'])} · {row['cluster_name']}", f"{row['customers']:,.0f} KH",
                       f"churn thực tế {fmt_pct(row['actual_churn_rate'])}", delta_color="off")
        sc = st.columns(2)
        prof["label"] = "Cụm " + prof["cluster_id"].astype(int).astype(str)
        m = prof.melt(id_vars="label", value_vars=["actual_churn_rate", "avg_churn_probability"],
                      var_name="metric", value_name="v")
        m["metric"] = m["metric"].map({"actual_churn_rate": "Churn thực tế",
                                       "avg_churn_probability": "Xác suất churn dự báo TB"})
        fig = px.bar(m, x="label", y="v", color="metric", barmode="group", template=TEMPLATE, height=360,
                     color_discrete_sequence=[CHURN_COLOR, ACCENT], title="Rủi ro churn theo cụm",
                     labels={"label": "", "v": "", "metric": ""})
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(legend=dict(orientation="h", y=-0.15), margin=dict(t=50, b=10, l=10, r=10))
        sc[0].plotly_chart(fig, use_container_width=True)

        pts = call("segment_points", FILTERS)
        fig = px.scatter(pts, x="tenure_months", y="monthly_charges", color="cluster_name", template=TEMPLATE,
                         opacity=0.55, height=360, color_discrete_sequence=CLUSTER_PALETTE,
                         title="Thâm niên × Cước tháng (mẫu khách hàng)",
                         labels={"tenure_months": "Thâm niên (tháng)", "monthly_charges": "Cước tháng ($)",
                                 "cluster_name": "Cụm"})
        fig.update_layout(legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10, l=10, r=10))
        sc[1].plotly_chart(fig, use_container_width=True)

        show = prof[["cluster_id", "cluster_name", "customers", "share", "avg_tenure", "avg_monthly_charges",
                     "avg_addons", "actual_churn_rate", "avg_churn_probability", "mrr"]].copy()
        show["Chiến lược DSS"] = show["cluster_id"].map(CLUSTER_STRATEGY)
        st.dataframe(show.style.format({
            "share": "{:.1%}", "avg_tenure": "{:.1f}", "avg_monthly_charges": "${:.2f}", "avg_addons": "{:.1f}",
            "actual_churn_rate": "{:.1%}", "avg_churn_probability": "{:.1%}", "mrr": "${:,.0f}",
            "customers": "{:,.0f}"}), use_container_width=True, hide_index=True)

# ======================================================================
# TAB 4: ƯU TIÊN GIỮ CHÂN
# ======================================================================
with tabs[3]:
    rc = st.columns([2, 3, 2])
    top_k = rc[0].slider("Số khách hàng Top-K", 10, 1000, 100, step=10)
    tiers = rc[1].multiselect("Mức rủi ro", ["Critical", "High", "Medium", "Low"],
                              format_func=lambda t: RISK_LABELS_VI[t])
    only_active = rc[2].checkbox("Chỉ khách đang hoạt động", True,
                                 help="Bỏ khách đã churn trong snapshot - đối tượng chiến dịch là khách còn ở lại.")
    top = call("retention_top", top_k, FILTERS, tiers or None, only_active)
    if top.empty:
        st.warning("Không có khách hàng phù hợp.")
    else:
        mc = st.columns(4)
        mc[0].metric("Khách trong danh sách", f"{len(top):,}")
        mc[1].metric("Xác suất churn TB", fmt_pct(top["churn_probability"].mean()))
        mc[2].metric("MRR của nhóm", fmt_money(top["monthly_charges"].sum()))
        mc[3].metric("MRR kỳ vọng mất", fmt_money((top["churn_probability"] * top["monthly_charges"]).sum()))
        st.dataframe(
            top, use_container_width=True, hide_index=True, height=420,
            column_config={
                "priority_score": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100,
                                                                  format="%.1f"),
                "churn_probability": st.column_config.ProgressColumn("Churn %", min_value=0, max_value=1,
                                                                     format="%.2f"),
                "monthly_charges": st.column_config.NumberColumn("Cước tháng", format="$%.2f"),
            })
        st.download_button("⬇ Tải danh sách chiến dịch (CSV)", top.to_csv(index=False).encode("utf-8"),
                           "retention_campaign_list.csv", "text/csv")
    metrics = call("retention_metrics")
    if metrics is not None and len(metrics):
        st.markdown("**Hiệu quả xếp hạng (5-fold out-of-fold, Model 3)**")
        mt = metrics[["top_fraction", "top_k_customers", "precision_at_k", "recall_at_k", "lift_at_k"]].copy()
        st.dataframe(mt.style.format({"top_fraction": "{:.0%}", "precision_at_k": "{:.1%}",
                                      "recall_at_k": "{:.1%}", "lift_at_k": "{:.2f}x"}),
                     use_container_width=True, hide_index=True)

# ======================================================================
# TAB 5: WHAT-IF KHÁCH HÀNG
# ======================================================================
DEFAULT_PROFILE = {
    "gender": "Female", "is_senior_citizen": False, "has_partner": False, "has_dependents": False,
    "region": "Bay Area", "tenure_months": 2, "has_phone_service": True, "multiple_lines": "No",
    "internet_service_type": "Fiber optic", "has_online_security": False, "has_online_backup": False,
    "has_device_protection": False, "has_tech_support": False, "has_streaming_tv": True,
    "has_streaming_movies": True, "contract_type": "Month-to-month", "is_paperless_billing": True,
    "payment_method": "Electronic check", "monthly_charges": 85.5, "total_charges": 171.0,
}
WI_FIELDS = list(DEFAULT_PROFILE)


def load_profile_into_form():
    cid = st.session_state.get("wi_customer")
    rec = DEFAULT_PROFILE
    if cid and cid != "— Khách hàng mới —":
        r = call("customer", cid)
        if r:
            rec = {f: r.get(f, DEFAULT_PROFILE[f]) for f in WI_FIELDS}
    st.session_state["wi_baseline"] = dict(rec)
    st.session_state.pop("wi_result", None)
    for f in WI_FIELDS:
        v = rec[f]
        if isinstance(DEFAULT_PROFILE[f], bool):
            v = str(v).lower() in ("true", "1", "yes")
        elif isinstance(DEFAULT_PROFILE[f], float):
            v = float(v)
        elif isinstance(DEFAULT_PROFILE[f], int):
            v = int(v)
        st.session_state[f"wi_{f}"] = v


def gauge(prob, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "valueformat": ".1f"},
        title={"text": title, "font": {"size": 15}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1F2A3C", "thickness": 0.25},
               "steps": [{"range": [0, 30], "color": "#D6EBEE"}, {"range": [30, 50], "color": "#F6E3B4"},
                         {"range": [50, 70], "color": "#F3C9A8"}, {"range": [70, 100], "color": "#EBB4AE"}]}))
    fig.update_layout(height=230, margin=dict(t=50, b=10, l=25, r=25))
    return fig


with tabs[4]:
    st.markdown("Chọn một khách hàng có sẵn trong DW (hoặc tạo khách mới), thay đổi thông số rồi so sánh "
                "**Baseline ↔ Kịch bản**. Model chạy real-time trên `churn_model.pkl` + `kmeans_model.pkl`.")
    if "wi_baseline" not in st.session_state:
        st.session_state["wi_customer"] = "— Khách hàng mới —"
        load_profile_into_form()
    ids = ["— Khách hàng mới —"] + call("customer_ids", 300)
    st.selectbox("Nạp hồ sơ khách hàng (300 khách đang hoạt động có rủi ro cao nhất)", ids,
                 key="wi_customer", on_change=load_profile_into_form)
    regions = call("distinct_values", "region")
    if st.session_state.get("wi_region") not in regions:
        st.session_state["wi_region"] = regions[0]

    with st.form("whatif_form"):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.markdown("**Nhân khẩu học**")
            st.selectbox("Giới tính", ["Female", "Male"], key="wi_gender")
            st.selectbox("Vùng", regions, key="wi_region")
            st.checkbox("Người cao tuổi", key="wi_is_senior_citizen")
            st.checkbox("Có bạn đời", key="wi_has_partner")
            st.checkbox("Có người phụ thuộc", key="wi_has_dependents")
        with f2:
            st.markdown("**Hợp đồng & Thanh toán**")
            st.selectbox("Loại hợp đồng", CONTRACT_TYPES, key="wi_contract_type")
            st.selectbox("Phương thức thanh toán", PAYMENT_METHODS, key="wi_payment_method")
            st.checkbox("Hóa đơn điện tử", key="wi_is_paperless_billing")
            st.slider("Thâm niên (tháng)", 0, 72, key="wi_tenure_months")
            st.number_input("Cước tháng ($)", 0.0, 200.0, step=1.0, key="wi_monthly_charges")
            st.number_input("Tổng cước tích lũy ($)", 0.0, 10000.0, step=10.0, key="wi_total_charges",
                            help="Đặt 0 để tự tính = thâm niên × cước tháng.")
        with f3:
            st.markdown("**Dịch vụ**")
            st.checkbox("Dịch vụ thoại", key="wi_has_phone_service")
            st.selectbox("Nhiều đường thoại", MULTIPLE_LINES, key="wi_multiple_lines")
            st.selectbox("Loại Internet", INTERNET_TYPES, key="wi_internet_service_type")
        with f4:
            st.markdown("**Add-on** (bỏ qua nếu không có Internet)")
            st.checkbox("Online Security", key="wi_has_online_security")
            st.checkbox("Online Backup", key="wi_has_online_backup")
            st.checkbox("Device Protection", key="wi_has_device_protection")
            st.checkbox("Tech Support", key="wi_has_tech_support")
            st.checkbox("Streaming TV", key="wi_has_streaming_tv")
            st.checkbox("Streaming Movies", key="wi_has_streaming_movies")
        submitted = st.form_submit_button("🔮 Chạy dự báo & so sánh kịch bản", type="primary",
                                          use_container_width=True)

    if submitted or "wi_result" in st.session_state:
        scenario = {f: st.session_state[f"wi_{f}"] for f in WI_FIELDS}
        baseline = dict(st.session_state["wi_baseline"])
        for prof in (scenario, baseline):
            if not prof.get("total_charges"):
                prof["total_charges"] = None
        if submitted:
            try:
                st.session_state["wi_result"] = call_nocache("compare", baseline, scenario)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Lỗi dự báo: {exc}")
                st.stop()
        out = st.session_state["wi_result"]
        b, s = out["baseline"], out["scenario"]

        g = st.columns([1, 1, 1.3])
        g[0].plotly_chart(gauge(b["churn_probability"], "Baseline"), use_container_width=True)
        g[1].plotly_chart(gauge(s["churn_probability"], "Kịch bản"), use_container_width=True)
        with g[2]:
            st.metric("Thay đổi xác suất churn", fmt_pct(s["churn_probability"]),
                      f"{out['delta_probability'] * 100:+.1f} điểm %", delta_color="inverse")
            st.metric("Mức rủi ro", RISK_LABELS_VI[s["risk_tier"]],
                      f"Dự đoán: {'RỜI MẠNG' if s['churn_prediction'] else 'Ở lại'} (ngưỡng {s['threshold']:.2f})",
                      delta_color="off")
            st.metric("Priority score", f"{s['priority_score']:.1f}", f"{out['delta_priority']:+.1f}",
                      delta_color="inverse")
            if s.get("cluster_name"):
                st.markdown(f"**Phân khúc:** Cụm {s['cluster_id']} · {s['cluster_name']}  \n"
                            f"<span class='note'>{s['cluster_strategy']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Doanh thu năm kỳ vọng mất:** {fmt_money(s['expected_annual_revenue_at_risk'])}")

        if out["changed_fields"]:
            ch = pd.DataFrame([{"Thuộc tính": k, "Baseline": str(v["baseline"]), "Kịch bản": str(v["scenario"])}
                               for k, v in out["changed_fields"].items()])
            with st.expander(f"Các thuộc tính đã thay đổi ({len(ch)})", expanded=False):
                st.dataframe(ch, use_container_width=True, hide_index=True)

        st.markdown("#### 💡 Gợi ý hành động giữ chân (định lượng bằng model)")
        recs = pd.DataFrame(s.get("recommendations", []))
        if recs.empty:
            st.info("Không có hành động phù hợp cho hồ sơ này.")
        else:
            rcol = st.columns([3, 2])
            with rcol[0]:
                view = recs[["action", "detail", "new_churn_probability", "delta", "expected_annual_value"]].rename(
                    columns={"action": "Hành động", "detail": "Mô tả", "new_churn_probability": "Churn sau",
                             "delta": "Δ churn", "expected_annual_value": "Giá trị kỳ vọng/năm"})
                st.dataframe(view.style.format({"Churn sau": "{:.1%}", "Δ churn": "{:+.1%}",
                                                "Giá trị kỳ vọng/năm": "${:,.0f}"}),
                             use_container_width=True, hide_index=True)
            with rcol[1]:
                fig = go.Figure(go.Bar(x=recs["delta"] * 100, y=recs["action"], orientation="h",
                                       marker_color=[STAY_COLOR if d < 0 else CHURN_COLOR for d in recs["delta"]],
                                       text=[f"{d * 100:+.1f}" for d in recs["delta"]]))
                fig.update_layout(template=TEMPLATE, height=60 * len(recs) + 80, title="Δ xác suất churn (điểm %)",
                                  yaxis=dict(autorange="reversed"), margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)
            st.caption("Δ churn là thay đổi rủi ro dự báo khi hồ sơ khách hàng có đặc điểm mới (mô hình tương quan); "
                       "hiệu quả thực của ưu đãi cần kiểm chứng bằng A/B test.")

# ======================================================================
# TAB 6: MÔ PHỎNG CHÍNH SÁCH
# ======================================================================
with tabs[5]:
    st.markdown("Mô phỏng tác động của chính sách lên **toàn bộ tập khách hàng đã lọc** ở sidebar. "
                "Mỗi khách được chấm điểm lại bằng model sau khi áp dụng thay đổi.")
    pc = st.columns(4)
    s_contract = pc[0].slider("% khách Month-to-month → hợp đồng 1 năm", 0, 100, 30, 5)
    s_autopay = pc[1].slider("% khách thanh toán thủ công → auto-pay", 0, 100, 0, 5)
    s_bundle = pc[2].slider("% khách Internet thiếu bảo vệ → tặng Security + TechSupport", 0, 100, 0, 5)
    s_price = pc[3].slider("Thay đổi cước tháng (%)", -30, 30, 0, 1)
    oc2 = st.columns(3)
    s_target = oc2[0].radio("Chọn khách áp dụng", ["risk", "random"], horizontal=True,
                            format_func=lambda x: "Rủi ro cao nhất trước" if x == "risk" else "Ngẫu nhiên")
    s_active = oc2[1].checkbox("Chỉ khách đang hoạt động", True, key="policy_active")
    seg_by = oc2[2].selectbox("Phân tích theo", ["contract_type", "internet_service_type", "region",
                                                 "payment_method", "tenure_group"], format_func=dim_label)
    scenario = {"contract_migration_pct": s_contract, "autopay_pct": s_autopay, "security_bundle_pct": s_bundle,
                "price_change_pct": s_price, "only_active": s_active, "targeting": s_target}
    try:
        sim = call("policy", scenario, FILTERS, seg_by)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Không mô phỏng được: {exc}")
        sim = None
    if sim:
        bse, scn, d = sim["baseline"], sim["scenario"], sim["delta"]
        mc = st.columns(4)
        mc[0].metric("Khách hàng mô phỏng", f"{bse['customers']:,}")
        mc[1].metric("Xác suất churn TB", fmt_pct(scn["avg_churn_probability"]),
                     f"{d['avg_churn_probability'] * 100:+.2f} điểm %", delta_color="inverse")
        mc[2].metric("Khách rời mạng kỳ vọng", f"{scn['expected_churners']:,.0f}",
                     f"{d['expected_churners']:+,.0f}", delta_color="inverse")
        mc[3].metric("MRR kỳ vọng giữ được", fmt_money(scn["expected_retained_mrr"]),
                     f"{d['expected_retained_mrr']:+,.0f} $/tháng")
        ap = sim["applied"]
        st.caption(f"Đã áp dụng: {ap['contract_migration']:,} khách đổi hợp đồng · {ap['autopay']:,} khách auto-pay · "
                   f"{ap['security_bundle']:,} khách nhận combo · cước {ap['price_change_pct']:+.0f}%.")
        bd = sim["breakdown"]
        cc = st.columns(2)
        m = bd.melt(id_vars=sim["segment_by"], value_vars=["baseline_churn", "scenario_churn"],
                    var_name="k", value_name="v")
        m["k"] = m["k"].map({"baseline_churn": "Hiện tại", "scenario_churn": "Kịch bản"})
        fig = px.bar(m, x=sim["segment_by"], y="v", color="k", barmode="group", template=TEMPLATE, height=360,
                     color_discrete_sequence=[NEUTRAL, STAY_COLOR], title="Xác suất churn TB: hiện tại vs kịch bản",
                     labels={sim["segment_by"]: dim_label(sim["segment_by"]), "v": "", "k": ""})
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(legend=dict(orientation="h", y=-0.18), margin=dict(t=50, b=10, l=10, r=10))
        cc[0].plotly_chart(fig, use_container_width=True)
        m2 = bd.melt(id_vars=sim["segment_by"], value_vars=["baseline_mrr_risk", "scenario_mrr_risk"],
                     var_name="k", value_name="v")
        m2["k"] = m2["k"].map({"baseline_mrr_risk": "Hiện tại", "scenario_mrr_risk": "Kịch bản"})
        fig = px.bar(m2, x=sim["segment_by"], y="v", color="k", barmode="group", template=TEMPLATE, height=360,
                     color_discrete_sequence=[NEUTRAL, CHURN_COLOR], title="MRR kỳ vọng mất ($/tháng)",
                     labels={sim["segment_by"]: dim_label(sim["segment_by"]), "v": "", "k": ""})
        fig.update_layout(legend=dict(orientation="h", y=-0.18), margin=dict(t=50, b=10, l=10, r=10))
        cc[1].plotly_chart(fig, use_container_width=True)
        st.dataframe(bd.style.format({"baseline_churn": "{:.1%}", "scenario_churn": "{:.1%}", "delta_churn": "{:+.2%}",
                                      "baseline_mrr_risk": "${:,.0f}", "scenario_mrr_risk": "${:,.0f}"}),
                     use_container_width=True, hide_index=True)
        st.caption("Giảm giá làm giảm MRR trên mỗi khách; hãy so sánh 'MRR kỳ vọng giữ được' để thấy đánh đổi "
                   "giữa doanh thu và tỷ lệ churn. Kết quả là ước lượng từ mô hình, không phải hiệu quả nhân quả.")
