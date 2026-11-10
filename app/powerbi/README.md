# Tích hợp Power BI

Power BI và Streamlit cùng đọc một Kho dữ liệu. Chia việc như sau:

| Công cụ | Phần phụ trách |
| :--- | :--- |
| **Power BI** | Báo cáo OLAP: Drill-down/Roll-up theo hierarchy, Slice/Dice bằng slicer, bản đồ, KPI, phân khúc, danh sách ưu tiên, What-If chính sách (tra lưới đã tính sẵn bằng model) |
| **Streamlit** (`app/main.py`) | Dự báo real-time cho từng khách hàng, so sánh Baseline ↔ Kịch bản, gợi ý giữ chân, mô phỏng chính sách tùy biến nhiều đòn bẩy cùng lúc |

Power BI không chạy được `churn_model.pkl` mỗi khi người dùng tương tác. Vì vậy `app/export_scores.py` chạy model một lần rồi ghi kết quả thành bảng để Power BI đọc.

## Bước 1 — Chuẩn bị dữ liệu

Từ thư mục gốc dự án:

```bash
# Có PostgreSQL (khuyến nghị): ghi điểm ML vào DW, tạo view edw.vw_bi_churn_scored
python -m app.export_scores --to-db

# Không có PostgreSQL: chỉ ghi CSV vào app/powerbi/data/
python -m app.export_scores
```

Các bảng được tạo:

| Bảng | Dòng | Nội dung |
| :--- | ---: | :--- |
| `edw.ml_customer_scores` / `ml_customer_scores.csv` | 7,043 | churn_probability, risk_tier, cluster_name, priority_score, priority_rank, expected_mrr_at_risk |
| `edw.vw_bi_churn_scored` (chỉ có ở PostgreSQL) | 7,043 | `vw_bi_churn_analytics` + các cột điểm ML |
| `edw.ml_whatif_policy_grid` / `whatif_policy_grid.csv` | 33 | 3 đòn bẩy × 11 mức (0–100%): churn TB, số khách rời mạng kỳ vọng, MRR giữ được |
| `edw.ml_whatif_policy_segment` / `whatif_policy_segment.csv` | 99 | Lưới trên chia theo loại hợp đồng |

Chạy lại script mỗi khi ETL nạp dữ liệu mới hoặc model được huấn luyện lại, sau đó bấm Refresh trong Power BI.

## Bước 2 — Nạp dữ liệu vào Power BI Desktop

**Cách A — PostgreSQL:** Get Data → PostgreSQL database → Server `localhost`, Database `telecom_dw` → chọn chế độ **Import** → tick:
- `edw.vw_bi_churn_scored` → đổi tên bảng thành **Churn**
- `edw.ml_whatif_policy_grid` → **WhatIfGrid**
- `edw.ml_whatif_policy_segment` → **WhatIfSeg**

**Cách B — CSV:** Get Data → Text/CSV, nạp:
- `data/processed/telecom_bi_churn_analytics.csv` → **Churn**
- `app/powerbi/data/ml_customer_scores.csv` → **Scores**
- `app/powerbi/data/whatif_policy_grid.csv` → **WhatIfGrid**
- `app/powerbi/data/whatif_policy_segment.csv` → **WhatIfSeg**

Với cách B, gộp điểm vào bảng Churn để dùng chung bộ DAX: Power Query → bảng Churn → *Merge Queries* với Scores theo `customer_id` (Left Outer) → mở rộng các cột `churn_probability, churn_prediction, risk_tier, cluster_id, cluster_name, priority_score, priority_rank, expected_mrr_at_risk`.

**Kiểu dữ liệu cần kiểm tra trong Power Query:**
- `zip_code` → Text (tránh Power BI cộng dồn mã bưu chính)
- `latitude`, `longitude` → Decimal; trong Column tools đặt *Data category* là Latitude / Longitude
- `churn_flag` → Whole Number; các cột `has_*`, `is_*` → True/False

## Bước 3 — Tạo Hierarchy (Drill-down / Roll-up)

Trong pane Data, chuột phải cột → *Create hierarchy*, rồi kéo các cột còn lại vào:

| Hierarchy | Các cấp |
| :--- | :--- |
| Geography | country → state → region → city → zip_code |
| Contract | contract_type → payment_category → payment_method |
| Service | internet_service_type → total_active_addons → has_tech_support |
| Customer | tenure_group → is_senior_citizen → has_dependents |

Đặt hierarchy vào trục X của biểu đồ cột. Mũi tên ↓ (drill-down vào phần tử được click), ⇊ (xuống cấp kế tiếp cho tất cả) và ↑ (roll-up) là các thao tác OLAP có sẵn của Power BI.

Sắp xếp `tenure_group` đúng thứ tự: tạo cột `tenure_order = SWITCH(Churn[tenure_group], "0-12 Months",1, "13-24 Months",2, "25-48 Months",3, 4)` rồi *Sort by column*.

## Bước 4 — Measures

Mở `measures.dax`, dán từng measure qua *Modeling → New measure*. Các measure dùng tên bảng Churn / WhatIfGrid / WhatIfSeg như trên.

## Bước 5 — What-If Parameters

*Modeling → New parameter → Numeric range*:

| Tên | Min | Max | Bước | Mặc định | Dùng cho |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `Policy Pct` | 0 | 100 | **10** | 30 | Tra lưới WhatIfGrid (bước phải là 10 để khớp lưới) |
| `Top K` | 10 | 1000 | 10 | 100 | Lọc danh sách ưu tiên |
| `Price Change Pct` | -30 | 30 | 1 | 0 | Tác động giá lên doanh thu |

Power BI tự sinh measure `[Policy Pct Value]`, `[Top K Value]`, `[Price Change Pct Value]` mà `measures.dax` đang dùng. Thêm slicer một lựa chọn trên `WhatIfGrid[lever_label]` để chọn đòn bẩy.

## Bước 6 — Gợi ý bố cục report

| Trang | Visual |
| :--- | :--- |
| **1. Tổng quan** | Card: Customers, Churn Rate, MRR, Revenue at Risk, Avg Tenure · Cột chồng churn theo contract_type, tenure_group · Map (latitude/longitude, legend churn_label) · Slicer: region, contract_type, internet_service_type, payment_category, tenure_group |
| **2. OLAP Explorer** | Cột: trục = hierarchy Geography, giá trị = Churn Rate, tooltip = Customers, Revenue at Risk · Matrix: hàng = hierarchy Contract, cột = internet_service_type, giá trị = Churn Rate (bật Subtotals để có roll-up) · Decomposition tree: Revenue at Risk phân rã theo region → contract_type → internet_service_type |
| **3. Phân khúc** | Cột nhóm theo cluster_name: Churn Rate và Avg Churn Probability · Scatter: tenure_months × monthly_charges, legend cluster_name · Bảng hồ sơ cụm |
| **4. Ưu tiên giữ chân** | Table: customer_id, priority_score, churn_probability, risk_tier, cluster_name, contract_type, monthly_charges, sắp theo priority_rank · Visual filter: churn_flag = 0 và `In Top K` = 1 · Slicer Top K, risk_tier |
| **5. What-If chính sách** | Slicer lever_label + slider Policy Pct · Card: WI Avg Churn, WI Delta Churn (pp), WI Customers Saved, WI Retained MRR Gain · Đường: trục = WhatIfGrid[pct], giá trị = avg_churn_probability, legend = lever_label · Cột nhóm: trục = WhatIfSeg[contract_type], giá trị = WI Seg Baseline Churn, WI Seg Scenario Churn |

Trang 4 và 5 nên dùng *Edit interactions* để slicer địa lý ở trang 1 không lọc WhatIfGrid (lưới được tính trên toàn bộ khách đang hoạt động).

## Lưu ý khi trình bày

- Lưới What-If chọn khách hàng có rủi ro cao nhất để áp dụng trước (chiến dịch có mục tiêu), chỉ trên khách đang hoạt động (`churn_flag = 0`).
- Kết quả What-If là thay đổi rủi ro dự báo của mô hình tương quan, không phải hiệu quả nhân quả của ưu đãi.
- `churn_probability` trong `ml_customer_scores` giống với `models/evaluation/churn_priority_scores.csv` của Model 3 (chênh lệch < 1e-13). Số liệu Precision@K / Lift@K chuẩn (out-of-fold) xem `models/evaluation/churn_priority_metrics.csv`.
