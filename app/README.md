# Phase 3 — Backend API, Streamlit Dashboard & Power BI
## Cấu trúc thư mục

```
app/
├── __init__.py
├── config.py              Đường dẫn, kết nối DB, biến môi trường
├── data_access.py         Truy vấn DW qua SQL Views + OLAP (Slice/Dice, Drill, ROLLUP, Pivot)
├── ml_service.py          Nạp model, dự báo churn, cụm, priority, gợi ý giữ chân, What-If
├── services.py            Facade dùng chung cho API, Dashboard, export
├── api.py                 FastAPI RESTful
├── client.py              ApiClient / LocalClient cho Dashboard
├── main.py                Streamlit Dashboard (6 tab)
├── export_scores.py       Ghi điểm ML + lưới What-If cho Power BI
├── requirements.txt       Thư viện bổ sung của Phase 3
├── sql/
│   ├── 04_dashboard_views.sql     View KPI, ROLLUP địa lý, CUBE hợp đồng × dịch vụ × thanh toán
│   └── 05_ml_customer_scores.sql  Bảng edw.ml_customer_scores + view edw.vw_bi_churn_scored
└── powerbi/
    ├── README.md          Hướng dẫn dựng report Power BI từng bước
    ├── measures.dax       Các measure DAX
    └── data/              CSV do export_scores.py sinh ra
```

## Chạy

Mọi lệnh chạy từ **thư mục gốc dự án**:

```bash
pip install -r requirements.txt -r app/requirements.txt

# Terminal 1 — Backend API (Swagger: http://localhost:8000/docs)
uvicorn app.api:app --reload --port 8000

# Terminal 2 — Streamlit Dashboard (http://localhost:8501)
streamlit run app/main.py

# Power BI — xuất điểm ML (xem app/powerbi/README.md)
python -m app.export_scores --to-db
```

## Dashboard Streamlit

| Tab | Nội dung |
| :--- | :--- |
| Tổng quan | KPI (churn %, MRR, MRR mất do churn), churn theo hợp đồng / thâm niên / Internet / thanh toán, bản đồ GIS |
| OLAP Explorer | Drill-down / Roll-up theo 4 phân cấp, Pivot 2 chiều, bảng `ROLLUP` có tổng con |
| Phân khúc | Hồ sơ 4 cụm K-Means và chiến lược theo cụm |
| Ưu tiên giữ chân | Top-K theo priority score (Model 3), tải CSV chiến dịch |
| What-If khách hàng | Dự báo real-time, Baseline ↔ Kịch bản, gợi ý giữ chân có Δ churn |
| Mô phỏng chính sách | Hợp đồng, auto-pay, combo add-on, giá cước → số khách rời mạng & MRR kỳ vọng |

Bộ lọc Slice/Dice ở sidebar áp dụng cho các tab, trừ What-If khách hàng.

## Endpoint chính

| Method | Path | Mô tả |
| :--- | :--- | :--- |
| GET | `/health` | Trạng thái, nguồn dữ liệu, thông tin model |
| GET | `/api/v1/meta` | Chiều, phân cấp, độ đo |
| POST | `/api/v1/olap/kpis` | KPI sau Slice/Dice |
| POST | `/api/v1/olap/aggregate` | Gộp theo danh sách chiều |
| POST | `/api/v1/olap/drill` | Drill-down theo phân cấp |
| POST | `/api/v1/olap/rollup` | `GROUP BY ROLLUP` có tổng con |
| POST | `/api/v1/olap/pivot` | Bảng chéo 2 chiều |
| POST | `/api/v1/predict` | Churn %, risk tier, cụm, priority, gợi ý giữ chân |
| POST | `/api/v1/whatif/compare` | Baseline ↔ kịch bản cho 1 khách hàng |
| POST | `/api/v1/whatif/policy` | Mô phỏng chính sách |
| POST | `/api/v1/retention/top` | Top-K ưu tiên giữ chân |
| POST | `/api/v1/segments/profile` | Hồ sơ 4 cụm K-Means |

```bash
curl -X POST localhost:8000/api/v1/olap/drill -H "Content-Type: application/json" \
  -d '{"hierarchy":"Địa lý (Geography)","path":["United States","California","Bay Area"],
       "filters":{"contract_type":["Month-to-month"]}}'

curl -X POST localhost:8000/api/v1/predict -H "Content-Type: application/json" \
  -d '{"tenure_months":2,"monthly_charges":85.5,"contract_type":"Month-to-month",
       "internet_service_type":"Fiber optic","payment_method":"Electronic check"}'
```