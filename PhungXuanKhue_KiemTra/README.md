# BÀI KIỂM TRA – MÔN TRÍ TUỆ NHÂN TẠO

**Sinh viên:** Phùng Xuân Khuê  **MSV:** 26ACN012
**Đề tài:** Xây dựng phần mềm AI (Machine Learning) dự đoán nguy cơ bệnh tiểu đường từ thông số sức khỏe.

## 1. Cấu trúc thư mục

```
PhungXuanKhue_KiemTra/
├── CSDL_TieuDuong.csv     # dữ liệu gốc (chép vào đây)
├── train.py               # Bước 1 → 6: tiền xử lý, chia dữ liệu, huấn luyện, đánh giá
├── app.py                 # Bước 7: giao diện nhập thông tin, cảnh báo, giải thích
├── requirements.txt       # thư viện cần cài
├── README.md              # file này: giải thích + hướng dẫn
├── model.pkl              # (sinh ra sau khi chạy train.py) mô hình tốt nhất
└── ket_qua_danh_gia.txt   # (sinh ra sau khi chạy train.py) báo cáo kết quả
```

## 2. Hướng dẫn sử dụng

Yêu cầu: Python 3.9 trở lên.

```bash
# 1) Vào thư mục bài làm
cd PhungXuanKhue_KiemTra

# 2) Cài thư viện (chỉ làm 1 lần)
pip install -r requirements.txt

# 3) Đặt file CSDL_TieuDuong.csv vào cùng thư mục, rồi huấn luyện mô hình
python train.py

# 4) Mở giao diện dự đoán (trình duyệt tự mở http://localhost:8501)
streamlit run app.py
```

Trên giao diện: nhập tuổi, giới tính và 11 chỉ số xét nghiệm → bấm **Dự đoán**.
Phần mềm hiển thị:
- **Mức cảnh báo** theo xác suất: thấp (< 30%), thấp–trung bình (30–50%), có nguy cơ (50–70%), nguy cơ cao (≥ 70%), kèm khuyến nghị.
- **Giải thích:** các chỉ số vượt ngưỡng tham khảo, và các chỉ số ảnh hưởng nhiều nhất tới dự đoán của mô hình (biểu đồ đóng góp).

### Deploy lên Streamlit Community Cloud (miễn phí)
1. Đẩy thư mục này (kèm `CSDL_TieuDuong.csv`) lên GitHub.
2. Vào https://share.streamlit.io → đăng nhập bằng GitHub → **Create app**.
3. Chọn repo, branch `main`, *Main file path*: `PhungXuanKhue_KiemTra/app.py` → **Deploy**.
4. Lần chạy đầu app tự huấn luyện mô hình (khoảng vài chục giây), sau đó dùng bình thường.

## 3. Giải thích các bước thực hiện

### Bước 1 – Xác định bài toán
Bài toán **phân loại nhị phân** (học có giám sát):
- Đầu vào: Tuổi, Giới tính, Glucose, HbA1C, Cholesterol, HDL, LDL, Triglycerid, Creatinine, Ure, Acid Uric, AST, ALT.
- Đầu ra: `NguyCoTieuDuong` = 1 (có nguy cơ) / 0 (không có nguy cơ).

### Bước 2 – Thu thập dữ liệu
Dùng file `CSDL_TieuDuong.csv` (dữ liệu y tế thực tế, 14 cột).

### Bước 3 – Tiền xử lý dữ liệu

**3.1. Làm sạch**
| Vấn đề phát hiện | Ví dụ trong dữ liệu | Cách xử lý |
|---|---|---|
| Không đồng bộ giới tính | `Nam`, `Male`, `Nữ`, `Female` | Quy về `Nam` / `Nữ` |
| Không đồng bộ nhãn | `1`, `0`, `TRUE`, `FALSE`, `Có nguy cơ`, `Không có nguy cơ` | Quy về 1 / 0 |
| Dữ liệu lỗi (sai dấu thập phân, nhập nhầm) | HbA1C = 8025, Glucose = 750, 59, 61; Cholesterol = 55 | Giá trị ngoài khoảng sinh học hợp lệ → coi là thiếu |
| Dữ liệu thiếu | ô trống ở Tuổi, Giới tính, Glucose, HbA1C, … | Cột số: điền **trung vị**; Giới tính: điền **giá trị xuất hiện nhiều nhất** |
| Dòng trùng lặp | | Xóa |
| Ngoại lai | Creatinine 1500, AST 534 … | Cắt (clip) về khoảng phân vị **1% – 99%** |

Trung vị và ngưỡng ngoại lai chỉ được tính trên **tập train** rồi áp dụng cho val/test, tránh rò rỉ dữ liệu.

**3.2. Biến đổi**
- Mã hóa: Giới tính Nam = 1, Nữ = 0.
- Chuẩn hóa: `StandardScaler` (đưa về trung bình 0, độ lệch chuẩn 1) – cần cho Logistic Regression và SVM.

**3.3. Phân chia** (có `stratify` để giữ tỷ lệ nhãn, `random_state=42`)
- 70% Train – 15% Validation – 15% Test.

### Bước 4 – Chọn mô hình
1. **Logistic Regression** – mô hình tuyến tính, dễ giải thích.
2. **Random Forest** (300 cây) – phi tuyến, cho biết độ quan trọng đặc trưng.
3. **SVM** (kernel RBF) – tốt với ranh giới phân lớp phức tạp.

Cả 3 đều dùng `class_weight="balanced"` vì số người có nguy cơ ít hơn số người không có nguy cơ.

### Bước 5 – Huấn luyện
Mỗi mô hình đặt trong `Pipeline(StandardScaler → mô hình)` và huấn luyện trên tập train.

### Bước 6 – Đánh giá
Các độ đo: **Accuracy, Precision, Recall, F1-score, ROC-AUC**, ma trận nhầm lẫn.
- Chọn mô hình tốt nhất theo **F1 trên tập validation**, rồi báo cáo kết quả cuối trên **tập test**.
- Với bài toán y tế, **Recall** rất quan trọng (không bỏ sót người bệnh); F1 cân bằng giữa Recall và Precision.
- Kết quả chi tiết được in ra màn hình và lưu ở `ket_qua_danh_gia.txt`.

### Bước 7 – Giao diện, cảnh báo và giải thích
`app.py` (Streamlit):
- Form nhập thông tin người bệnh.
- Dự đoán xác suất nguy cơ → sinh cảnh báo 4 mức kèm khuyến nghị.
- Giải thích:
  - So sánh từng chỉ số với ngưỡng tham khảo lâm sàng (vd. Glucose đói ≥ 7.0 mmol/L hoặc HbA1C ≥ 6.5% là ngưỡng chẩn đoán đái tháo đường theo ADA/WHO).
  - Mức đóng góp của từng chỉ số: thay chỉ số đó bằng mức bình thường và xem xác suất thay đổi bao nhiêu.
  - Độ quan trọng đặc trưng tổng quát (Random Forest).

## 4. Lưu ý
- Trong dữ liệu, nhiều ô có giá trị lặp lại giống nhau (HbA1C = 8.025, LDL = 3.45, AcidUric = 426.2, …), có thể là giá trị đã được điền sẵn cho ô thiếu từ trước. Vì vậy mô hình dựa chủ yếu vào **Glucose** và ít dùng HbA1C/LDL.
- Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ.
