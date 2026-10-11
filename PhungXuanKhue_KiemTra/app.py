"""
Bước 7: Giao diện nhập thông tin người bệnh, sinh cảnh báo và giải thích kết quả.
Sinh viên: Phùng Xuân Khuê - MSV: 26ACN012
Chạy:  streamlit run app.py   (cần chạy train.py trước để có model.pkl)
"""
import os
import joblib
import pandas as pd
import streamlit as st

import train
from train import transform, NUM_COLS


@st.cache_resource
def load_bundle():
    # Khi deploy lần đầu chưa có model.pkl -> tự huấn luyện từ CSDL_TieuDuong.csv
    if not os.path.exists(train.MODEL_FILE):
        train.main()
    return joblib.load(train.MODEL_FILE)


bundle = load_bundle()
model, params, FEATURES = bundle["model"], bundle["params"], bundle["features"]

# Ngưỡng tham khảo lâm sàng (đơn vị như trong dữ liệu): (thấp, cao, tên hiển thị, đơn vị)
REF = {
    "Glucose":     (3.9, 6.9, "Glucose máu đói", "mmol/L"),
    "HbA1C":       (4.0, 6.4, "HbA1C", "%"),
    "Cholesterol": (None, 5.2, "Cholesterol toàn phần", "mmol/L"),
    "HDL":         (1.0, None, "HDL-C", "mmol/L"),
    "LDL":         (None, 3.4, "LDL-C", "mmol/L"),
    "Triglycerid": (None, 1.7, "Triglycerid", "mmol/L"),
    "Creatinine":  (44, 110, "Creatinine", "µmol/L"),
    "Ure":         (2.5, 7.5, "Ure", "mmol/L"),
    "AcidUric":    (150, 420, "Acid Uric", "µmol/L"),
    "AST":         (None, 40, "AST (GOT)", "U/L"),
    "ALT":         (None, 40, "ALT (GPT)", "U/L"),
}
DEFAULT = {"Tuoi": 45, "Glucose": 5.6, "HbA1C": 5.5, "Cholesterol": 5.0, "HDL": 1.4,
           "LDL": 3.0, "Triglycerid": 1.5, "Creatinine": 80.0, "Ure": 5.0,
           "AcidUric": 350.0, "AST": 25.0, "ALT": 22.0}

st.set_page_config(page_title="Dự đoán nguy cơ tiểu đường", page_icon="🩺", layout="wide")
st.title("🩺 Dự đoán nguy cơ bệnh tiểu đường")
st.caption(f"SV: Phùng Xuân Khuê – 26ACN012 · Mô hình: {bundle['model_name']} · "
           f"F1 test = {bundle['test_metrics']['F1']:.3f}, Recall test = {bundle['test_metrics']['Recall']:.3f}")

with st.form("form"):
    st.subheader("Thông tin người bệnh")
    cols = st.columns(4)
    inp = {}
    inp["Tuoi"] = cols[0].number_input("Tuổi", 0, 110, DEFAULT["Tuoi"])
    inp["GioiTinh"] = cols[1].selectbox("Giới tính", ["Nam", "Nữ"])
    for i, k in enumerate(REF, start=2):
        lo, hi, name, unit = REF[k]
        inp[k] = cols[i % 4].number_input(f"{name} ({unit})", min_value=0.0,
                                          value=float(DEFAULT[k]), step=0.1, format="%.2f")
    ok = st.form_submit_button("🔍 Dự đoán", type="primary")

if ok:
    raw = pd.DataFrame([inp])
    X = transform(raw, params)
    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)

    st.subheader("Kết quả")
    if prob >= 0.7:
        st.error(f"⚠️ NGUY CƠ CAO mắc tiểu đường — xác suất {prob:.1%}. "
                 "Khuyến nghị đi khám chuyên khoa Nội tiết, xét nghiệm lại Glucose đói và HbA1C.")
    elif prob >= 0.5:
        st.warning(f"⚠️ CÓ NGUY CƠ mắc tiểu đường — xác suất {prob:.1%}. "
                   "Nên tái khám, điều chỉnh chế độ ăn và vận động.")
    elif prob >= 0.3:
        st.info(f"ℹ️ Nguy cơ THẤP – TRUNG BÌNH — xác suất {prob:.1%}. Nên theo dõi định kỳ.")
    else:
        st.success(f"✅ Nguy cơ THẤP — xác suất {prob:.1%}. Duy trì lối sống lành mạnh.")
    st.progress(float(prob))

    # Giải thích 1: so sánh từng chỉ số với ngưỡng tham khảo
    st.subheader("Giải thích kết quả")
    abnormal = []
    for k, (lo, hi, name, unit) in REF.items():
        v = inp[k]
        if hi is not None and v > hi:
            abnormal.append(f"- **{name}** = {v} {unit} → **cao** (bình thường ≤ {hi})")
        elif lo is not None and v < lo:
            abnormal.append(f"- **{name}** = {v} {unit} → **thấp** (bình thường ≥ {lo})")
    if abnormal:
        st.markdown("**Các chỉ số bất thường:**\n" + "\n".join(abnormal))
    else:
        st.markdown("Tất cả chỉ số nằm trong ngưỡng tham khảo.")
    if inp["Glucose"] >= 7.0 or inp["HbA1C"] >= 6.5:
        st.markdown("> Theo tiêu chuẩn ADA/WHO: Glucose đói ≥ 7.0 mmol/L hoặc HbA1C ≥ 6.5% "
                    "là ngưỡng chẩn đoán đái tháo đường.")

    # Giải thích 2: mức đóng góp của từng chỉ số vào dự đoán của mô hình
    # (thay chỉ số đó bằng mức bình thường DEFAULT, xem xác suất thay đổi bao nhiêu)
    contrib = {}
    for k in NUM_COLS:
        alt = raw.copy()
        alt[k] = DEFAULT[k]
        contrib[k] = prob - model.predict_proba(transform(alt, params))[0, 1]
    contrib = pd.Series(contrib).sort_values(key=abs, ascending=False)
    top = contrib[contrib.abs() > 0.01].head(5)
    if len(top):
        st.markdown("**Các chỉ số ảnh hưởng nhiều nhất tới dự đoán của mô hình:**")
        for k, d in top.items():
            arrow = "tăng" if d > 0 else "giảm"
            st.markdown(f"- {k} = {inp[k]}: làm **{arrow}** nguy cơ {abs(d):.1%} "
                        f"(so với mức bình thường {DEFAULT[k]})")
    st.bar_chart(contrib.rename("Đóng góp vào xác suất"))

    st.caption("Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ.")

with st.expander("Độ quan trọng các đặc trưng (toàn bộ mô hình – Random Forest)"):
    st.bar_chart(bundle["importance"].rename("Độ quan trọng"))
