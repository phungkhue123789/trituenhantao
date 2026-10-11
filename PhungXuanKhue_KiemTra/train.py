"""
Bài kiểm tra môn Trí tuệ nhân tạo
Sinh viên: Phùng Xuân Khuê - MSV: 26ACN012
Đề tài: Dự đoán nguy cơ bệnh tiểu đường từ thông số sức khỏe

File này thực hiện Bước 1 -> Bước 6: đọc dữ liệu, tiền xử lý, chia dữ liệu,
huấn luyện 3 mô hình, đánh giá và lưu mô hình tốt nhất ra file model.pkl.
Chạy:  python train.py
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE, "CSDL_TieuDuong.csv")
MODEL_FILE = os.path.join(BASE, "model.pkl")
REPORT_FILE = os.path.join(BASE, "ket_qua_danh_gia.txt")

TARGET = "NguyCoTieuDuong"
NUM_COLS = ["Tuoi", "Glucose", "HbA1C", "Cholesterol", "HDL", "LDL",
            "Triglycerid", "Creatinine", "Ure", "AcidUric", "AST", "ALT"]
FEATURES = ["Tuoi", "GioiTinh"] + NUM_COLS[1:]

# Khoảng giá trị hợp lệ về mặt sinh học; ngoài khoảng này coi là dữ liệu lỗi
# (ví dụ HbA1C = 8025, Glucose = 750, Cholesterol = 55 do nhập sai dấu phẩy)
VALID_RANGE = {
    "Tuoi": (0, 110), "Glucose": (1, 40), "HbA1C": (3, 20),
    "Cholesterol": (1, 30), "HDL": (0.1, 5), "LDL": (0.1, 15),
    "Triglycerid": (0.1, 80), "Creatinine": (5, 2000), "Ure": (0.5, 80),
    "AcidUric": (50, 1500), "AST": (1, 2000), "ALT": (1, 2000),
}

log_lines = []


def log(s=""):
    print(s)
    log_lines.append(str(s))


# ---------------------------------------------------------------- Bước 2
def load_data():
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig", dtype=str)
    log(f"Đọc dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")
    return df


# ---------------------------------------------------------------- Bước 3.1
def clean_data(df):
    df = df.copy()

    # Đồng bộ dữ liệu: giới tính và nhãn có nhiều cách ghi khác nhau
    gender_map = {"nam": "Nam", "male": "Nam", "nữ": "Nữ", "nu": "Nữ", "female": "Nữ"}
    df["GioiTinh"] = df["GioiTinh"].str.strip().str.lower().map(gender_map)

    label_map = {"1": 1, "true": 1, "có nguy cơ": 1,
                 "0": 0, "false": 0, "không có nguy cơ": 0}
    df[TARGET] = df[TARGET].str.strip().str.lower().map(label_map)
    log(f"Đồng bộ nhãn: {df[TARGET].isna().sum()} nhãn không nhận dạng được -> loại bỏ")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    # Chuyển cột số; giá trị không phải số -> NaN
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Dữ liệu lỗi: ngoài khoảng hợp lệ -> NaN (sẽ được điền ở bước sau)
    log("Dữ liệu lỗi (ngoài khoảng hợp lệ) -> đặt là thiếu:")
    for c, (lo, hi) in VALID_RANGE.items():
        bad = (df[c] < lo) | (df[c] > hi)
        if bad.sum():
            log(f"  {c}: {bad.sum()} giá trị, ví dụ {df.loc[bad, c].head(3).tolist()}")
        df.loc[bad, c] = np.nan

    # Dữ liệu trùng lặp
    n = len(df)
    df = df.drop_duplicates()
    log(f"Xóa {n - len(df)} dòng trùng lặp")

    # Dữ liệu thiếu
    log("Số giá trị thiếu mỗi cột:")
    miss = df.isna().sum()
    log("  " + ", ".join(f"{k}={v}" for k, v in miss.items() if v > 0))
    return df.reset_index(drop=True)


def fit_imputer_and_clipper(train):
    """Tính trung vị (điền thiếu) và ngưỡng ngoại lai (P1-P99) CHỈ trên tập train."""
    medians = train[NUM_COLS].median()
    gender_mode = train["GioiTinh"].mode()[0]
    lower = train[NUM_COLS].quantile(0.01)
    upper = train[NUM_COLS].quantile(0.99)
    return {"medians": medians, "gender_mode": gender_mode,
            "lower": lower, "upper": upper}


def transform(df, params):
    df = df.copy()
    df[NUM_COLS] = df[NUM_COLS].fillna(params["medians"])          # điền thiếu
    df["GioiTinh"] = df["GioiTinh"].fillna(params["gender_mode"])
    df[NUM_COLS] = df[NUM_COLS].clip(params["lower"], params["upper"], axis=1)  # ngoại lai
    df["GioiTinh"] = (df["GioiTinh"] == "Nam").astype(int)         # mã hóa: Nam=1, Nữ=0
    return df[FEATURES]


# ---------------------------------------------------------------- Bước 6
def evaluate(name, model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "Mô hình": name,
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, prob),
    }


def main():
    log("=" * 60)
    log("BƯỚC 1: Bài toán phân loại nhị phân - dự đoán NguyCoTieuDuong (0/1)")
    log("=" * 60)
    log("BƯỚC 2: Thu thập dữ liệu")
    df = load_data()

    log("=" * 60)
    log("BƯỚC 3.1: Làm sạch dữ liệu")
    df = clean_data(df)
    log(f"Sau làm sạch: {len(df)} dòng. Tỷ lệ nhãn: {df[TARGET].value_counts().to_dict()}")

    log("=" * 60)
    log("BƯỚC 3.3: Chia dữ liệu 70% train / 15% validation / 15% test (stratify)")
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    log(f"Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_te)}")

    log("=" * 60)
    log("BƯỚC 3.2: Điền thiếu (trung vị), xử lý ngoại lai (cắt P1-P99), "
        "mã hóa giới tính, chuẩn hóa StandardScaler")
    params = fit_imputer_and_clipper(X_tr)
    X_tr, X_val, X_te = transform(X_tr, params), transform(X_val, params), transform(X_te, params)

    log("=" * 60)
    log("BƯỚC 4-5: Huấn luyện các mô hình")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        # bọc CalibratedClassifierCV để SVM trả về xác suất (predict_proba)
        "SVM": CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced", random_state=42),
                                      ensemble=False),
    }
    fitted, val_rows = {}, []
    for name, clf in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        val_rows.append(evaluate(name, pipe, X_val, y_val))

    log("=" * 60)
    log("BƯỚC 6: Đánh giá trên tập VALIDATION")
    val_df = pd.DataFrame(val_rows).set_index("Mô hình").round(4)
    log(val_df.to_string())

    # Bài toán y tế: ưu tiên không bỏ sót người bệnh -> chọn theo F1 (cân bằng Recall & Precision)
    best_name = val_df["F1"].idxmax()
    best = fitted[best_name]
    log(f"\n=> Mô hình tốt nhất theo F1 trên validation: {best_name}")

    log("\nĐánh giá TẤT CẢ mô hình trên tập TEST:")
    test_df = pd.DataFrame([evaluate(n, m, X_te, y_te) for n, m in fitted.items()]).set_index("Mô hình").round(4)
    log(test_df.to_string())

    pred = best.predict(X_te)
    log(f"\nChi tiết mô hình {best_name} trên TEST:")
    log(classification_report(y_te, pred, target_names=["Không nguy cơ", "Có nguy cơ"], digits=4))
    cm = confusion_matrix(y_te, pred)
    log(f"Ma trận nhầm lẫn [[TN, FP], [FN, TP]]: {cm.tolist()}")

    # Độ quan trọng đặc trưng (dùng Random Forest để giải thích tổng quát)
    rf = fitted["Random Forest"].named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    log("\nĐộ quan trọng các đặc trưng (Random Forest):")
    log(imp.round(4).to_string())

    joblib.dump({"model": best, "model_name": best_name, "params": params,
                 "features": FEATURES, "importance": imp,
                 "test_metrics": test_df.loc[best_name].to_dict()}, MODEL_FILE)
    log(f"\nĐã lưu mô hình vào {MODEL_FILE}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"Đã lưu báo cáo vào {REPORT_FILE}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # in tiếng Việt trên console Windows
    main()
