import math
import shap
import numpy as np
from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

# ---------------------------
# APP CONFIG
# ---------------------------

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lightGBM_model.pkl")

# ── Credit limit config ───────────────────────────────────────────────────────
LGD         = 0.45   # Loss Given Default — Basel standard
BASE_RATE   = 0.10   # Lãi suất cơ bản 10%/năm
MAX_RATE    = 0.35   # Trần lãi suất 35%/năm (quy định)
MAX_DTI     = 0.35   # DTI tối đa 35% (annuity / income)
INCOME_MULT = 5      # Hạn mức trần = thu nhập năm × 5

# ── Break-even PD ─────────────────────────────────────────────────────────────
PD_BREAKEVEN = MAX_RATE / (MAX_RATE + LGD)   # ≈ 43.75%

# ---------------------------
# FEATURE LIST & LABELS
# ---------------------------

FEATURES = [
    "EXT_SOURCE_MEAN","EXT_SOURCE_3","EXT_SOURCE_2","EXT_SOURCE_1",
    "YEARS_EMPLOYED","ANNUITY LENGTH","CREDIT_TO_GOODS_RATIO",
    "EMPLOYMENT_STABILITY","AGE","LOG_AMT_ANNUITY","LOG_AMT_GOODS_PRICE",
    "CONVERTED_DAYS_ID_PUBLISH","CONVERTED_DAYS_LAST_PHONE_CHANGE",
    "LOG_AMT_CREDIT","ANNUITY_BY_INCOME","CONVERTED_DAYS_REGISTRATION",
    "CREDIT_BY_INCOME","GOODS_PRICE_BY_INCOME",
    "REGION_POPULATION_RELATIVE","INCOME_PER_PERSON"
]

# Tên hiển thị tiếng Việt cho từng feature
FEATURE_LABELS = {
    "EXT_SOURCE_MEAN":                  "Điểm tín dụng tổng hợp",
    "EXT_SOURCE_3":                     "Điểm tín dụng nguồn 3",
    "EXT_SOURCE_2":                     "Điểm tín dụng nguồn 2",
    "EXT_SOURCE_1":                     "Điểm tín dụng nguồn 1",
    "YEARS_EMPLOYED":                   "Số năm làm việc",
    "ANNUITY LENGTH":                   "Thời hạn vay (năm)",
    "CREDIT_TO_GOODS_RATIO":            "Tỷ lệ vay / giá trị hàng",
    "EMPLOYMENT_STABILITY":             "Độ ổn định việc làm",
    "AGE":                              "Tuổi",
    "LOG_AMT_ANNUITY":                  "Khoản trả góp tháng (log)",
    "LOG_AMT_GOODS_PRICE":              "Giá trị hàng hóa (log)",
    "CONVERTED_DAYS_ID_PUBLISH":        "Năm kể từ cấp CCCD",
    "CONVERTED_DAYS_LAST_PHONE_CHANGE": "Năm kể từ đổi số ĐT",
    "LOG_AMT_CREDIT":                   "Số tiền vay (log)",
    "ANNUITY_BY_INCOME":                "Tỷ lệ trả góp / thu nhập",
    "CONVERTED_DAYS_REGISTRATION":      "Năm kể từ đăng ký thường trú",
    "CREDIT_BY_INCOME":                 "Tỷ lệ vay / thu nhập",
    "GOODS_PRICE_BY_INCOME":            "Tỷ lệ giá hàng / thu nhập",
    "REGION_POPULATION_RELATIVE":       "Mật độ dân số khu vực",
    "INCOME_PER_PERSON":                "Thu nhập bình quân đầu người",
}

# ---------------------------
# LOAD MODEL
# ---------------------------

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print("Error loading model:", e)
        return None

model = load_model()

# Khởi tạo SHAP explainer một lần duy nhất để tránh overhead
_shap_explainer = None

def _extract_booster(pipeline_or_model):
    """
    Nếu model là sklearn Pipeline, lấy step cuối cùng (LightGBM / XGBoost...).
    Nếu là LGBMClassifier, lấy booster_ bên trong.
    Nếu đã là booster / model thuần, trả thẳng về.
    """
    from sklearn.pipeline import Pipeline
    m = pipeline_or_model
    # Unwrap Pipeline — lấy step cuối
    if isinstance(m, Pipeline):
        m = m.steps[-1][1]
        print(f"Pipeline detected → extracted final step: {type(m).__name__}")
    # LGBMClassifier → lấy booster_
    if hasattr(m, "booster_"):
        print("LGBMClassifier → using .booster_")
        return m.booster_
    # Đã là lightgbm.Booster hoặc model thuần
    return m


def get_explainer():
    global _shap_explainer
    if _shap_explainer is None and model is not None:
        try:
            booster = _extract_booster(model)
            _shap_explainer = shap.TreeExplainer(booster)
            print("SHAP explainer initialized successfully")
        except Exception as e:
            print("SHAP init error:", e)
    return _shap_explainer


# ---------------------------
# SHAP EXPLANATION
# ---------------------------

def compute_shap_explain(df: pd.DataFrame, feature_values: list) -> dict | None:
    """
    Tính SHAP values cho một mẫu, trả về dict gồm:
    - risk_factors: top 5 feature đẩy PD lên
    - strengths:    top 3 feature kéo PD xuống
    - base_value:   xác suất vỡ nợ trung bình của tập huấn luyện (%)
    - max_abs:      SHAP tuyệt đối lớn nhất (để normalize thanh bar)
    """
    explainer = get_explainer()
    if explainer is None:
        return None

    try:
        # Nếu model là Pipeline, transform data qua các bước trước classifier
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline) and len(model.steps) > 1:
            import numpy as np
            arr = df.values.astype(float)
            for _, step in model.steps[:-1]:
                try:
                    arr = step.transform(arr)
                except Exception:
                    pass  # bước không transform được thì bỏ qua
            shap_input = pd.DataFrame(arr, columns=df.columns[:arr.shape[1]] if arr.shape[1] == len(df.columns) else range(arr.shape[1]))
        else:
            shap_input = df

        shap_values    = explainer.shap_values(shap_input)
        expected_value = explainer.expected_value

        # LightGBM binary: shap_values là list [class0, class1]
        if isinstance(shap_values, list) and len(shap_values) > 1:
            sv       = shap_values[1][0].tolist()
            base_val = float(expected_value[1]) if hasattr(expected_value, '__len__') else float(expected_value)
        else:
            sv_raw   = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
            sv       = sv_raw.tolist() if hasattr(sv_raw, 'tolist') else list(sv_raw)
            base_val = float(expected_value[0]) if hasattr(expected_value, '__len__') else float(expected_value)

        items = []
        for i, (feat, val) in enumerate(zip(FEATURES, sv)):
            items.append({
                "feature":     feat,
                "label":       FEATURE_LABELS.get(feat, feat),
                "shap_val":    round(val, 5),
                "feature_val": round(float(feature_values[i]), 4),
                "pct":         round(abs(val) * 100, 2),
            })

        max_abs = max(abs(x["shap_val"]) for x in items) if items else 1.0

        risk_factors = sorted(
            [x for x in items if x["shap_val"] > 0],
            key=lambda x: x["shap_val"], reverse=True
        )[:5]

        strengths = sorted(
            [x for x in items if x["shap_val"] < 0],
            key=lambda x: x["shap_val"]        # most negative first
        )[:3]

        return {
            "risk_factors": risk_factors,
            "strengths":    strengths,
            "base_value":   round(base_val * 100, 2),
            "max_abs":      max_abs,
        }

    except Exception as e:
        print("SHAP compute error:", e)
        return None


# ---------------------------
# CREDIT TERMS CALCULATOR
# ---------------------------

def calc_credit_terms(prob: float, income_per_person: float, annuity_by_income: float,
                      log_amt_credit: float = None) -> dict:
    prob = float(prob)

    if prob >= 1.0:
        r_total = MAX_RATE
    else:
        r_risk  = (prob * LGD) / (1.0 - prob)
        r_total = min(BASE_RATE + r_risk, MAX_RATE)

    ev_raw    = r_total * (1.0 - prob) - LGD * prob
    ev_factor = max(0.0, min(1.0, ev_raw / BASE_RATE))

    aff_ratio     = float(annuity_by_income) if annuity_by_income and float(annuity_by_income) > 0 else MAX_DTI
    affordability = min(1.0, MAX_DTI / aff_ratio)

    base_limit = float(income_per_person) * INCOME_MULT * affordability
    if prob < 0.35:
        credit_limit = base_limit
    elif prob < PD_BREAKEVEN:
        credit_limit = base_limit * ev_factor
    else:
        credit_limit = 0.0

    if prob < 0.20:
        tier = "Ưu tiên"
    elif prob < 0.35:
        tier = "Tiêu chuẩn"
    elif prob < PD_BREAKEVEN:
        tier = "Thận trọng"
    else:
        tier = "Từ chối"

    requested_amount = round(math.exp(float(log_amt_credit)), 0) if log_amt_credit else None
    if requested_amount and credit_limit > 0:
        if requested_amount <= credit_limit:
            counter_offer = None
            offer_status  = "full"
        else:
            counter_offer = round(credit_limit, 0)
            offer_status  = "counter"
    elif credit_limit == 0:
        counter_offer = None
        offer_status  = "reject"
    else:
        counter_offer = None
        offer_status  = "full"

    actual_loan       = counter_offer if offer_status == "counter" else (requested_amount or credit_limit)
    expected_interest = actual_loan * r_total * (1.0 - prob) if actual_loan else 0
    expected_loss     = actual_loan * LGD * prob             if actual_loan else 0
    net_ev            = expected_interest - expected_loss

    return {
        "interest_rate":     round(r_total * 100, 2),
        "credit_limit":      round(credit_limit, 0),
        "credit_tier":       tier,
        "risk_spread":       round((r_total - BASE_RATE) * 100, 2),
        "ev_factor":         round(ev_factor * 100, 1),
        "requested_amount":  requested_amount,
        "counter_offer":     counter_offer,
        "offer_status":      offer_status,
        "expected_interest": round(expected_interest, 0),
        "expected_loss":     round(expected_loss, 0),
        "net_ev":            round(net_ev, 0),
        "pd_breakeven":      round(PD_BREAKEVEN * 100, 2),
    }


# ---------------------------
# HOME PAGE
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)


# ---------------------------
# MANUAL PREDICTION
# ---------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = []
        for f in FEATURES:
            val = request.form.get(f)
            if val is None or val == "":
                return f"Missing value for {f}"
            values.append(float(val))

        df   = pd.DataFrame([values], columns=FEATURES)
        prob = model.predict_proba(df)[0][1]

        try:
            threshold = float(request.form.get("threshold") or PD_BREAKEVEN)
            threshold = max(0.05, min(0.95, threshold))
        except (ValueError, TypeError):
            threshold = PD_BREAKEVEN

        decision = "Reject" if prob >= threshold else "Approve"

        if prob < 0.20:
            risk = "Low"
        elif prob < 0.35:
            risk = "Medium"
        else:
            risk = "High"

        income        = float(request.form.get("INCOME_PER_PERSON", 0) or 0)
        annuity_ratio = float(request.form.get("ANNUITY_BY_INCOME", MAX_DTI) or MAX_DTI)
        log_credit    = request.form.get("LOG_AMT_CREDIT")
        credit_terms  = calc_credit_terms(prob, income, annuity_ratio,
                                          log_amt_credit=float(log_credit) if log_credit else None)

        # ── SHAP explanation ──────────────────────────────────────────────────
        shap_explain = compute_shap_explain(df, values)

        return render_template(
            "result.html",
            probability=round(prob * 100, 2),
            decision=decision,
            risk=risk,
            threshold=round(threshold * 100, 1),
            shap_explain=shap_explain,
            **credit_terms,
        )

    except Exception as e:
        return f"Prediction error: {str(e)}"


# ---------------------------
# CSV BATCH PREDICTION
# ---------------------------

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    try:
        file = request.files["file"]
        if file.filename == "":
            return "No file uploaded"

        df = pd.read_csv(file)
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            return f"Missing features: {missing}"

        try:
            threshold = float(request.form.get("threshold") or PD_BREAKEVEN)
            threshold = max(0.05, min(0.95, threshold))
        except (ValueError, TypeError):
            threshold = PD_BREAKEVEN

        X     = df[FEATURES]
        probs = model.predict_proba(X)[:, 1]

        df["default_probability"] = probs
        df["decision"] = ["Reject" if p >= threshold else "Approve" for p in probs]

        results = []
        for _, row in df.head(20).iterrows():
            income        = float(row.get("INCOME_PER_PERSON", 0) or 0)
            annuity_ratio = float(row.get("ANNUITY_BY_INCOME", MAX_DTI) or MAX_DTI)
            log_credit    = row.get("LOG_AMT_CREDIT")
            terms         = calc_credit_terms(row["default_probability"], income, annuity_ratio,
                                              log_amt_credit=float(log_credit) if log_credit else None)

            # SHAP cho từng hàng trong batch
            row_df       = pd.DataFrame([row[FEATURES].tolist()], columns=FEATURES)
            row_vals     = row[FEATURES].tolist()
            shap_explain = compute_shap_explain(row_df, row_vals)

            record = row.to_dict()
            record.update(terms)
            record["shap_explain"] = shap_explain
            results.append(record)

        return render_template(
            "result.html",
            table=results,
            threshold=round(threshold * 100, 1),
        )

    except Exception as e:
        return f"CSV prediction error: {str(e)}"


# ---------------------------
# DEBUG SHAP ROUTE
# ---------------------------

@app.route("/debug-shap")
def debug_shap():
    """Truy cập http://localhost:5000/debug-shap để kiểm tra SHAP"""
    import traceback
    lines = []

    # 1. Kiểm tra shap cài chưa
    try:
        import shap as _shap
        lines.append(f"✅ shap version: {_shap.__version__}")
    except ImportError:
        lines.append("❌ shap CHƯA được cài — chạy: pip install shap")
        return "<br>".join(lines)

    # 2. Kiểm tra model load được chưa
    if model is None:
        lines.append("❌ Model chưa load được — kiểm tra đường dẫn MODEL_PATH")
        return "<br>".join(lines)
    lines.append(f"✅ Model loaded: {type(model).__name__}")

    # 3. Thử khởi tạo explainer
    try:
        explainer = _shap.TreeExplainer(model)
        lines.append("✅ TreeExplainer khởi tạo thành công")
    except Exception as e:
        lines.append(f"❌ TreeExplainer lỗi: {e}")
        lines.append(f"<pre>{traceback.format_exc()}</pre>")
        return "<br>".join(lines)

    # 4. Thử tính SHAP với dữ liệu giả
    try:
        dummy = pd.DataFrame([[0.5]*len(FEATURES)], columns=FEATURES)
        sv = explainer.shap_values(dummy)
        ev = explainer.expected_value
        lines.append(f"✅ shap_values shape: {type(sv)} / len={len(sv) if isinstance(sv, list) else 'array'}")
        lines.append(f"✅ expected_value: {ev}")

        # Kiểm tra dạng output
        if isinstance(sv, list):
            lines.append(f"   → List với {len(sv)} phần tử (binary classification)")
            for i, arr in enumerate(sv):
                lines.append(f"   → sv[{i}] shape: {arr.shape}")
        else:
            lines.append(f"   → Array shape: {sv.shape}")

    except Exception as e:
        lines.append(f"❌ shap_values lỗi: {e}")
        lines.append(f"<pre>{traceback.format_exc()}</pre>")
        return "<br>".join(lines)

    # 5. Thử chạy compute_shap_explain
    try:
        dummy = pd.DataFrame([[0.5]*len(FEATURES)], columns=FEATURES)
        result = compute_shap_explain(dummy, [0.5]*len(FEATURES))
        if result:
            lines.append(f"✅ compute_shap_explain OK")
            lines.append(f"   base_value={result['base_value']}%")
            lines.append(f"   risk_factors ({len(result['risk_factors'])}): "
                         + ", ".join(x['label'] for x in result['risk_factors']))
            lines.append(f"   strengths ({len(result['strengths'])}): "
                         + ", ".join(x['label'] for x in result['strengths']))
        else:
            lines.append("❌ compute_shap_explain trả về None — xem terminal để biết lỗi")
    except Exception as e:
        lines.append(f"❌ compute_shap_explain exception: {e}")
        lines.append(f"<pre>{traceback.format_exc()}</pre>")

    return "<br>".join(lines)


# ---------------------------
# RUN APP
# ---------------------------

if __name__ == "__main__":
    app.run(debug=True)