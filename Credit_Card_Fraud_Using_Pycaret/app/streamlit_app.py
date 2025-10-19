# app/streamlit_app.py
# مرحله 1/4: اسکلت اپ + سایدبار انتخاب مدل/آستانه + چهار تب خالی
import os
import json
import io
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# ===== تنظیمات صفحه (حتماً اولین دستور Streamlit باشد)
st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide")

# ===== مسیرها
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Credit_Card_Fraud_Using_Pycaret
MODELS_DIR = PROJECT_ROOT / "models"

# ===== ابزارها
@st.cache_resource
def load_model(model_name: Optional[str] = None):
    """مدل (Pipeline یا Estimator) را از پوشه models می‌خواند."""
    if model_name is None:
        model_path = MODELS_DIR / "best_model.pkl"
    else:
        model_path = MODELS_DIR / model_name
    if not model_path.exists():
        st.error(f"مدل '{model_path.name}' یافت نشد. اول آموزش را اجرا کن و مدل‌ها را در {MODELS_DIR} بساز.")
        st.stop()
    model = joblib.load(model_path)
    return model

@st.cache_resource
def load_scaler():
    sc_path = MODELS_DIR / "scaler.pkl"
    if not sc_path.exists():
        st.error("scaler.pkl در پوشه models پیدا نشد. ابتدا آموزش نوت‌بوک را اجرا کن.")
        st.stop()
    return joblib.load(sc_path)

def read_metrics_summary():
    ms_path = MODELS_DIR / "metrics_summary.json"
    if not ms_path.exists():
        return None
    with open(ms_path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_models():
    models = []
    if MODELS_DIR.exists():
        for p in MODELS_DIR.glob("*_model.pkl"):
            models.append(p.name)
    # همیشه best_model را هم پیشنهاد بده
    if (MODELS_DIR / "best_model.pkl").exists():
        models = ["best_model.pkl"] + [m for m in models if m != "best_model.pkl"]
    return models

def preprocess_df(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """Amount/Time را scale می‌کند و آن‌ها را حذف می‌کند. نبود برخی Vها را با صفر پر می‌کند."""
    df = df.copy()
    # حذف ستون Class اگر هست
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    # چک الزامات
    required_cols = ["Amount", "Time"]
    missing_req = [c for c in required_cols if c not in df.columns]
    if missing_req:
        st.error(f"ستون‌های ضروری {missing_req} در فایل ورودی وجود ندارند.")
        st.stop()
    # پر کردن V1..V28 اگر کم دارند
    v_cols = [f"V{i}" for i in range(1, 29)]
    for c in v_cols:
        if c not in df.columns:
            df[c] = 0.0
    # scale
    arr = scaler.transform(df[["Amount", "Time"]])
    df["scaled_amount"] = arr[:, 0]
    df["scaled_time"] = arr[:, 1]
    # حذف Amount/Time
    df = df.drop(columns=["Amount", "Time"])
    return df

def predict_df(model, X: pd.DataFrame, threshold: float):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return prob, pred

# ===== عنوان
st.title("🛡️ Credit Card Fraud Detection — Dashboard")

# ===== سایدبار: انتخاب مدل و آستانه
with st.sidebar:
    st.header("⚙️ تنظیمات مدل")
    models = list_models()
    if not models:
        st.warning("مدلی پیدا نشد. ابتدا نوت‌بوک آموزش را اجرا کن تا پوشه models ساخته شود.")
    model_choice = st.selectbox("انتخاب مدل", options=models if models else ["(نیست)"], index=0)
    metrics_summary = read_metrics_summary()
    default_thr = 0.5
    if metrics_summary:
        # تلاش برای خواندن آستانه بهینهٔ همان مدل
        try:
            if model_choice == "best_model.pkl":
                best_name = metrics_summary.get("best_model")
                if best_name and "thresholds" in metrics_summary:
                    default_thr = float(metrics_summary["thresholds"][best_name]["t"] if "t" in metrics_summary["thresholds"][best_name] 
                                        else metrics_summary["thresholds"][best_name]["best_threshold"])
            else:
                key = model_choice.replace("_model.pkl", "")
                throbj = metrics_summary.get("thresholds", {}).get(key)
                if throbj:
                    default_thr = float(throbj.get("t", throbj.get("best_threshold", 0.5)))
        except Exception:
            pass
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=float(default_thr), step=0.01)
    st.caption("آستانه‌ی بزرگ‌تر → Precision بالاتر / Recall پایین‌تر. آستانه‌ی کوچک‌تر → Recall بالاتر / Precision پایین‌تر.")

# لود مدل و اسکلر
if models:
    model = load_model(None if model_choice == "best_model.pkl" else model_choice)
    scaler = load_scaler()
else:
    model = None
    scaler = None
# === Helpers for analytics and inspection ===
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc

def evaluate_if_labeled(df_with_preds: pd.DataFrame):
    """
    اگر ستون Class وجود داشت، متریک‌ها و منحنی‌ها را حساب می‌کند.
    خروجی: dict شامل متریک‌ها و نقاط ROC/PR
    """
    if "Class" not in df_with_preds.columns:
        return None
    y_true = df_with_preds["Class"].astype(int).values
    y_prob = df_with_preds["fraud_prob"].values
    y_pred = df_with_preds["fraud_pred"].astype(int).values

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc":  float(average_precision_score(y_true, y_prob)),
    }
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)

    curves = {"fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec, "thr": thr}
    return {"metrics": metrics, "curves": curves}

def plot_roc(ax, fpr, tpr, title="ROC Curve"):
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0,1],[0,1],"--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(alpha=0.3)

def plot_pr(ax, rec, prec, title="Precision-Recall"):
    ax.plot(rec, prec, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(alpha=0.3)

def feature_importance_df(model, feature_names: list):
    """برگشت دیتافریم اهمیت ویژگی برای RF/XGB/LGBM یا ضرایب LogReg"""
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        return pd.DataFrame({"feature": feature_names, "importance": vals}).sort_values("importance", ascending=False)
    if hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        return pd.DataFrame({"feature": feature_names, "importance": np.abs(coefs), "coef": coefs}).sort_values("importance", ascending=False)
    return None

# ===== تب‌ها
tab1, tab2, tab3, tab4 = st.tabs([
    "📥 Predict CSV",
    "📊 Analytics & Charts",
    "🧾 Single Transaction",
    "🧠 Model & Threshold"
])

# فعلاً محتوا خالی—در مرحلهٔ بعد Tab1 را کامل می‌کنیم
with tab1:
    st.subheader("📥 Predict on CSV")

    up = st.file_uploader("یک فایل CSV با ستون‌های Amount, Time, V1..V28 آپلود کن", type=["csv"])
    if up is not None and model is not None and scaler is not None:
        # خواندن
        df_in = pd.read_csv(up)
        st.write("پیش‌نمایش داده‌ها:", df_in.head())

        # پیش‌پردازش
        try:
            X = preprocess_df(df_in, scaler)
        except Exception as e:
            st.error(str(e))
            st.stop()

        # پیش‌بینی
        prob, pred = predict_df(model, X, threshold)
        out = df_in.copy()
        out["fraud_prob"] = prob
        out["fraud_pred"] = pred

        # نمایش خروجی
        st.success("پیش‌بینی انجام شد ✅")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("تعداد کل ردیف‌ها", len(out))
        with c2:
            st.metric("مشکوک (pred=1)", int(out["fraud_pred"].sum()))
        with c3:
            ratio = 100.0 * (out["fraud_pred"].sum() / max(1, len(out)))
            st.metric("درصد مشکوک", f"{ratio:.3f}%")

        st.write("چند ردیف نمونه از خروجی:")
        st.dataframe(out.head(20), use_container_width=True)

        # دانلود
        csv_buf = io.StringIO()
        out.to_csv(csv_buf, index=False)
        st.download_button(
            "دانلود نتایج (CSV)",
            data=csv_buf.getvalue(),
            file_name="predictions.csv",
            mime="text/csv"
        )

    # === Histogram of predicted fraud probabilities (enhanced with zoom & controls) ===
    st.markdown("### Histogram of Predicted Fraud Probabilities")

    if "out" in locals():
        probs = out["fraud_prob"].values

        # کنترل‌های نمایشی
        c1, c2, c3 = st.columns([2, 1.5, 1.5])
        with c1:
            zoom_range = st.slider("Zoom range (X-axis)", 0.0, 1.0, (0.0, 1.0), 0.01)
        with c2:
            bins = st.slider("Number of bins", 20, 200, 100, 5)
        with c3:
            yscale = st.selectbox("Y-axis scale", ["log", "linear"], index=0)

        # فیلتر بازه
        mask = (probs >= zoom_range[0]) & (probs <= zoom_range[1])
        probs_view = probs[mask]

        if len(probs_view) == 0:
            st.warning("در این بازه هیچ مقداری وجود ندارد. بازه را تغییر بده.")
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            vals, bins_edges, patches = ax.hist(
                probs_view,
                bins=bins,
                color="steelblue",
                alpha=0.8,
                edgecolor="white"
            )

            # مقیاس لگاریتمی برای نمایش بهتر
            if yscale == "log":
                ax.set_yscale("log")

            # رنگ قرمز برای ستون‌هایی که احتمال بالا دارند
            centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
            for c, p in zip(centers, patches):
                if c >= threshold:
                    p.set_facecolor("crimson")

            # نمایش خطوط آماری
            mean_val = probs_view.mean()
            median_val = np.median(probs_view)
            ax.axvline(mean_val, color="orange", linestyle="--", lw=2, label=f"Mean = {mean_val:.3f}")
            ax.axvline(median_val, color="green", linestyle="--", lw=2, label=f"Median = {median_val:.3f}")
            ax.axvline(threshold, color="crimson", linestyle="--", lw=2, label=f"Threshold = {threshold:.2f}")

            # برچسب‌ها و ظاهر
            ax.set_title("Histogram of Predicted Fraud Probabilities (Zoomable)")
            ax.set_xlabel("Fraud Probability")
            ax.set_ylabel("Count" + (" (log scale)" if yscale == "log" else ""))
            ax.grid(alpha=0.3)
            ax.legend()

            st.pyplot(fig)

            # خلاصه آماری
            n_total = len(probs_view)
            n_high = int((probs_view >= threshold).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Samples in view", n_total)
            c2.metric("≥ Threshold", n_high)
            c3.metric("% High-risk", f"{100.0 * n_high / max(1, n_total):.3f}%")

            # نمودار تجمعی (برای درک توزیع دُم)
            st.markdown("#### Cumulative Distribution (CCDF)")
            xs = np.sort(probs_view)
            ccdf = 1.0 - np.arange(1, len(xs) + 1) / len(xs)
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.plot(xs, ccdf, lw=2)
            if yscale == "log":
                ax2.set_yscale("log")
            ax2.axvline(threshold, color="crimson", linestyle="--", lw=2)
            ax2.set_xlabel("Fraud Probability")
            ax2.set_ylabel("Fraction ≥ x")
            ax2.set_title("CCDF of Fraud Probabilities")
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)

            st.caption("""
            **راهنما:**  
            - از اسلایدر **Zoom range** برای زوم روی بازه‌های خاص استفاده کن (مثلاً 0.05–1.0).  
            - محور **Y (log)** باعث می‌شه ستون‌های کوچک‌تر هم دیده بشن.  
            - نوارهای قرمز نشانه‌ی تراکنش‌های با احتمال بالای آستانه فعلی هستن.  
            - خطوط نارنجی و سبز میانگین و میانه‌ی احتمال‌ها رو نشون می‌دن.
            """)
    else:
        st.info("فایل را آپلود کن تا پیش‌بینی انجام شود.")



with tab2:
    st.subheader("📊 Analytics & Charts")

    st.caption("ابتدا در تب «Predict CSV» فایل را آپلود و پیش‌بینی بگیرید؛ اینجا تحلیل‌های آماری روی همان خروجی انجام می‌شود.")

    if "out" in locals():
        # out همان خروجی تب 1 است (df ورودی + fraud_prob + fraud_pred)
        df_out = out.copy()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total rows", len(df_out))
        with c2:
            st.metric("Predicted Fraud (1)", int(df_out["fraud_pred"].sum()))
        with c3:
            st.metric("Avg fraud prob", f"{df_out['fraud_prob'].mean():.4f}")
        with c4:
            st.metric("Max fraud prob", f"{df_out['fraud_prob'].max():.4f}")

        st.markdown("---")

        # Distribution charts
        cL, cR = st.columns([2, 1])
        with cL:
            fig, ax = plt.subplots(figsize=(7,3))
            ax.hist(df_out["fraud_prob"], bins=50)
            ax.set_title("Histogram of predicted fraud probabilities")
            ax.set_xlabel("fraud_prob")
            ax.set_ylabel("count")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        with cR:
            counts = df_out["fraud_pred"].value_counts().reindex([0,1], fill_value=0)
            fig, ax = plt.subplots(figsize=(4,3))
            ax.pie(counts.values, labels=["Not Fraud", "Fraud"], autopct="%1.2f%%", startangle=90)
            ax.set_title("Fraud vs Not Fraud (pred)")
            st.pyplot(fig)

        # Amount bins vs fraud rate
        st.markdown("### Fraud rate by Amount bins")
        try:
            bins = st.slider("Number of bins", 5, 50, 10, 1)
            df_tmp = df_out.copy()
            if "Amount" in df_tmp.columns:
                df_tmp["amount_bin"] = pd.qcut(df_tmp["Amount"], q=bins, duplicates="drop")
                grp = df_tmp.groupby("amount_bin")["fraud_pred"].mean().reset_index()
                fig, ax = plt.subplots(figsize=(10,3))
                ax.bar(range(len(grp)), grp["fraud_pred"].values)
                ax.set_xticks(range(len(grp)))
                ax.set_xticklabels([str(x) for x in grp["amount_bin"]], rotation=60, ha="right")
                ax.set_ylabel("Fraud rate (pred)")
                ax.set_title("Fraud rate by transaction amount (quantile bins)")
                ax.grid(axis="y", alpha=0.3)
                st.pyplot(fig)
            else:
                st.info("ستون Amount در داده‌ی آپلودی نبود؛ این بخش را رد می‌کنیم.")
        except Exception as e:
            st.warning(f"تحلیل Amount با خطا مواجه شد: {e}")

        # Top-K suspicious
        st.markdown("### Top-K suspicious transactions")
        K = st.slider("Top-K", 5, 100, 20, 1)
        st.dataframe(df_out.sort_values("fraud_prob", ascending=False).head(K), use_container_width=True)

        st.markdown("---")

        # If labeled, show ROC/PR on uploaded set
        eval_res = evaluate_if_labeled(df_out)
        if eval_res is not None:
            st.markdown("### Evaluation on uploaded CSV (ground-truth detected)")
            met = eval_res["metrics"]
            curves = eval_res["curves"]
            c1, c2 = st.columns(2)
            with c1:
                st.metric("ROC AUC (uploaded)", f"{met['roc_auc']:.4f}")
            with c2:
                st.metric("PR AUC (uploaded)", f"{met['pr_auc']:.4f}")

            fig, ax = plt.subplots(figsize=(5,4))
            plot_roc(ax, curves["fpr"], curves["tpr"], title="ROC (uploaded)")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(5,4))
            plot_pr(ax, curves["rec"], curves["prec"], title="Precision-Recall (uploaded)")
            st.pyplot(fig)
        else:
            st.info("برچسب واقعی (ستون Class) در فایل نبود؛ نمودارهای ROC/PR روی فایل آپلودی قابل محاسبه نیست.")
    else:
        st.info("ابتدا تب «Predict CSV» را اجرا کن تا اینجا قابل استفاده باشد.")


with tab3:
    st.subheader("🧾 Single Transaction Tester")

    if model is None or scaler is None:
        st.warning("مدل/اسکلر لود نشده. به سایدبار نگاه کن.")
        st.stop()

    st.caption("حداقل Amount و Time لازم است. اگر Vها را ندهی، صفر در نظر گرفته می‌شود.")

    # ورودی‌ها در گرید
    cols_top = st.columns(4)
    Amount = cols_top[0].number_input("Amount", value=100.0, step=1.0)
    Time   = cols_top[1].number_input("Time", value=50000.0, step=100.0)
    default_fill_zero = cols_top[2].checkbox("Fill missing V1..V28 with 0", value=True)
    show_v = cols_top[3].checkbox("Show V1..V28 inputs", value=False)

    v_vals = {}
    if show_v:
        # 28 ویژگی را در 4 ستون تقسیم می‌کنیم
        grid = [st.columns(4) for _ in range(7)]  # 7 ردیف × 4 ستون = 28
        idx = 0
        for r in range(7):
            for c in range(4):
                vi = idx + 1
                v_vals[f"V{vi}"] = grid[r][c].number_input(f"V{vi}", value=0.0, step=0.1, format="%.4f")
                idx += 1
    else:
        if default_fill_zero:
            for i in range(1, 29):
                v_vals[f"V{i}"] = 0.0

    if st.button("Predict"):
        row = {"Amount": Amount, "Time": Time, **v_vals}
        df_one = pd.DataFrame([row])

        # preprocess
        arr = scaler.transform(df_one[["Amount","Time"]])
        df_one["scaled_amount"] = arr[:, 0]
        df_one["scaled_time"]   = arr[:, 1]
        X = df_one.drop(columns=["Amount","Time"])

        # predict
        prob = float(model.predict_proba(X)[:, 1][0])
        pred = int(prob >= threshold)

        # نمایش نتیجه
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Fraud Probability", f"{prob:.4f}")
        with c2:
            st.metric("Prediction", "FRAUD" if pred==1 else "OK")
        with c3:
            st.metric("Threshold used", f"{threshold:.2f}")

        # Progress bar برای حس بهتر
        st.progress(min(max(prob, 0.0), 1.0))


with tab4:
    st.subheader("🧠 Model & Threshold Inspector")

    if model is None or scaler is None:
        st.warning("مدل/اسکلر لود نشده.")
        st.stop()

    # نمایش اطلاعات خلاصه از فایل metrics_summary.json
    ms = read_metrics_summary()
    if ms:
        best_name = ms.get("best_model", None)
        st.write("**Best model (from training):**", best_name if best_name else "-")

        # اگر اطلاعات summary قدیمی (نسخه قبلی نوت‌بوک) باشد
        # دو حالت اسم کلیدها را هندل می‌کنیم:
        if "summary" in ms:
            df_sum = pd.DataFrame(ms["summary"]).T
        elif "global" in ms:
            df_sum = pd.DataFrame(ms["global"]).T
        else:
            df_sum = None

        if df_sum is not None:
            st.markdown("**Training summary:**")
            st.dataframe(df_sum, use_container_width=True)

        st.caption("اگر فایل متریک موجود نباشد یا ساختار متفاوت باشد، جدول بالا ممکن است خالی باشد.")
    else:
        st.info("metrics_summary.json پیدا نشد.")

    st.markdown("---")

    # اهمیت ویژگی‌ها (اگر وجود داشته باشد)
    # ساخت لیست نام ویژگی‌ها مطابق preprocess_df: V1..V28 + scaled_amount + scaled_time
    feature_names = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
    fi = feature_importance_df(model, feature_names)
    if fi is not None:
        st.markdown("### Feature Importance / Coefficients")
        st.dataframe(fi.head(30), use_container_width=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        head = fi.head(15)
        ax.barh(head["feature"][::-1], head["importance"][::-1])
        ax.set_title("Top-15 Feature Importance")
        ax.set_xlabel("Importance (abs coef or split gain)")
        st.pyplot(fig)
    else:
        st.info("این مدل فیچر ایمپورتنس/ضریب قابل نمایش ندارد.")

    st.markdown("---")
    st.markdown("### Threshold tips")
    st.write(
        """
        - آستانهٔ بزرگ‌تر → **Precision** بیشتر، **Recall** کمتر (مناسب کاهش آلارم‌های اشتباه).
        - آستانهٔ کوچک‌تر → **Recall** بیشتر، **Precision** کمتر (مناسب کشف حداکثری).
        - در سایدبار آستانه را تنظیم و در تب اول و سوم رفتار مدل را ببین.
        """
    )

