import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------
# Catatan struktur data (hasil pengecekan CSV):
# - user_id di tabel `user` adalah profil demografis (usia, gender, negara).
# - user_session dan health_behavior TIDAK berpasangan baris per baris
#   (session_id dan record_id hanya nomor urut), jadi keduanya tidak boleh
#   di-JOIN satu sama lain. Masing-masing hanya di-JOIN ke tabel `user`.
TARGET = "addiction_level"
NUM_COLS = ["age", "mental_health_score", "screen_time_before_sleep", "night_usage"]
CAT_COLS = ["gender", "country"]

LABEL = {
    "age": "Usia",
    "mental_health_score": "Skor kesehatan mental",
    "screen_time_before_sleep": "Screen time sebelum tidur (menit)",
    "night_usage": "Pemakaian malam (0/1)",
    "gender": "Gender",
    "country": "Negara",
    "daily_usage_hours": "Pemakaian harian (jam)",
    "avg_session_minutes": "Durasi sesi (menit)",
    "num_platforms_used": "Jumlah platform",
}
SESSION_METRICS = ["daily_usage_hours", "avg_session_minutes", "num_platforms_used"]
HEALTH_METRICS = ["mental_health_score", "screen_time_before_sleep"]


# ---------------------------------------------------------------
# Data (JOIN banyak-ke-satu, jadi jumlah baris tidak meledak)
# ---------------------------------------------------------------
@st.cache_resource
def get_engine():
    return create_engine("mysql+mysqlconnector://root:@127.0.0.1:3306/genz_erd")


@st.cache_data(show_spinner="Mengambil data kesehatan dari MySQL...")
def load_health() -> pd.DataFrame:
    query = text("""
        SELECT
            h.record_id,
            u.age,
            g.gender_name AS gender,
            c.country_name AS country,
            h.mental_health_score,
            h.night_usage,
            h.screen_time_before_sleep,
            al.level_name AS addiction_level
        FROM health_behavior h
        JOIN user u ON h.user_id = u.user_id
        JOIN gender g ON u.gender_id = g.gender_id
        JOIN country c ON u.country_id = c.country_id
        JOIN addiction_level al ON h.level_id = al.level_id
    """)
    df = pd.read_sql(query, get_engine())
    df = df.dropna(subset=NUM_COLS + CAT_COLS + [TARGET])  # preprocessing
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Mengambil data sesi dari MySQL...")
def load_session() -> pd.DataFrame:
    query = text("""
        SELECT
            s.session_id,
            u.age,
            g.gender_name AS gender,
            c.country_name AS country,
            p.platform_name AS platform,
            up.purpose_name AS purpose,
            s.daily_usage_hours,
            s.num_platforms_used,
            s.avg_session_minutes
        FROM user_session s
        JOIN user u ON s.user_id = u.user_id
        JOIN gender g ON u.gender_id = g.gender_id
        JOIN country c ON u.country_id = c.country_id
        JOIN platform p ON s.platform_id = p.platform_id
        JOIN usage_purpose up ON s.purpose_id = up.purpose_id
    """)
    df = pd.read_sql(query, get_engine())
    df = df.dropna()
    # buang baris kembar (selain nomor sesi)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "session_id"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------
# Model: Random Forest Classifier
# ---------------------------------------------------------------
def build_model(n_estimators: int, max_depth):
    pre = ColumnTransformer([
        ("num", "passthrough", NUM_COLS),  # Random Forest tidak butuh scaling
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",  # kelas High jauh lebih sedikit dari Medium
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("rf", rf)])


def aggregate_importance(model: Pipeline) -> pd.Series:
    """Gabungkan importance kolom one-hot kembali ke fitur aslinya."""
    names = model.named_steps["pre"].get_feature_names_out()
    imp = model.named_steps["rf"].feature_importances_
    agg = {}
    for name, value in zip(names, imp):
        base = name.split("__", 1)[1]
        original = next((c for c in CAT_COLS if base.startswith(c + "_")), base)
        agg[original] = agg.get(original, 0.0) + value
    return pd.Series(agg).sort_values(ascending=False)


@st.cache_resource(show_spinner="Melatih Random Forest...")
def train_model(_df: pd.DataFrame, n_rows: int, n_estimators: int, max_depth):
    # n_rows ikut jadi kunci cache, _df tidak di-hash (terlalu berat)
    X, y = _df[NUM_COLS + CAT_COLS], _df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_model(n_estimators, max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).transpose()
    minority = y_test.value_counts().idxmin()  # kelas paling jarang (High)
    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "baseline": y_test.value_counts(normalize=True).max(),
        "balanced": balanced_accuracy_score(y_test, y_pred),
        "minority": minority,
        "min_recall": report.loc[minority, "recall"],
        "min_f1": report.loc[minority, "f1-score"],
        "report": report,
        "classes": list(model.classes_),
        "importance": aggregate_importance(model),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


# ---------------------------------------------------------------
# Tampilan
# ---------------------------------------------------------------
def apply_style():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        }
        .stMetric {
            background-color: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .stSubheader, h1, h2, h3 {
            color: #F9F6EE !important;
        }
        div[data-testid="metric-container"] {
            background-color: rgba(233,69,96,0.1);
            border: 1px solid rgba(233,69,96,0.3);
            border-radius: 10px;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)


def plot_importance(importance: pd.Series):
    imp = importance.sort_values()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#ffffff")
    ax.barh([LABEL.get(i, i) for i in imp.index], imp.values, color="#7d62bd")
    ax.set_xlabel("Feature importance")
    ax.set_title("Fitur yang paling berpengaruh terhadap level kecanduan", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.2, linestyle="--")
    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion matrix", fontweight="bold")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


def section_trend(df_session: pd.DataFrame, df_health: pd.DataFrame):
    st.subheader("🔮 Simulasi Tren berdasarkan Usia")
    metric = st.selectbox("Pilih metrik:", SESSION_METRICS + HEALTH_METRICS,
                          format_func=lambda m: LABEL[m])
    n_future = st.slider("Proyeksi sampai berapa usia ke depan?", 1, 10, 3)

    source = df_session if metric in SESSION_METRICS else df_health
    age_trend = source.groupby("age")[metric].mean().reset_index(name="value")

    slope, intercept = np.polyfit(age_trend["age"], age_trend["value"], 1)
    max_age = int(age_trend["age"].max())
    x_full = np.arange(age_trend["age"].min(), max_age + n_future + 1)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor("#ffffff")
    ax.scatter(age_trend["age"], age_trend["value"], color="black", alpha=0.6, s=40,
               label="Rata-rata aktual", zorder=3)
    ax.plot(x_full, slope * x_full + intercept, color="#7d62bd", linewidth=2,
            label="Garis tren (regresi linear)", zorder=2)
    ax.axvline(max_age, color="black", linestyle=":", linewidth=1.5, label="Batas data")
    ax.set_xlabel("Usia")
    ax.set_ylabel(LABEL[metric])
    ax.set_title(f"Tren {LABEL[metric]} berdasarkan usia", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.legend()
    st.pyplot(fig)
    st.caption("Garis tren adalah regresi linear sederhana per usia (simulasi), bukan time series.")


def section_session(df_session: pd.DataFrame):
    st.subheader("📱 Pola Pemakaian: Platform, Tujuan, dan Durasi Sesi")
    st.caption("Bagian ini bersifat deskriptif. Data sesi tidak bisa dipasangkan baris per baris "
               "dengan label level kecanduan, sehingga tidak dipakai sebagai fitur model.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Jumlah sesi per platform**")
        st.bar_chart(df_session["platform"].value_counts())
    with c2:
        st.markdown("**Jumlah sesi per tujuan penggunaan**")
        st.bar_chart(df_session["purpose"].value_counts())

    summary = (df_session.groupby("platform")[SESSION_METRICS].mean()
               .rename(columns=LABEL).round(2))
    st.markdown("**Rata-rata per platform**")
    st.dataframe(summary)


def section_model(df_health: pd.DataFrame, n_rows: int, n_estimators: int, max_depth):
    st.subheader("🌲 Prediksi Level Kecanduan dengan Random Forest")

    df = df_health.sample(n=min(n_rows, len(df_health)), random_state=42)
    result = train_model(df, n_rows, n_estimators, max_depth)
    model = result["model"]

    dist = df[TARGET].value_counts(normalize=True).mul(100).round(1)
    st.caption("Distribusi level kecanduan: " + ", ".join(f"{k} {v}%" for k, v in dist.items()))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi", f"{result['accuracy'] * 100:.2f}%")
    c2.metric("Baseline (selalu tebak kelas terbanyak)", f"{result['baseline'] * 100:.2f}%")
    c3.metric("Balanced accuracy", f"{result['balanced'] * 100:.2f}%")
    c4.metric(f"Recall kelas {result['minority']}", f"{result['min_recall'] * 100:.2f}%")

    if result["baseline"] > 0.9:
        st.info("Data sangat tidak seimbang, jadi akurasi mudah terlihat tinggi. "
                f"Perhatikan recall dan F1 kelas {result['minority']} "
                f"(F1 = {result['min_f1']:.2f}) untuk menilai model.")
    st.caption(f"Data latih {result['n_train']:,} baris, data uji {result['n_test']:,} baris "
               "(split 80:20, proporsi kelas dijaga).")

    left, right = st.columns([3, 2])
    with left:
        st.pyplot(plot_importance(result["importance"]))
    with right:
        st.pyplot(plot_confusion(result["y_test"], result["y_pred"], result["classes"]))

    st.markdown("#### 🔍 Coba Prediksi Manual")
    st.caption("Isi nilai fitur untuk melihat prediksi level kecanduannya")

    values = {}
    cols = st.columns(3)
    with cols[0]:
        values["age"] = st.number_input(LABEL["age"], min_value=int(df["age"].min()),
                                        max_value=int(df["age"].max()),
                                        value=int(df["age"].median()), step=1)
        values["gender"] = st.selectbox(LABEL["gender"], sorted(df["gender"].unique()))
    with cols[1]:
        values["mental_health_score"] = st.number_input(
            LABEL["mental_health_score"], min_value=0.0, max_value=10.0,
            value=float(df["mental_health_score"].median()))
        values["country"] = st.selectbox(LABEL["country"], sorted(df["country"].unique()))
    with cols[2]:
        values["screen_time_before_sleep"] = st.number_input(
            LABEL["screen_time_before_sleep"], min_value=0.0,
            max_value=float(df["screen_time_before_sleep"].max()),
            value=float(df["screen_time_before_sleep"].median()))
        values["night_usage"] = st.selectbox(LABEL["night_usage"], [0, 1], index=1)

    if st.button("Prediksi Sekarang"):
        row = pd.DataFrame([values])[NUM_COLS + CAT_COLS]
        pred = model.predict(row)[0]
        proba = pd.Series(model.predict_proba(row)[0], index=model.classes_, name="Probabilitas")
        st.success(f"📌 Prediksi Level Kecanduan: **{pred}**")
        st.bar_chart(proba)

    with st.expander("📈 Lihat Detail Classification Report"):
        st.dataframe(result["report"].style.format("{:.2f}"))


def main():
    st.set_page_config(page_title="GenZ Dashboard", page_icon="snowfve.png", layout="wide")
    apply_style()

    st.sidebar.header("Pengaturan model")
    n_rows = st.sidebar.slider("Jumlah baris untuk model", 10000, 100000, 50000, step=10000)
    n_estimators = st.sidebar.slider("Jumlah pohon (n_estimators)", 50, 300, 100, step=50)
    depth_option = st.sidebar.select_slider(
        "Kedalaman maksimum pohon", options=[5, 10, 15, 20, "Tanpa batas"], value="Tanpa batas")
    max_depth = None if depth_option == "Tanpa batas" else int(depth_option)

    st.title("GenZ Social Media Usage Dashboard")
    st.markdown("Data diambil langsung dari database MySQL (`genz_erd`)")

    df_health = load_health()
    df_session = load_session()

    c1, c2 = st.columns(2)
    c1.metric("Baris data kesehatan (untuk model)", f"{len(df_health):,}")
    c2.metric("Baris data sesi (untuk eksplorasi)", f"{len(df_session):,}")

    section_trend(df_session, df_health)
    st.divider()
    section_session(df_session)
    st.divider()
    section_model(df_health, n_rows, n_estimators, max_depth)


if __name__ == "__main__":
    main()