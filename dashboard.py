import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Konfigurasi fitur (sesuai kolom dataset di database genz_erd)
# ---------------------------------------------------------------
TARGET = "addiction_level"
NUM_COLS = [
    "age",
    "daily_usage_hours",
    "avg_session_minutes",
    "screen_time_before_sleep",
    "mental_health_score",
    "num_platforms_used",
    "night_usage",
]
CAT_COLS = ["gender", "country", "primary_platform", "purpose"]

LABEL = {
    "age": "Usia",
    "daily_usage_hours": "Pemakaian harian (jam)",
    "avg_session_minutes": "Durasi sesi (menit)",
    "screen_time_before_sleep": "Screen time sebelum tidur (menit)",
    "mental_health_score": "Skor kesehatan mental",
    "num_platforms_used": "Jumlah platform",
    "night_usage": "Pemakaian malam (0/1)",
    "gender": "Gender",
    "country": "Negara",
    "primary_platform": "Platform utama",
    "purpose": "Tujuan penggunaan",
}


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
@st.cache_resource
def get_engine():
    return create_engine("mysql+mysqlconnector://root:@127.0.0.1:3306/genz_erd")


@st.cache_data(show_spinner="Mengambil data dari MySQL...")
def load_data(n_rows: int) -> pd.DataFrame:
    """Ambil sampel BARIS secara acak dari hasil JOIN 8 tabel.

    Catatan penting: user_id di tabel `user` ternyata adalah profil demografis
    (usia x gender x negara), bukan satu orang. Jadi baris sesi dan baris
    kesehatan harus dicocokkan lewat session_id = record_id (plus user_id),
    bukan hanya lewat user_id, supaya tidak terjadi ledakan baris (baris x baris).
    RAND(42) dipakai supaya sampelnya selalu sama (reproducible).
    """
    query = text(f"""
        SELECT
            s.session_id,
            u.age,
            g.gender_name AS gender,
            c.country_name AS country,
            s.daily_usage_hours,
            p.platform_name AS primary_platform,
            s.num_platforms_used,
            up.purpose_name AS purpose,
            s.avg_session_minutes,
            h.night_usage,
            h.mental_health_score,
            al.level_name AS addiction_level,
            h.screen_time_before_sleep
        FROM user_session s
        JOIN health_behavior h ON s.session_id = h.record_id AND s.user_id = h.user_id
        JOIN user u ON s.user_id = u.user_id
        JOIN gender g ON u.gender_id = g.gender_id
        JOIN country c ON u.country_id = c.country_id
        JOIN platform p ON s.platform_id = p.platform_id
        JOIN usage_purpose up ON s.purpose_id = up.purpose_id
        JOIN addiction_level al ON h.level_id = al.level_id
        ORDER BY RAND(42)
        LIMIT {int(n_rows)}
    """)
    df = pd.read_sql(query, get_engine())
    # Preprocessing: buang baris kosong dan duplikat (selain kolom id)
    df = df.dropna(subset=NUM_COLS + CAT_COLS + [TARGET])
    df = df.drop_duplicates(subset=NUM_COLS + CAT_COLS + [TARGET])
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def check_join() -> dict:
    """Cek apakah session_id dan record_id benar-benar sejajar."""
    query = text("""
        SELECT
            (SELECT COUNT(*) FROM user_session) AS n_session,
            (SELECT COUNT(*) FROM health_behavior) AS n_health,
            (SELECT COUNT(*) FROM user_session s
             JOIN health_behavior h
               ON s.session_id = h.record_id AND s.user_id = h.user_id) AS n_match
    """)
    return pd.read_sql(query, get_engine()).iloc[0].to_dict()


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
        class_weight="balanced",
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

    # Split 80:20, proporsi kelas dijaga (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(n_estimators, max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "baseline": y_test.value_counts(normalize=True).max(),
        "report": pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        ).transpose(),
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
    fig, ax = plt.subplots(figsize=(10, 5))
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


def section_trend(df: pd.DataFrame):
    st.subheader("🔮 Simulasi Tren berdasarkan Usia")
    trend_cols = ["daily_usage_hours", "mental_health_score",
                  "avg_session_minutes", "screen_time_before_sleep"]

    col_a, col_b = st.columns(2)
    with col_a:
        metric = st.selectbox("Pilih metrik:", trend_cols, format_func=lambda m: LABEL[m])
    with col_b:
        n_future = st.slider("Proyeksi sampai berapa usia ke depan?", 1, 10, 3)

    # rata-rata metrik untuk tiap usia
    age_trend = df.groupby("age")[metric].mean().reset_index(name="value")

    slope, intercept = np.polyfit(age_trend["age"], age_trend["value"], 1)
    max_age = int(age_trend["age"].max())
    x_full = np.arange(age_trend["age"].min(), max_age + n_future + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
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


def section_model(df: pd.DataFrame, n_rows: int, n_estimators: int, max_depth):
    st.subheader("🌲 Prediksi Level Kecanduan dengan Random Forest")

    result = train_model(df, n_rows, n_estimators, max_depth)
    model = result["model"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Akurasi model", f"{result['accuracy'] * 100:.2f}%")
    c2.metric("Baseline (selalu tebak kelas terbanyak)", f"{result['baseline'] * 100:.2f}%")
    c3.metric("Data train / test", f"{result['n_train']:,} / {result['n_test']:,}")

    if result["accuracy"] >= 0.99:
        st.warning("Akurasi hampir sempurna. Cek apakah addiction_level diturunkan langsung "
                   "dari salah satu fitur (kebocoran label) sebelum mengklaim hasil ini.")
    elif result["accuracy"] <= result["baseline"]:
        st.info("Akurasi belum melampaui baseline. Model belum lebih baik daripada "
                "selalu menebak kelas terbanyak.")
    st.caption("Split 80:20 dengan proporsi kelas dijaga (stratified).")

    left, right = st.columns([3, 2])
    with left:
        st.pyplot(plot_importance(result["importance"]))
    with right:
        st.pyplot(plot_confusion(result["y_test"], result["y_pred"], result["classes"]))

    st.markdown("#### 🔍 Coba Prediksi Manual")
    st.caption("Isi nilai fitur untuk melihat prediksi level kecanduannya")

    values = {}
    cols = st.columns(4)
    for i, col in enumerate(NUM_COLS):
        with cols[i % 4]:
            if col == "night_usage":
                values[col] = st.selectbox(LABEL[col], [0, 1], index=1)
            elif col in ("age", "num_platforms_used"):
                values[col] = st.number_input(
                    LABEL[col], min_value=int(df[col].min()), max_value=int(df[col].max()),
                    value=int(df[col].median()), step=1)
            else:
                values[col] = st.number_input(
                    LABEL[col], min_value=float(df[col].min()), max_value=float(df[col].max()),
                    value=float(df[col].median()))
    cols = st.columns(4)
    for i, col in enumerate(CAT_COLS):
        with cols[i]:
            options = sorted(df[col].unique())
            values[col] = st.selectbox(LABEL[col], options)

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

    st.sidebar.header("Pengaturan")
    n_rows = st.sidebar.slider("Jumlah baris sampel", 2000, 50000, 10000, step=2000)
    n_estimators = st.sidebar.slider("Jumlah pohon (n_estimators)", 50, 300, 100, step=50)
    depth_option = st.sidebar.select_slider(
        "Kedalaman maksimum pohon", options=[5, 10, 15, 20, "Tanpa batas"], value="Tanpa batas")
    max_depth = None if depth_option == "Tanpa batas" else int(depth_option)

    st.title("GenZ Social Media Usage Dashboard")
    st.markdown("Data diambil langsung dari database MySQL (`genz_erd`)")

    df = load_data(n_rows)

    chk = check_join()
    ratio = chk["n_match"] / max(min(chk["n_session"], chk["n_health"]), 1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Baris dipakai", f"{len(df):,}")
    c2.metric("Baris sesi cocok dengan data kesehatan", f"{int(chk['n_match']):,} dari {int(min(chk['n_session'], chk['n_health'])):,}")
    c3.metric("Kelas level kecanduan", ", ".join(sorted(df[TARGET].unique())))
    if ratio < 0.9:
        st.error("session_id dan record_id tampaknya TIDAK sejajar (kecocokan rendah). "
                 "Hasil model di bawah belum bisa dipercaya, cek isi tabelnya dulu.")

    section_trend(df)
    st.divider()
    section_model(df, n_rows, n_estimators, max_depth)

    st.subheader("Raw Data")
    st.dataframe(df.head(1000))


if __name__ == "__main__":
    main()