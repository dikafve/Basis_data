from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
import warnings
from sklearn.linear_model import LinearRegression
import numpy as np
warnings.filterwarnings('ignore')

st.set_page_config(page_title="GenZ Dashboard", page_icon="snowfve.png", layout="wide")
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

conn = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="genz_erd"
)

query = """
SELECT 
    u.age,
    g.gender_name AS gender,
    c.country_name AS country,
    us.daily_usage_hours,
    p.platform_name AS primary_platform,
    us.num_platforms_used,
    up.purpose_name AS purpose,
    us.avg_session_minutes,
    hb.night_usage,
    hb.mental_health_score,
    al.level_name AS addiction_level,
    hb.screen_time_before_sleep
FROM user u
JOIN gender g ON u.gender_id = g.gender_id
JOIN country c ON u.country_id = c.country_id
JOIN user_session us ON u.user_id = us.user_id
JOIN platform p ON us.platform_id = p.platform_id
JOIN usage_purpose up ON us.purpose_id = up.purpose_id
JOIN health_behavior hb ON u.user_id = hb.user_id
JOIN addiction_level al ON hb.level_id = al.level_id
LIMIT 10000
"""

df = pd.read_sql(query, conn)

# Header
st.title("GenZ Social Media Usage Dashboard")
st.markdown("Data diambil langsung dari database MySQL")


# Forecasting
st.subheader("🔮 Simulasi Tren Penggunaan Sosmed berdasarkan Usia")

query_forecast = """
SELECT age, daily_usage_hours, mental_health_score, avg_session_minutes, screen_time_before_sleep 
FROM genz_social_media_usage
"""
df_forecast = pd.read_sql(query_forecast, conn)

col_a, col_b = st.columns(2)
with col_a:
    metric = st.selectbox("Pilih Metrik:", [
        'daily_usage_hours', 
        'mental_health_score', 
        'avg_session_minutes', 
        'screen_time_before_sleep'
    ])
with col_b:
    tahun_prediksi = st.slider("Prediksi sampai berapa usia ke depan?", min_value=1, max_value=20, value=5)

age_trend = df_forecast.groupby('age')[metric].mean().reset_index()

X = age_trend['age'].values.reshape(-1, 1)
y = age_trend[metric].values
model = LinearRegression()
model.fit(X, y)

current_max_age = int(age_trend['age'].max())
future_ages = np.arange(current_max_age + 1, current_max_age + tahun_prediksi + 1).reshape(-1, 1)
future_preds = model.predict(future_ages)

from scipy.ndimage import uniform_filter1d

fig_forecast, ax_forecast = plt.subplots(figsize=(12, 5))
fig_forecast.patch.set_facecolor('#ffffff')
ax_forecast.set_facecolor('#ffffff')

# Scatter data aktual
ax_forecast.scatter(age_trend['age'], age_trend[metric], 
                    color='black', alpha=0.6, s=40, zorder=2)

# Garis smooth (moving average)
y_smooth = uniform_filter1d(age_trend[metric].values, size=3)
ax_forecast.plot(age_trend['age'], y_smooth, 
                 color='green', linewidth=2, label='Tren Aktual', zorder=3)

# Garis linear regression (prediksi keseluruhan)
x_full = np.concatenate([age_trend['age'].values, future_ages.flatten()])
y_line = model.predict(x_full.reshape(-1, 1))
ax_forecast.plot(x_full, y_line, 
                 color='black', linewidth=1.5, label='Garis Prediksi', zorder=3)

# Garis batas
ax_forecast.axvline(x=current_max_age, color='black', linestyle=':', linewidth=1.5, label='Batas Prediksi')

# Styling
ax_forecast.spines['top'].set_visible(False)
ax_forecast.spines['right'].set_visible(False)
ax_forecast.grid(True, alpha=0.2, linestyle='--')
ax_forecast.set_xlabel('Usia', fontsize=12)
ax_forecast.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
ax_forecast.set_title(f'Tren & Prediksi {metric.replace("_", " ").title()} berdasarkan Usia', 
                      fontsize=14, fontweight='bold')
ax_forecast.legend(fontsize=10)

st.pyplot(fig_forecast)

st.info(f"📊 Prediksi untuk usia {current_max_age+1} hingga {current_max_age+tahun_prediksi} tahun.")

st.divider()

# KNN Classification
st.subheader("🤖 Prediksi Level Kecanduan dengan KNN")

# Siapkan data buat training
df_knn = df.dropna(subset=[
    'daily_usage_hours', 'avg_session_minutes', 
    'screen_time_before_sleep', 'mental_health_score',
    'num_platforms_used', 'addiction_level'
]).copy()

features = ['daily_usage_hours', 'avg_session_minutes', 
            'screen_time_before_sleep', 'mental_health_score', 
            'num_platforms_used']

X = df_knn[features]
y = df_knn['addiction_level']

# Encode label (addiction_level biasanya berupa teks: Low, Medium, High)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scaling fitur (penting buat KNN karena dia berbasis jarak)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

col_k1, col_k2 = st.columns(2)
with col_k1:
    k_value = st.slider("Pilih jumlah K (tetangga):", min_value=1, max_value=20, value=5)

# Training model
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

with col_k2:
    st.metric("Akurasi Model", f"{akurasi*100:.2f}%")

st.markdown("#### 🔍 Coba Prediksi Manual")
st.caption("Masukkan nilai fitur buat liat prediksi level kecanduannya")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    input_daily = st.number_input("Daily Usage (jam)", min_value=0.0, max_value=24.0, value=float(df_knn['daily_usage_hours'].mean()))
with c2:
    input_session = st.number_input("Avg Session (menit)", min_value=0.0, max_value=300.0, value=float(df_knn['avg_session_minutes'].mean()))
with c3:
    input_screen = st.number_input("Screen Time Before Sleep (menit)", min_value=0.0, max_value=300.0, value=float(df_knn['screen_time_before_sleep'].mean()))
with c4:
    input_mental = st.number_input("Mental Health Score", min_value=0.0, max_value=10.0, value=float(df_knn['mental_health_score'].mean()))
with c5:
    input_platforms = st.number_input("Jumlah Platform", min_value=1, max_value=10, value=int(df_knn['num_platforms_used'].mean()))

if st.button("Prediksi Sekarang"):
    input_data = np.array([[input_daily, input_session, input_screen, input_mental, input_platforms]])
    input_scaled = scaler.transform(input_data)
    pred_encoded = knn.predict(input_scaled)
    pred_label = le.inverse_transform(pred_encoded)
    st.success(f"📌 Prediksi Level Kecanduan: **{pred_label[0]}**")

with st.expander("📈 Lihat Detail Classification Report"):
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))
    
# Raw data
st.subheader("Raw Data")
st.dataframe(df)