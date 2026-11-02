"""
diagnosa_db.py - hanya MEMBACA database (tidak mengubah apa pun).
Tujuan: mencari tahu kenapa model di komputermu memberi hasil berbeda dari simulasi di mesin lain.
Jalankan:  python diagnosa_db.py
Salin SEMUA teks yang muncul di terminal dan kirim ke chat.
"""
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")
URL = sys.argv[1] if len(sys.argv) > 1 else "mysql+mysqlconnector://root:@127.0.0.1:3306/genz_erd"
engine = create_engine(URL)

print("=== 1. VERSI ===")
print(f"Python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__} | scikit-learn {sklearn.__version__}")
print("Default RandomForestClassifier():", RandomForestClassifier().get_params())

print("\n=== 2. TIPE KOLOM DI MYSQL ===")
with engine.connect() as conn:
    try:
        cols = pd.read_sql(text("SELECT table_name AS tabel, column_name AS kolom, column_type AS tipe "
                                "FROM information_schema.columns WHERE table_schema=DATABASE() "
                                "AND table_name IN ('health_behavior','user')"), conn)
        print(cols.to_string(index=False))
    except Exception as e:  # DB non-MySQL
        print("(lewati)", type(e).__name__)

print("\n=== 3. DATA HASIL QUERY DASHBOARD ===")
sql = """SELECT u.age, g.gender_name AS gender, c.country_name AS country, h.mental_health_score,
         h.night_usage, h.screen_time_before_sleep, al.level_name AS addiction_level
         FROM health_behavior h JOIN user u ON h.user_id=u.user_id JOIN gender g ON u.gender_id=g.gender_id
         JOIN country c ON u.country_id=c.country_id JOIN addiction_level al ON h.level_id=al.level_id"""
df = pd.read_sql(text(sql), engine).dropna()
print("baris:", len(df))
print(df.dtypes.to_string())
num = ["age", "mental_health_score", "screen_time_before_sleep", "night_usage"]
print(df[num].agg(["min", "mean", "max"]).round(4).to_string())
print("contoh 3 baris pertama:\n", df.head(3).to_string(index=False))

print("\n=== 4. MODEL (sampel 50.000, split 80:20, seed 42) ===")
cat = ["gender", "country"]
d = df.sample(n=50000, random_state=42)
X, y = d[num + cat].copy(), d.addiction_level
for c_ in num:
    X[c_] = X[c_].astype("float64")          # paksa numerik biasa, buang kemungkinan tipe aneh
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def jalan(nama, **kw):
    m = Pipeline([("pre", ColumnTransformer([("n", "passthrough", num), ("c", OneHotEncoder(handle_unknown="ignore"), cat)])),
                  ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1, **kw))]).fit(Xtr, ytr)
    p = m.predict(Xte); r = classification_report(yte, p, output_dict=True, zero_division=0)
    print(f"{nama:34s} akurasi {accuracy_score(yte, p)*100:5.2f} | balanced {balanced_accuracy_score(yte, p)*100:5.2f} | recall High {r['High']['recall']*100:5.2f}")


jalan("default (seperti dashboard)")
jalan("parameter klasik dieksplisitkan", max_depth=None, min_samples_leaf=1, max_features="sqrt", min_samples_split=2)
print("\nSelesai. Salin semua teks di atas dan kirim ke chat.")