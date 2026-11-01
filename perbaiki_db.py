"""
perbaiki_db.py - audit dan perbaikan ISI tabel genz_erd yang sudah ada (skema 8 tabel TIDAK diubah).

Masalah yang diperbaiki: tabel health_behavior (dan user_session) berisi subset yang bias dari dataset
sumber (hanya profil Male, kelas Low tidak ada, High 3,8% padahal 15,9% di sumber). Skrip ini mengisi
ulang kedua tabel itu dengan sampel acak dari CSV sumber, memakai ID dari tabel lookup yang ADA di DB-mu.

Aman dipakai karena:
  1. Default-nya DRY-RUN: hanya membaca dan membuat laporan. Tidak ada yang diubah.
  2. --apply membuat tabel cadangan lebih dulu (health_behavior_bak_<tanggal>, dst.), lalu mengganti
     isi dalam satu transaksi. Kalau ada error, perubahan dibatalkan.
  3. --rollback mengembalikan isi dari tabel cadangan.
  4. Laporan sebelum/sesudah disimpan ke laporan_db.txt agar perubahannya bisa ditunjukkan ke dosen.

Contoh:
  python perbaiki_db.py --raw genz_social_media_usage_1M.csv                 # audit saja
  python perbaiki_db.py --raw genz_social_media_usage_1M.csv --apply         # perbaiki
  python perbaiki_db.py --rollback 20261101                                  # kembalikan
Koneksi default sama dengan dashboard.py. Ganti dengan --url "mysql+mysqlconnector://user:pass@host:3306/db".
"""
import argparse
import datetime as dt
import sys

import pandas as pd
from sqlalchemy import create_engine, inspect, text

DEFAULT_URL = "mysql+mysqlconnector://root:@127.0.0.1:3306/genz_erd"
REF = {"level": {"Medium": 58.98, "Low": 25.12, "High": 15.89},          # proporsi pada 1.000.000 baris sumber
       "gender": {"Female": 48.0, "Male": 48.0, "Other": 4.0}}
LOG = []


def say(msg=""):
    print(msg)
    LOG.append(str(msg))


def read(engine, sql):
    return pd.read_sql(text(sql), engine)


def audit(engine, title):
    say(f"\n=== {title} ===")
    for t in ("user", "gender", "country", "addiction_level", "platform", "usage_purpose", "health_behavior", "user_session"):
        n = read(engine, f"SELECT COUNT(*) AS n FROM {t}").n[0]
        say(f"  {t:16s} {n:>9,} baris")
    h = read(engine, """SELECT al.level_name AS level, g.gender_name AS gender, h.user_id
                        FROM health_behavior h JOIN user u ON h.user_id=u.user_id
                        JOIN gender g ON u.gender_id=g.gender_id JOIN addiction_level al ON h.level_id=al.level_id""")
    lv = (h.level.value_counts(normalize=True) * 100).round(2)
    gd = (h.gender.value_counts(normalize=True) * 100).round(2)
    say("  Level kecanduan di health_behavior (%), acuan sumber dalam kurung:")
    for k, ref in REF["level"].items():
        v = lv.get(k, 0.0); flag = "" if abs(v - ref) < 2 else "   <-- menyimpang dari sumber"
        say(f"    {k:7s} {v:6.2f}  ({ref}){flag}")
    say("  Gender di health_behavior (%), acuan sumber dalam kurung:")
    for k, ref in REF["gender"].items():
        v = gd.get(k, 0.0); flag = "" if abs(v - ref) < 2 else "   <-- menyimpang dari sumber"
        say(f"    {k:7s} {v:6.2f}  ({ref}){flag}")
    say(f"  Profil user unik di health_behavior: {h.user_id.nunique()} dari {read(engine, 'SELECT COUNT(*) AS n FROM user').n[0]}")
    return lv, gd


def quick_model(engine):
    """Cek cepat memakai pipeline dashboard.py (6 fitur, sampel 50.000, split 80:20, RF default)."""
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    df = read(engine, """SELECT u.age, g.gender_name AS gender, c.country_name AS country, h.mental_health_score,
                         h.night_usage, h.screen_time_before_sleep, al.level_name AS addiction_level
                         FROM health_behavior h JOIN user u ON h.user_id=u.user_id JOIN gender g ON u.gender_id=g.gender_id
                         JOIN country c ON u.country_id=c.country_id JOIN addiction_level al ON h.level_id=al.level_id""").dropna()
    num = ["age", "mental_health_score", "screen_time_before_sleep", "night_usage"]; cat = ["gender", "country"]
    d = df.sample(n=min(50000, len(df)), random_state=42)
    X, y = d[num + cat], d.addiction_level
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = Pipeline([("pre", ColumnTransformer([("n", "passthrough", num), ("c", OneHotEncoder(handle_unknown="ignore"), cat)])),
                  ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1))]).fit(Xtr, ytr)
    p = m.predict(Xte); rep = classification_report(yte, p, output_dict=True, zero_division=0)
    minority = yte.value_counts().idxmin()
    say(f"  Cek model (pipeline dashboard): akurasi {accuracy_score(yte, p)*100:.2f}% | baseline {yte.value_counts(normalize=True).max()*100:.2f}% | "
        f"balanced {balanced_accuracy_score(yte, p)*100:.2f}% | recall {minority} {rep[minority]['recall']*100:.2f}%")


def build_rows(engine, raw_path, n_health, n_session, seed):
    raw = pd.read_csv(raw_path)
    maps = {
        "gender": read(engine, "SELECT gender_id, gender_name FROM gender").set_index("gender_name").gender_id,
        "country": read(engine, "SELECT country_id, country_name FROM country").set_index("country_name").country_id,
        "level": read(engine, "SELECT level_id, level_name FROM addiction_level").set_index("level_name").level_id,
        "platform": read(engine, "SELECT platform_id, platform_name FROM platform").set_index("platform_name").platform_id,
        "purpose": read(engine, "SELECT purpose_id, purpose_name FROM usage_purpose").set_index("purpose_name").purpose_id,
    }
    user = read(engine, "SELECT user_id, age, gender_id, country_id FROM user")
    raw["gender_id"] = raw.gender.map(maps["gender"]); raw["country_id"] = raw.country.map(maps["country"])
    raw["level_id"] = raw.addiction_level.map(maps["level"]); raw["platform_id"] = raw.primary_platform.map(maps["platform"])
    raw["purpose_id"] = raw.purpose.map(maps["purpose"])
    raw = raw.merge(user, on=["age", "gender_id", "country_id"], how="left")
    need = ["gender_id", "country_id", "level_id", "platform_id", "purpose_id", "user_id"]
    bad = raw[need].isna().any(axis=1).sum()
    if bad:
        sys.exit(f"BERHENTI: {bad} baris CSV tidak bisa dipetakan ke tabel lookup di DB. Tidak ada yang diubah.")
    raw[need] = raw[need].astype(int)
    h = raw.sample(n=n_health, random_state=seed).sort_index()
    health = pd.DataFrame({"record_id": range(1, len(h) + 1), "user_id": h.user_id.values, "level_id": h.level_id.values,
                           "mental_health_score": h.mental_health_score.round(5).values, "night_usage": h.night_usage.values,
                           "screen_time_before_sleep": h.screen_time_before_sleep.round(4).values})
    s = raw.sample(n=n_session, random_state=seed).sort_index()
    session = pd.DataFrame({"session_id": range(1, len(s) + 1), "user_id": s.user_id.values, "platform_id": s.platform_id.values,
                            "purpose_id": s.purpose_id.values, "daily_usage_hours": s.daily_usage_hours.round(5).values,
                            "num_platforms_used": s.num_platforms_used.values, "avg_session_minutes": s.avg_session_minutes.round(4).values})
    return {"health_behavior": health, "user_session": session}


def check_columns(engine, new):
    insp = inspect(engine)
    for t, df in new.items():
        cols = [c["name"] for c in insp.get_columns(t)]
        if set(cols) != set(df.columns):
            sys.exit(f"BERHENTI: kolom tabel {t} di DB {cols} berbeda dari data baru {list(df.columns)}. Tidak ada yang diubah.")


def count(engine, table):
    with engine.connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar())


def backup(engine, new, tag):
    """Buat tabel cadangan untuk SEMUA tabel dulu (sebelum ada data yang dihapus), per 500 baris
    supaya tidak melewati max_allowed_packet MySQL."""
    for t in new:
        bak = f"{t}_bak_{tag}"
        src_n = count(engine, t)
        if inspect(engine).has_table(bak):
            bak_n = count(engine, bak)
            if bak_n == src_n and src_n > 0:
                sys.exit(f"BERHENTI: cadangan {bak} sudah ada dan lengkap ({bak_n:,} baris). "
                         f"Hapus tabel itu atau pakai --tag lain. Tidak ada data yang diubah.")
            say(f"  cadangan {bak} sisa percobaan sebelumnya tidak lengkap ({bak_n:,} baris), dibuat ulang")
            with engine.begin() as conn:
                conn.execute(text(f"DROP TABLE {bak}"))
        old = read(engine, f"SELECT * FROM {t}")
        old.to_sql(bak, engine, if_exists="fail", index=False, chunksize=500, method="multi")
        if count(engine, bak) != len(old):
            sys.exit(f"BERHENTI: cadangan {bak} tidak lengkap. Tidak ada data yang diubah.")
        say(f"  cadangan: {bak} ({len(old):,} baris)")


def apply(engine, new, tag):
    check_columns(engine, new)
    backup(engine, new, tag)                                       # tahap 1: semua cadangan dulu
    with engine.begin() as conn:                                   # tahap 2: ganti isi dalam satu transaksi
        for t, df in new.items():
            conn.execute(text(f"DELETE FROM {t}"))
            df.to_sql(t, conn, if_exists="append", index=False, chunksize=500, method="multi")
            n = int(conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar())
            if n != len(df):
                raise RuntimeError(f"jumlah baris {t} tidak cocok ({n} vs {len(df)})")
            say(f"  {t}: diisi ulang {n:,} baris")


def rollback(engine, tag):
    with engine.begin() as conn:
        for t in ("health_behavior", "user_session"):
            bak = pd.read_sql(text(f"SELECT * FROM {t}_bak_{tag}"), conn)
            conn.execute(text(f"DELETE FROM {t}"))
            bak.to_sql(t, conn, if_exists="append", index=False, chunksize=500, method="multi")
            say(f"  {t}: dikembalikan {len(bak):,} baris dari {t}_bak_{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL); ap.add_argument("--raw")
    ap.add_argument("--apply", action="store_true"); ap.add_argument("--rollback", metavar="TANGGAL")
    ap.add_argument("--n-health", type=int, default=100_000); ap.add_argument("--n-session", type=int, default=110_000)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--tag", default=dt.date.today().strftime("%Y%m%d"), help="akhiran nama tabel cadangan")
    a = ap.parse_args()
    engine = create_engine(a.url)
    say(f"perbaiki_db.py | {dt.datetime.now():%Y-%m-%d %H:%M} | mode: " +
        ("ROLLBACK" if a.rollback else "APPLY" if a.apply else "DRY-RUN (tidak ada yang diubah)"))
    if a.rollback:
        rollback(engine, a.rollback); audit(engine, "SESUDAH ROLLBACK")
    else:
        if not a.raw: sys.exit("butuh --raw <csv sumber>")
        audit(engine, "KONDISI SAAT INI")
        if not a.no_model: quick_model(engine)
        new = build_rows(engine, a.raw, a.n_health, a.n_session, a.seed)
        check_columns(engine, new)
        say("\nRencana: isi ulang health_behavior dan user_session dengan sampel acak dari CSV sumber (skema tidak berubah).")
        if a.apply:
            apply(engine, new, a.tag)
            audit(engine, "SESUDAH PERBAIKAN")
            if not a.no_model: quick_model(engine)
        else:
            say("DRY-RUN selesai. Tambahkan --apply untuk menjalankan perbaikan (cadangan dibuat otomatis).")
    open("laporan_db.txt", "w", encoding="utf-8").write("\n".join(LOG))
    print("\nLaporan disimpan di laporan_db.txt")


if __name__ == "__main__":
    main()