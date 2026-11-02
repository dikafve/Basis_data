"""
Analisis GenZ Social Media Usage (1M baris) - versi CSV mentah.
Pipeline model sama dengan dashboard.py: Random Forest, class_weight="balanced",
split 80:20 stratified, one-hot untuk fitur kategorikal, fitur numerik tanpa scaling.

Jalankan:  python analysis.py  [path_csv]
Keluaran:  results.json + figur PNG di folder figs/
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "raw/genz_social_media_usage_1M.csv"
OUT = Path("figs"); OUT.mkdir(exist_ok=True)
TARGET = "addiction_level"
ORDER = ["Low", "Medium", "High"]
N_ROWS = 50000          # sama dengan nilai default slider dashboard
SEED = 42

NUM_ALL = ["age", "daily_usage_hours", "num_platforms_used", "avg_session_minutes",
           "night_usage", "mental_health_score", "screen_time_before_sleep"]
CAT_ALL = ["gender", "country", "primary_platform", "purpose"]
CONFIGS = {
    "A": dict(name="11 fitur (termasuk daily_usage_hours)",
              num=NUM_ALL, cat=CAT_ALL),
    "B": dict(name="10 fitur (tanpa daily_usage_hours)",
              num=[c for c in NUM_ALL if c != "daily_usage_hours"], cat=CAT_ALL),
    "C": dict(name="6 fitur awal (usia, mental health, screen time sebelum tidur, pemakaian malam, gender, negara)",
              num=["age", "mental_health_score", "screen_time_before_sleep", "night_usage"],
              cat=["gender", "country"]),
}

LABEL = {"age": "Usia", "daily_usage_hours": "Pemakaian harian (jam)",
         "num_platforms_used": "Jumlah platform", "avg_session_minutes": "Durasi sesi (menit)",
         "night_usage": "Pemakaian malam (0/1)", "mental_health_score": "Skor kesehatan mental",
         "screen_time_before_sleep": "Screen time sebelum tidur (menit)", "gender": "Gender",
         "country": "Negara", "primary_platform": "Platform utama", "purpose": "Tujuan pemakaian"}

plt.rcParams.update({"font.size": 9, "axes.linewidth": 1.0, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def build_model(num, cat, n_estimators=100, max_depth=None, seed=SEED):
    pre = ColumnTransformer([("num", "passthrough", num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                class_weight="balanced", random_state=seed, n_jobs=-1)
    return Pipeline([("pre", pre), ("rf", rf)])


def aggregate_importance(model, cat):
    names = model.named_steps["pre"].get_feature_names_out()
    imp = model.named_steps["rf"].feature_importances_
    agg = {}
    for n, v in zip(names, imp):
        base = n.split("__", 1)[1]
        orig = next((c for c in cat if base.startswith(c + "_")), base)
        agg[orig] = agg.get(orig, 0.0) + float(v)
    return dict(sorted(agg.items(), key=lambda kv: -kv[1]))


def run(df, cfg, seed=SEED, n_rows=N_ROWS, max_depth=None, perm=False):
    num, cat = cfg["num"], cfg["cat"]
    d = df.sample(n=n_rows, random_state=seed)
    X, y = d[num + cat], d[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    m = build_model(num, cat, max_depth=max_depth, seed=seed).fit(Xtr, ytr)
    p = m.predict(Xte)
    rep = classification_report(yte, p, output_dict=True, zero_division=0)
    res = {
        "n_train": len(Xtr), "n_test": len(Xte),
        "accuracy": accuracy_score(yte, p),
        "baseline": float(yte.value_counts(normalize=True).max()),
        "balanced_accuracy": balanced_accuracy_score(yte, p),
        "macro_f1": f1_score(yte, p, average="macro"),
        "per_class": {k: {m_: float(rep[k][m_]) for m_ in ("precision", "recall", "f1-score", "support")}
                      for k in ORDER},
        "confusion": confusion_matrix(yte, p, labels=ORDER).tolist(),
        "importance": aggregate_importance(m, cat),
    }
    if perm:
        pi = permutation_importance(m, Xte, yte, n_repeats=5, random_state=seed,
                                    scoring="accuracy", n_jobs=1)
        res["perm_importance"] = {c: {"mean": float(a), "std": float(s)}
                                  for c, a, s in zip(X.columns, pi.importances_mean, pi.importances_std)}
    return res, m


def cramers_v(a, b):
    tab = pd.crosstab(a, b)
    chi2, p, dof, _ = chi2_contingency(tab)
    n = tab.values.sum()
    v = np.sqrt((chi2 / n) / (min(tab.shape) - 1))
    return {"chi2": float(chi2), "dof": int(dof), "p": float(p), "cramers_v": float(v)}


def main():
    df = pd.read_csv(CSV)
    R = {"env": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "n_rows_sample": N_ROWS, "seed": SEED}}

    # ---- 1. kualitas data & deskriptif
    R["data"] = {
        "shape": list(df.shape), "nulls": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "level_counts": df[TARGET].value_counts().to_dict(),
        "level_pct": (df[TARGET].value_counts(normalize=True) * 100).round(2).to_dict(),
        "numeric_describe": df[NUM_ALL].describe().round(4).to_dict(),
        "gender_pct": (df.gender.value_counts(normalize=True) * 100).round(1).to_dict(),
        "country_pct": (df.country.value_counts(normalize=True) * 100).round(1).to_dict(),
        "platform_pct": (df.primary_platform.value_counts(normalize=True) * 100).round(1).to_dict(),
        "purpose_pct": (df.purpose.value_counts(normalize=True) * 100).round(1).to_dict(),
    }
    h = df.daily_usage_hours
    rule = np.where(h <= 2.5, "Low", np.where(h <= 5.0, "Medium", "High"))
    R["label_rule"] = {
        "match_rate": float((rule == df[TARGET]).mean()), "mismatch": int((rule != df[TARGET]).sum()),
        "range_by_level": df.groupby(TARGET).daily_usage_hours.agg(["min", "max", "mean"]).round(4).to_dict("index"),
    }
    R["level_profile_mean"] = df.groupby(TARGET)[NUM_ALL].mean().round(4).to_dict("index")
    R["level_profile_std"] = df.groupby(TARGET)[NUM_ALL].std().round(4).to_dict("index")
    R["corr"] = df[NUM_ALL].corr().round(4).to_dict()
    sl, ic = np.polyfit(df.daily_usage_hours, df.mental_health_score, 1)
    R["mental_vs_daily"] = {"corr": float(df[["daily_usage_hours", "mental_health_score"]].corr().iloc[0, 1]),
                            "slope": float(sl), "intercept": float(ic)}
    R["assoc_with_level"] = {c: cramers_v(df[c], df[TARGET]) for c in ["age", "gender", "country", "primary_platform", "purpose", "night_usage", "num_platforms_used"]}
    R["mean_daily_by_platform"] = df.groupby("primary_platform").daily_usage_hours.mean().round(3).to_dict()
    R["mean_daily_by_purpose"] = df.groupby("purpose").daily_usage_hours.mean().round(3).to_dict()
    R["mean_daily_by_country"] = df.groupby("country").daily_usage_hours.mean().round(3).to_dict()

    # ---- 2. tren per usia (RQ3)
    trend = {}
    for m in ["daily_usage_hours", "avg_session_minutes", "num_platforms_used", "mental_health_score",
              "screen_time_before_sleep", "night_usage"]:
        t = df.groupby("age")[m].mean()
        slope, intercept = np.polyfit(t.index, t.values, 1)
        r2 = float(np.corrcoef(t.index, t.values)[0, 1] ** 2)
        trend[m] = {"slope": float(slope), "intercept": float(intercept), "r2": r2,
                    "min_mean": float(t.min()), "max_mean": float(t.max()),
                    "overall_mean": float(df[m].mean()),
                    "range_pct_of_mean": float((t.max() - t.min()) / df[m].mean() * 100)}
    R["trend"] = trend

    # ---- 3. model utama (seed 42) untuk tiga konfigurasi
    models = {}
    R["models"] = {}
    for k, cfg in CONFIGS.items():
        res, m = run(df, cfg, perm=(k in ("B", "C")))
        res["name"] = cfg["name"]
        R["models"][k] = res
        models[k] = m
        print(k, round(res["accuracy"], 4), round(res["balanced_accuracy"], 4), round(res["macro_f1"], 4))

    # ---- 4. stabilitas lintas seed (sampel + split + RF berubah bersama)
    seeds = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    R["stability"] = {}
    for k, cfg in CONFIGS.items():
        rows = []
        for s in seeds:
            r, _ = run(df, cfg, seed=s)
            rows.append([r["accuracy"], r["baseline"], r["balanced_accuracy"], r["macro_f1"],
                         r["per_class"]["High"]["recall"], r["per_class"]["High"]["f1-score"]])
        a = np.array(rows)
        names = ["accuracy", "baseline", "balanced_accuracy", "macro_f1", "recall_high", "f1_high"]
        R["stability"][k] = {n: {"mean": float(a[:, i].mean()), "sd": float(a[:, i].std(ddof=1)),
                                 "min": float(a[:, i].min()), "max": float(a[:, i].max())}
                             for i, n in enumerate(names)}
        R["stability"][k]["n_runs"] = len(seeds)

    # ---- 5. sensitivitas kedalaman pohon (konfigurasi B)
    R["depth_sensitivity_B"] = {}
    for dpt in [5, 10, 15, 20, None]:
        r, _ = run(df, CONFIGS["B"], max_depth=dpt)
        R["depth_sensitivity_B"][str(dpt) if dpt else "None"] = {
            "accuracy": r["accuracy"], "balanced_accuracy": r["balanced_accuracy"],
            "macro_f1": r["macro_f1"], "recall_high": r["per_class"]["High"]["recall"]}

    json.dump(R, open("results.json", "w"), indent=2, default=float)

    # ================= FIGUR =================
    colors = {"Low": "#2a9d8f", "Medium": "#e9c46a", "High": "#c0392b"}

    # Fig 2: distribusi daily_usage_hours per level + skor mental health per level
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for lv in ORDER:
        ax[0].hist(df.loc[df[TARGET] == lv, "daily_usage_hours"], bins=60, alpha=0.85,
                   color=colors[lv], label=lv, edgecolor="black", linewidth=0.3)
    for x in (2.5, 5.0):
        ax[0].axvline(x, color="black", linestyle="--", linewidth=1)
    ax[0].set_xlabel("Pemakaian harian (jam)"); ax[0].set_ylabel("Jumlah baris"); ax[0].set_title("(a) Pemakaian harian menurut level", fontweight="bold")
    ax[0].legend(title="Level kecanduan", frameon=False)
    data = [df.loc[df[TARGET] == lv, "mental_health_score"] for lv in ORDER]
    bp = ax[1].boxplot(data, patch_artist=True, showfliers=False, medianprops=dict(color="black", linewidth=1.2))
    ax[1].set_xticks([1, 2, 3]); ax[1].set_xticklabels(ORDER)   # kompatibel dengan matplotlib lama dan baru
    for patch, lv in zip(bp["boxes"], ORDER):
        patch.set_facecolor(colors[lv]); patch.set_linewidth(1)
    ax[1].set_xlabel("Level kecanduan"); ax[1].set_ylabel("Skor kesehatan mental"); ax[1].set_title("(b) Skor kesehatan mental menurut level", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig2_distribusi.png"); plt.close(fig)

    # Fig 3: confusion matrix A (kiri) dan B (kanan)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
    for a, k in zip(ax, ["A", "B"]):
        cm = np.array(R["models"][k]["confusion"])
        a.imshow(cm, cmap="Blues")
        a.set_xticks(range(3), ORDER); a.set_yticks(range(3), ORDER)
        a.set_xlabel("Prediksi"); a.set_ylabel("Aktual")
        a.set_title(f"({'a' if k=='A' else 'b'}) Konfigurasi {k}", fontweight="bold")
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{cm[i,j]:,}".replace(",", "."), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        for s in a.spines.values(): s.set_visible(True)
    fig.tight_layout(); fig.savefig(OUT / "fig3_confusion.png"); plt.close(fig)

    # Fig 4: importance konfigurasi B (impurity vs permutation)
    imp = R["models"]["B"]["importance"]; pim = R["models"]["B"]["perm_importance"]
    order = sorted(imp, key=lambda c: imp[c])
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    ax[0].barh([LABEL[c] for c in order], [imp[c] for c in order], color="#5b4b9a", edgecolor="black", linewidth=0.5)
    ax[0].set_xlabel("Importance (penurunan impurity, diagregasi)"); ax[0].set_title("(a) Impurity-based", fontweight="bold")
    ax[1].barh([LABEL[c] for c in order], [pim[c]["mean"] for c in order], xerr=[pim[c]["std"] for c in order],
               color="#2a9d8f", edgecolor="black", linewidth=0.5, capsize=2)
    ax[1].axvline(0, color="black", linewidth=1)
    ax[1].set_xlabel("Penurunan akurasi saat diacak (data uji)"); ax[1].set_title("(b) Permutation", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "fig4_importance.png"); plt.close(fig)

    # Fig 5: tren rata-rata per usia
    metrics = ["daily_usage_hours", "mental_health_score", "screen_time_before_sleep"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    for a, m in zip(ax, metrics):
        t = df.groupby("age")[m].mean()
        sl, ic = np.polyfit(t.index, t.values, 1)
        a.scatter(t.index, t.values, color="black", s=18, zorder=3)
        a.plot(t.index, sl * t.index + ic, color="#5b4b9a", linewidth=1.5)
        mu = df[m].mean(); a.set_ylim(mu * 0.9, mu * 1.1)
        a.set_xlabel("Usia"); a.set_title(LABEL[m], fontsize=8.5, fontweight="bold")
        a.text(0.04, 0.06, f"slope = {sl:+.5f}\nR² = {trend[m]['r2']:.3f}", transform=a.transAxes, fontsize=7.5)
    ax[0].set_ylabel("Rata-rata per usia")
    fig.tight_layout(); fig.savefig(OUT / "fig5_tren_usia.png"); plt.close(fig)
    print("selesai")


if __name__ == "__main__":
    main()