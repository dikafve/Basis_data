import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("results.json"))
ORDER = ["Low", "Medium", "High"]
plt.rcParams.update({"font.size": 9, "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
LABEL = {"age": "Usia", "daily_usage_hours": "Pemakaian harian (jam)", "num_platforms_used": "Jumlah platform", "avg_session_minutes": "Durasi sesi (menit)",
         "night_usage": "Pemakaian malam (0/1)", "mental_health_score": "Skor kesehatan mental", "screen_time_before_sleep": "Screen time sebelum tidur (menit)",
         "gender": "Gender", "country": "Negara", "primary_platform": "Platform utama", "purpose": "Tujuan pemakaian"}
# confusion matrix dashboard (screenshot dashboard; urutan baris/kolom: High, Low, Medium) -> diubah ke Low, Medium, High
D = np.array([[996, 21, 559], [30, 1724, 744], [933, 1259, 3734]]); src = ["High", "Low", "Medium"]
idx = [src.index(c) for c in ORDER]; Dm = D[np.ix_(idx, idx)]
mats = [("(a) Konfigurasi C, dashboard (MySQL)", Dm), ("(b) Konfigurasi B, analisis CSV", np.array(R["models"]["B"]["confusion"]))]
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.7))
for a, (t, cm) in zip(ax, mats):
    a.imshow(cm, cmap="Blues"); a.set_xticks(range(3), ORDER); a.set_yticks(range(3), ORDER)
    a.set_xlabel("Prediksi"); a.set_ylabel("Aktual"); a.set_title(t, fontweight="bold", fontsize=9)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{cm[i,j]:,}".replace(",", "."), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    for s in a.spines.values(): s.set_visible(True)
fig.tight_layout(); fig.savefig("figs/fig3_confusion_v3.png"); plt.close(fig)
# importance konfigurasi C (impurity vs permutation)
imp = R["models"]["C"]["importance"]; pim = R["models"]["C"]["perm_importance"]
order = sorted(imp, key=lambda c: imp[c])
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
ax[0].barh([LABEL[c] for c in order], [imp[c] for c in order], color="#5b4b9a", edgecolor="black", linewidth=0.5)
ax[0].set_xlabel("Importance (penurunan impurity, diagregasi)"); ax[0].set_title("(a) Impurity-based", fontweight="bold")
ax[1].barh([LABEL[c] for c in order], [pim[c]["mean"] for c in order], xerr=[pim[c]["std"] for c in order], color="#2a9d8f", edgecolor="black", linewidth=0.5, capsize=2)
ax[1].axvline(0, color="black", linewidth=1); ax[1].set_xlabel("Penurunan akurasi saat diacak (data uji)"); ax[1].set_title("(b) Permutation", fontweight="bold")
fig.tight_layout(); fig.savefig("figs/fig4_importance_C.png"); plt.close(fig)
print("ok")
