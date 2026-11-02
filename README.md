# Basis_data: Prediksi Level Kecanduan Media Sosial Gen Z (Random Forest)

Tugas Basis Data (D4 Teknik Informatika, ULBI) dan kode pendukung manuskrip jurnal.

## Data
Gen-Z Social Media Usage Dataset (Stable Space, 2026, Kaggle, lisensi MIT):
https://www.kaggle.com/datasets/sharmajicoder/gen-z-social-media-usage-dataset
Data bersifat **sintetis** (1.000.000 baris, 12 kolom). Berkas CSV tidak disimpan di repositori ini (117 MB), unduh sendiri dari Kaggle.

## Isi repositori
| Berkas | Fungsi |
|---|---|
| `restore_db.py` | Mengimpor 8 tabel CSV ke MySQL `genz_erd` |
| `perbaiki_db.py` | Audit dan perbaikan isi tabel `health_behavior` dan `user_session` dari CSV sumber (dry-run secara default, `--apply` membuat tabel cadangan dulu, `--rollback` mengembalikan) |
| `diagnosa_db.py` | Hanya membaca database: versi library, tipe kolom, dan cek model |
| `dashboard.py` | Dashboard Streamlit (data dari MySQL) |
| `analysis.py` | Analisis dari CSV: audit label, konfigurasi A/B/C, kestabilan, kedalaman pohon, permutation importance, tren usia. Hasil di `results.json` dan `figs/` |
| `figs_v3.py` | Gambar confusion matrix dan importance untuk naskah |
| `laporan_db_sebelum.txt`, `laporan_db.txt` | Laporan audit database sebelum dan sesudah perbaikan |
| `genz.ipynb`, `prediksigenz.ipynb` | Notebook eksplorasi awal |

## Cara menjalankan
```
pip install -r requirements.txt
python perbaiki_db.py --raw "<path>/genz_social_media_usage_1M.csv"           # audit saja
python perbaiki_db.py --raw "<path>/genz_social_media_usage_1M.csv" --apply   # perbaiki (cadangan otomatis)
streamlit run dashboard.py
python analysis.py "<path>/genz_social_media_usage_1M.csv"
python figs_v3.py
```
Koneksi database bawaan: `mysql+mysqlconnector://root:@127.0.0.1:3306/genz_erd`.

## Catatan penting
- Hasil Random Forest **berbeda antar versi scikit-learn** (contoh pada konfigurasi C: akurasi 66,97% di 1.8.0 dan 63,11% di 1.9.0). Semua angka di naskah memakai scikit-learn 1.9.0.
- Pada data ini `addiction_level` adalah versi terkategori dari `daily_usage_hours` (Low <= 2,5 jam, Medium <= 5,0 jam, High > 5,0 jam). Model yang memuat `daily_usage_hours` bocor (akurasi mendekati 100%), sehingga dashboard memakai enam fitur tanpa variabel itu.
