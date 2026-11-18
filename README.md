# 🍱 MBG Insight — Dashboard UAS Pengenalan Pola

Dashboard interaktif untuk mengeksplorasi dataset Program Makan Bergizi Gratis (MBG),
mengevaluasi model Random Forest, mencoba prediksi, dan memantau progres pengerjaan tugas.

## Cara menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Browser otomatis terbuka di `http://localhost:8501`.
Jika belum terbuka, buka manual alamat tersebut.

## Isi dashboard (6 tab)

| Tab | Fungsi |
|-----|--------|
| 📊 Overview | KPI ringkas (jumlah paket, rata-rata plate waste, suhu tiba, kepatuhan BGN) + distribusi kelas target |
| 🔍 Eksplorasi Data | Histogram, boxplot, crosstab heatmap, korelasi, scatter — semuanya reaktif ke filter sidebar |
| 🤖 Performa Model | Latih Random Forest (skenario biner / multikelas) → confusion matrix, classification report, feature importance |
| 🔮 Prediksi | Form input 5 fitur → prediksi pemicu food waste + probabilitas tiap kelas |
| 📋 Data Explorer | Tabel data, pencarian, download CSV hasil filter, dan narasi rationale per paket |
| ✅ Progress | Checklist 10 tahapan tugas + catatan, tersimpan permanen di `progress.json` |

## Filter global (sidebar)

Provinsi · Klaster regional · Kategori penerima · Jenis protein · Tingkat kepatuhan BGN.
Semua tab mengikuti filter yang aktif.

## Struktur proyek

```
app.py                          # entry point + seluruh tab
requirements.txt
progress.json                   # dibuat otomatis saat progres pertama kali disimpan
modules/
├── loader.py                   # load CSV (cached) + fungsi filter
├── eda.py                      # semua grafik Plotly
├── modeling.py                 # pipeline Random Forest (sklearn)
└── tracker.py                  # baca/tulis progress.json
indonesia_mbg_nationwide_enriched_2026.csv
UASPengPol.ipynb                # notebook analisis (EDA + modeling)
```

## Deploy ke internet (Streamlit Community Cloud)

Gratis dan permanen, URL berakhiran `*.streamlit.app`.

1. **Buat repo GitHub kosong** di <https://github.com/new>
   - Nama bebas, misal `mbg-insight`
   - Visibilitas **Public**
   - **Jangan** centang "Add a README file", ".gitignore", atau "license" (harus benar-benar kosong)
2. **Push kode** dari folder proyek ini:
   ```bash
   git remote add origin https://github.com/USERNAME/mbg-insight.git
   git push -u origin main
   ```
   Push pertama akan membuka browser untuk login GitHub — cukup izinkan sekali.
3. **Deploy** di <https://share.streamlit.io>
   - Sign in with GitHub
   - **New app** → pilih repo `mbg-insight`, branch `main`, **Main file path: `app.py`**
   - **Advanced settings** → pilih **Python 3.12**
   - **Deploy** — proses build pertama ± 2–4 menit
4. URL final: `https://USERNAME-mbg-insight.streamlit.app` → bagikan ke teman-teman.

### Catatan hosting

- File CSV 17 MB ikut ter-upload, jadi tidak perlu database.
- Tab **Progress** memakai `st.session_state` saat online (penyimpanan file di cloud
  bersifat sementara), sehingga progres tidak bertahan setelah app restart.
- Dataset tidak berubah, jadi model tidak perlu dilatih ulang saat deploy.

## Catatan penting untuk presentasi

Dataset ini **sintetis dan deterministik**: label `primary_food_waste_driver` hampir
sepenuhnya bisa diturunkan dari fitur (akurasi ~100% bahkan untuk 5 kelas). Dashboard
sengaja menampilkan banner peringatan pada tab Performa Model agar hasil ini tidak
disalahartikan sebagai bukti performa pada data nyata.
