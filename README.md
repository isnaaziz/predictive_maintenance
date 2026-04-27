# Predictive Maintenance Project - Matematika Statistika

Proyek ini bertujuan untuk memprediksi kegagalan pada perangkat IoT menggunakan pendekatan Matematika dan Statistika. Dataset yang digunakan adalah **AI4I 2020 Predictive Maintenance Dataset**.

## Materi RPAS yang Diimplementasikan

1. **Regresi-Korelasi**: Mengukur hubungan antara suhu lingkungan (`Air temperature [K]`) dengan suhu proses (`Process temperature [K]`).
2. **Distribusi Probabilitas**: Menganalisis probabilitas kegagalan menggunakan distribusi Normal atau Poisson.
3. **Data Deskriptif (EDA)**: Eksplorasi data sensor untuk melihat pola sebelum terjadi kegagalan.
4. **Z-Score Threshold**: Menentukan ambang batas (threshold) kegagalan berdasarkan penyebaran data.

## Alur Kerja

1. **Setup Environment**: Menggunakan virtual environment dan menginstal library yang dibutuhkan (`ucimlrepo`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`).
2. **Data Fetching**: Mengambil data langsung dari UCI Machine Learning Repository.
3. **Exploratory Data Analysis (EDA)**: Visualisasi distribusi data dan statistik deskriptif.
4. **Linear Regression**: Memprediksi kenaikan suhu proses berdasarkan suhu lingkungan atau beban kerja.
5. **Probability Analysis**: Menghitung probabilitas kegagalan dalam rentang waktu tertentu.
6. **Outlier Detection & Thresholding**: Menggunakan Z-score untuk mengidentifikasi anomali yang berpotensi menjadi kegagalan.

Lihat penjelasan mendalam mengenai metode matematika yang digunakan di: [DOKUMENTASI_METODE.md](file:///Users/macbookairm3/predictive_maintenance/DOKUMENTASI_METODE.md).

## Struktur Project (Modular)

Proyek ini dibagi menjadi beberapa modul dalam direktori `src/` untuk memudahkan pemeliharaan:

- `main.py`: Entry point utama untuk menjalankan seluruh analisis.
- `src/data_loader.py`: Mengambil data dari UCI Repository.
- `src/eda.py`: Statistik deskriptif dan visualisasi.
- `src/stats_models.py`: Implementasi Regresi dan Distribusi Probabilitas.
- `src/anomaly_detection.py`: Deteksi anomali menggunakan Z-score.
- `predictive_maintenance_analysis.ipynb`: Notebook interaktif yang mengimpor modul-modul di atas.

## Cara Menjalankan

1. Aktifkan virtual environment:
   ```bash
   source venv/bin/activate
   ```
2. Jalankan skrip utama:
   ```bash
   python main.py
   ```
3. Atau jalankan notebook:
   ```bash
   jupyter notebook predictive_maintenance_analysis.ipynb
   ```
