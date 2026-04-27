# Dokumentasi Metode dan Analisis Statistika
## Proyek: Prediksi Kegagalan Perangkat IoT (Predictive Maintenance)

Laporan ini merinci metodologi matematika dan statistika yang digunakan dalam proyek *Predictive Maintenance* menggunakan dataset AI4I 2020.

---

### 1. Eksplorasi Data Deskriptif (EDA)
**Tujuan**: Memahami karakteristik dasar data sensor sebelum pemodelan.

**Metode**:
Menggunakan metrik pemusatan dan penyebaran data:
- **Mean ($\mu$)**: Rata-rata nilai sensor.
- **Standar Deviasi ($\sigma$)**: Mengukur sebaran data terhadap rata-rata.
- **Kuartil**: Memahami distribusi data pada persentil 25%, 50% (Median), dan 75%.

**Hasil**: Kita mengidentifikasi rentang suhu operasional normal mesin, di mana suhu proses rata-rata berada di kisaran 310K dengan fluktuasi yang relatif stabil ($\sigma \approx 1.48$).

---

### 2. Analisis Korelasi dan Regresi Linier
**Tujuan**: Memprediksi kenaikan suhu proses berdasarkan beban kerja mesin (*Torque*).

**Metode Matematika**:
- **Korelasi Pearson ($r$)**: Mengukur kekuatan hubungan linier antara dua variabel.
  $$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
- **Regresi Linier Sederhana**: Membuat model prediksi suhu.
  $$\hat{y} = \beta_0 + \beta_1 x$$
  Dimana $x$ adalah *Torque* (Beban) dan $\hat{y}$ adalah estimasi suhu proses.

**Hasil**: Ditemukan bahwa korelasi antara *Torque* dan suhu proses sangat rendah ($r \approx -0.01$). Secara statistika, ini menunjukkan bahwa beban kerja sesaat tidak secara langsung menaikkan suhu dalam model linier pada dataset ini. Hal ini penting untuk menjelaskan bahwa kegagalan mungkin disebabkan oleh akumulasi panas atau faktor lain (seperti *Tool Wear*), bukan hanya beban kerja sesaat.

---

### 3. Distribusi Probabilitas
**Tujuan**: Menghitung peluang kejadian kritis dan kegagalan.

**Metode Matematika**:
- **Distribusi Normal (Kontinu)**: Digunakan untuk variabel suhu.
  $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$
  Kita menghitung $P(X > 312K)$ untuk mengetahui peluang mesin mencapai suhu kritis.
- **Distribusi Poisson (Diskrit)**: Digunakan untuk memodelkan jumlah kegagalan dalam interval tertentu (100 siklus kerja).
  $$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$
  Dimana $\lambda$ adalah rata-rata kegagalan per 100 siklus.

**Hasil**: 
- Peluang suhu melebihi 312K adalah sekitar **8.94%**.
- Peluang terjadi tepat 1 kegagalan dalam 100 siklus kerja adalah **11.43%**.

---

### 4. Deteksi Anomali dengan Z-Score
**Tujuan**: Menentukan ambang batas (*threshold*) kegagalan berdasarkan penyimpangan data.

**Metode Matematika**:
- **Z-Score**: Transformasi nilai sensor menjadi satuan standar deviasi.
  $$z = \frac{x - \mu}{\sigma}$$
- **Thresholding**: Data dianggap anomali jika $|z| > 3$ (berdasarkan Aturan Tiga Sigma / *Three-Sigma Rule*).

**Hasil**: Analisis pada *Rotational Speed* menunjukkan adanya 164 titik anomali. Dari titik-titik tersebut, ditemukan 33 kejadian kegagalan nyata. Ini membuktikan bahwa anomali statistik pada kecepatan putaran adalah indikator kuat terjadinya kegagalan perangkat IoT.

---

### Kesimpulan Akhir
Dengan menggabungkan analisis deskriptif, regresi, probabilitas, dan thresholding Z-score, kita dapat membangun sistem peringatan dini. Meskipun regresi menunjukkan hubungan beban-suhu yang lemah, pendekatan distribusi probabilitas dan Z-score terbukti efektif dalam memetakan risiko kegagalan operasional.
