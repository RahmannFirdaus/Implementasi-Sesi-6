# Deforestation Driver Classification

> **Studi Kasus Lengkap**: *Atribusi Pemicu Langsung Deforestasi Tropis Menggunakan Citra Satelit Multi-Sensor (Sentinel-1 & Sentinel-2) dan Machine Learning*


---

## 📌 Latar Belakang

Pemantauan deforestasi secara spasial sering kali hanya mendeteksi **hilangnya tutupan pohon** tanpa mengidentifikasi **penyebabnya**. Studi kasus ini mengikuti metodologi Dr. Bart Slachter yang menggabungkan:

| Sensor | Jenis Data | Keunggulan |
|---|---|---|
| **Sentinel-1 SAR** | Radar (VV & VH backscatter, dB) | Menembus awan & kondisi cuaca buruk |
| **Sentinel-2 MSI** | Optik (Blue, Green, Red, NIR, reflektansi 0–1) | Informasi kesehatan vegetasi (NDVI) |

Tujuan: mengklasifikasikan **5 Pemicu Langsung (*Direct Drivers*) Deforestasi Tropis**:

| No | Kelas | Deskripsi |
|---|---|---|
| 1 | `Agriculture_Small` | Pertanian Skala Kecil (subsisten) |
| 2 | `Agriculture_Large` | Pertanian Skala Besar (komersial: kelapa sawit, kedelai) |
| 3 | `Logging_Roads` | Penebangan & Jalan Pembuka Hutan |
| 4 | `Mining` | Pertambangan (emas, mineral) |
| 5 | `Fire_Natural` | Kebakaran Hutan & Gangguan Alam |

---

## 🧠 Metodologi & Tantangan Kunci

### Class Imbalance (Ketidakseimbangan Kelas)

Distribusi kelas mencerminkan kondisi nyata di daerah tropis:

```
Agriculture_Small  ████████████████████████████ 45%
Agriculture_Large  ███████████████ 25%
Logging_Roads      █████████ 15%
Mining             █████ 8%
Fire_Natural       ████ 7%
```

**Implikasi**: Model yang hanya mengejar *Overall Accuracy* akan bias ke kelas mayoritas dan **gagal mendeteksi** pertambangan dan kebakaran. Oleh karena itu, **Macro F1 Score** digunakan sebagai metrik utama karena menghitung rata-rata F1 per kelas secara setara, tanpa memandang ukuran kelas.

### Fitur yang Digunakan

| Fitur | Sumber | Deskripsi |
|---|---|---|
| `S1_VV` | Sentinel-1 | Backscatter polarisasi vertikal-vertikal (dB) |
| `S1_VH` | Sentinel-1 | Backscatter polarisasi vertikal-horizontal (dB) |
| `S2_B2_Blue` | Sentinel-2 | Reflektansi Band Biru (0,49 µm) |
| `S2_B3_Green` | Sentinel-2 | Reflektansi Band Hijau (0,56 µm) |
| `S2_B4_Red` | Sentinel-2 | Reflektansi Band Merah (0,665 µm) |
| `S2_B8_NIR` | Sentinel-2 | Reflektansi Near-Infrared (0,842 µm) |
| `NDVI` | Turunan S2 | (NIR-Red)/(NIR+Red) - Indeks Kesehatan Vegetasi |
| `SAR_VH_VV_Ratio` | Turunan S1 | VH - VV (dB) - Indeks Struktur Vegetasi/Kekasaran |

---

## 📊 Hasil Eksperimen

| Model | Overall Accuracy | Macro F1 Score |
|---|---|---|
| **Random Forest** (n=100, balanced) | ~91.8% | ~92.1% |
| **Neural Network** (MLP 128-64-32, Adam) | ~85.8% | ~84.6% |

> **Interpretasi**: Random Forest mengungguli MLP pada dataset ini. Perbedaan yang relatif kecil antara Accuracy dan Macro F1 pada Random Forest menunjukkan model mampu mendeteksi kelas minoritas (Mining, Fire) dengan baik berkat parameter `class_weight='balanced'`.

---

## 📐 Alasan Pemilihan Metrik

| Metrik | Kelebihan | Keterbatasan |
|---|---|---|
| **Overall Accuracy** | Mudah dipahami; menunjukkan performa global | Menyesatkan pada distribusi imbalanced |
| **Macro F1 Score** ✅ | Adil untuk semua kelas; penting untuk kelas minoritas | Tidak mempertimbangkan frekuensi kelas |
| Weighted F1 | Mempertimbangkan frekuensi kelas | Bias ke kelas mayoritas seperti Accuracy |

---

## 🔬 Referensi

- Slachter, B. (2024). *Identifying Direct Drivers of Tropical Deforestation Using Multi-Sensor Satellite Imagery & Machine Learning*. Wageningen University & Research.
- Hansen, M.C., et al. (2013). High-Resolution Global Maps of 21st-Century Forest Cover Change. *Science*, 342(6160), 850-853.
- Sentinel Hub. (2024). *Sentinel-1 SAR GRD & Sentinel-2 L2A Documentation*. ESA Copernicus.
- Scikit-learn Developers. (2024). *sklearn.metrics: Classification metrics*. scikit-learn.org.


