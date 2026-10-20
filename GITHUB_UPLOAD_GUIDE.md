# Panduan Unggah Proyek ke GitHub

## Langkah 1 – Buat Repositori di GitHub

1. Buka [github.com](https://github.com) → Login ke akun Anda
2. Klik tombol **"New"** (hijau, kanan atas)
3. Isi formulir:
   - **Repository name**: `deforestation-driver-classification`
   - **Description**: `Klasifikasi Pemicu Deforestasi Tropis menggunakan Sentinel-1 & Sentinel-2 + Machine Learning (Dr. Bart Slachter, Wageningen University)`
   - **Visibility**: Public ✅
   - **JANGAN** centang "Add a README file" (sudah ada lokal)
4. Klik **"Create repository"**

---

## Langkah 2 – Inisialisasi Git Lokal

Buka terminal (CMD / PowerShell) di dalam folder proyek:

```powershell
cd "e:\Kuliah\Semester 7\Tugas\Sesi 6\deforestation-driver-classification"

# Inisialisasi Git
git init

# Set nama branch utama
git branch -M main

# Tambahkan semua file
git add .

# Commit pertama
git commit -m "Initial commit: Deforestation Driver Classification (Sentinel-1 & Sentinel-2 + ML)"
```

---

## Langkah 3 – Hubungkan ke GitHub & Push

Ganti `<username>` dengan username GitHub Anda:

```powershell
# Tambahkan remote origin
git remote add origin https://github.com/<username>/deforestation-driver-classification.git

# Push ke GitHub
git push -u origin main
```

---

## Langkah 4 – Verifikasi Struktur di GitHub

Setelah push, struktur repositori di GitHub akan terlihat seperti ini:

```
deforestation-driver-classification/
├── 📁 data/
│   └── satellite_deforestation_drivers.csv
├── 📁 src/
│   ├── dataset_generator.py
│   └── train_model.py
├── 📁 results/
│   ├── deforestation-confusion-matrix.png
│   ├── deforestation-metrics-comparison.png
│   └── deforestation-feature-importance.png
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Langkah 5 – Update README dengan Badge (Opsional)

Tambahkan badge di bagian atas README.md untuk tampilan profesional:

```markdown
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-Academic-green)
```

---

## Tips Tambahan

| Situasi | Perintah |
|---|---|
| Update file & push lagi | `git add . && git commit -m "Update" && git push` |
| Lihat status file | `git status` |
| Lihat riwayat commit | `git log --oneline` |
| Buat branch baru | `git checkout -b feature/nama-fitur` |
