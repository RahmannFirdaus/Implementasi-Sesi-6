"""
dataset_generator.py
--------------------
Pembuat dataset multi-sensor sintetis yang mensimulasikan:
  - Sentinel-1: nilai backscatter radar (VV & VH) yang mampu menembus awan
  - Sentinel-2: reflektansi optik (Blue, Green, Red, NIR)
  - Indeks turunan: NDVI (kesehatan vegetasi) & SAR VH/VV Ratio

Distribusi kelas merepresentasikan ketidakseimbangan nyata (class imbalance)
yang ditemukan di kawasan tropis (Metodologi: Dr. Bart Slachter, Wageningen University).

Referensi:
  - Slachter, B. (2024). Identifying Direct Drivers of Tropical Deforestation
    Using Multi-Sensor Satellite Imagery & Machine Learning. Wageningen University.
"""

import numpy as np
import pandas as pd
import os


# ─── Konfigurasi ────────────────────────────────────────────────────────────────

CLASSES = [
    'Agriculture_Small',   # Pertanian Skala Kecil  → kelas mayoritas
    'Agriculture_Large',   # Pertanian Skala Besar
    'Logging_Roads',       # Penebangan & Jalan Pembuka
    'Mining',              # Pertambangan           → kelas minoritas
    'Fire_Natural',        # Kebakaran & Gangguan Alam → kelas minoritas
]

# Probabilitas sesuai distribusi deforestasi tropis aktual
CLASS_PROBS = [0.45, 0.25, 0.15, 0.08, 0.07]

# Statistik tiap kelas: (mean, std) berdasarkan nilai lapangan referensi
CLASS_PARAMS = {
    #                   VV             VH           B2-Blue        B3-Green       B4-Red         B8-NIR
    'Agriculture_Small': {
        'VV': (-11.0, 1.5), 'VH': (-17.0, 1.5),
        'B2': (0.05, 0.01), 'B3': (0.08, 0.01), 'B4': (0.09, 0.02), 'B8': (0.25, 0.04),
    },
    'Agriculture_Large': {
        'VV': (-13.0, 1.2), 'VH': (-19.0, 1.2),
        'B2': (0.07, 0.01), 'B3': (0.10, 0.02), 'B4': (0.14, 0.02), 'B8': (0.18, 0.03),
    },
    'Logging_Roads': {
        'VV': (-9.5, 1.8),  'VH': (-15.5, 1.8),
        'B2': (0.06, 0.01), 'B3': (0.09, 0.02), 'B4': (0.11, 0.02), 'B8': (0.28, 0.04),
    },
    'Mining': {
        'VV': (-14.5, 2.0), 'VH': (-21.0, 2.0),
        'B2': (0.12, 0.02), 'B3': (0.15, 0.02), 'B4': (0.18, 0.03), 'B8': (0.12, 0.02),
    },
    'Fire_Natural': {
        'VV': (-8.0, 2.2),  'VH': (-14.0, 2.2),
        'B2': (0.04, 0.01), 'B3': (0.06, 0.01), 'B4': (0.13, 0.03), 'B8': (0.15, 0.03),
    },
}

OUTPUT_COLUMNS = [
    'S1_VV', 'S1_VH',
    'S2_B2_Blue', 'S2_B3_Green', 'S2_B4_Red', 'S2_B8_NIR',
    'NDVI', 'SAR_VH_VV_Ratio',
    'Driver_Class',
]


# ─── Fungsi Utama ────────────────────────────────────────────────────────────────

def generate_multisensor_dataset(num_samples: int = 3000, random_seed: int = 42) -> pd.DataFrame:
    """
    Menghasilkan DataFrame dataset multi-sensor dengan distribusi kelas imbalanced.

    Parameter
    ----------
    num_samples : int
        Jumlah sampel total yang akan dibuat (default 3.000).
    random_seed : int
        Seed untuk reproducibility (default 42).

    Kembalian
    ---------
    pd.DataFrame
        DataFrame dengan kolom fitur Sentinel-1, Sentinel-2, indeks turunan,
        dan label kelas pemicu deforestasi.
    """
    np.random.seed(random_seed)

    # Sampling label kelas sesuai probabilitas distribusi tropis
    y_labels = np.random.choice(CLASSES, size=num_samples, p=CLASS_PROBS)

    rows = []
    for label in y_labels:
        p = CLASS_PARAMS[label]

        # Sentinel-1 SAR backscatter (dB)
        vv = np.random.normal(*p['VV'])
        vh = np.random.normal(*p['VH'])

        # Sentinel-2 surface reflectance (0–1)
        b2 = np.random.normal(*p['B2'])
        b3 = np.random.normal(*p['B3'])
        b4 = np.random.normal(*p['B4'])
        b8 = np.random.normal(*p['B8'])

        # Indeks turunan
        ndvi = (b8 - b4) / (b8 + b4 + 1e-6)           # Normalized Difference Vegetation Index
        sar_ratio = vh - vv                              # SAR Cross-polarization Ratio (dB)

        rows.append([vv, vh, b2, b3, b4, b8, ndvi, sar_ratio, label])

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return df


# ─── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', 'satellite_deforestation_drivers.csv')

    print("=" * 60)
    print("  Deforestation Driver Dataset Generator")
    print("  Metodologi: Dr. Bart Slachter (Wageningen University)")
    print("=" * 60)

    df = generate_multisensor_dataset(num_samples=3000)
    df.to_csv(output_path, index=False)

    print(f"\n[OK] Dataset berhasil dibuat: {output_path}")
    print(f"   Jumlah sampel  : {len(df):,}")
    print(f"   Jumlah fitur   : {len(df.columns) - 1}")
    print(f"\n[INFO] Distribusi kelas:\n")
    print(df['Driver_Class'].value_counts().to_string())
    print("\n[INFO] 5 Baris pertama:\n")
    print(df.head().to_string())
