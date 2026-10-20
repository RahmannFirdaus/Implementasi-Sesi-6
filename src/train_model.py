"""
train_model.py
--------------
Pelatihan & Evaluasi Model Klasifikasi Pemicu Deforestasi Tropis
menggunakan citra multi-sensor Sentinel-1 & Sentinel-2.

Model yang dibandingkan:
  1. Random Forest  (baseline, dengan class_weight='balanced')
  2. Neural Network - MLP (MLPClassifier)

Metrik Evaluasi Utama:
  - Overall Accuracy : proporsi total prediksi benar
  - Macro F1 Score   : rata-rata F1 per kelas (sensitif terhadap kelas minoritas)

Metodologi: Dr. Bart Slachter, Wageningen University (2024)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend untuk server/CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


# ─── Konfigurasi ────────────────────────────────────────────────────────────────

DATA_PATH    = os.path.join('data', 'satellite_deforestation_drivers.csv')
RESULTS_DIR  = 'results'
RANDOM_STATE = 42
TEST_SIZE    = 0.25

FEATURE_COLS = [
    'S1_VV', 'S1_VH',
    'S2_B2_Blue', 'S2_B3_Green', 'S2_B4_Red', 'S2_B8_NIR',
    'NDVI', 'SAR_VH_VV_Ratio',
]

CLASS_LABELS = [
    'Agriculture\nSmall',
    'Agriculture\nLarge',
    'Logging\n& Roads',
    'Mining',
    'Fire &\nNatural',
]

# Palet warna konsisten untuk setiap kelas
CLASS_COLORS = {
    'Agriculture_Small': '#4CAF50',
    'Agriculture_Large': '#8BC34A',
    'Logging_Roads':     '#FF9800',
    'Mining':            '#F44336',
    'Fire_Natural':      '#9C27B0',
}

# ─── Helpers ─────────────────────────────────────────────────────────────────────

def load_data(path: str):
    """Memuat dataset, memisahkan fitur & label, serta memvalidasi kolom."""
    if not os.path.exists(path):
        sys.exit(
            f"[ERROR] Dataset tidak ditemukan di '{path}'.\n"
            "Jalankan terlebih dahulu: python src/dataset_generator.py"
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + ['Driver_Class'] if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Kolom hilang dari dataset: {missing}")
    X = df[FEATURE_COLS].values
    y = df['Driver_Class'].values
    return X, y, df['Driver_Class'].unique().tolist()


def print_section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ─── Visualisasi ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    model_name: str,
    save_path: str,
) -> None:
    """Menyimpan heatmap confusion matrix ke file PNG."""
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        linewidths=0.5,
        linecolor='#333355',
        ax=ax,
        cbar_kws={'shrink': 0.8},
    )

    ax.set_title(
        f'Confusion Matrix – {model_name}\n'
        'Atribusi Pemicu Deforestasi Tropis (Sentinel-1 & Sentinel-2)',
        color='white', fontsize=13, fontweight='bold', pad=16,
    )
    ax.set_xlabel('Predicted Label', color='#ccccdd', fontsize=11, labelpad=10)
    ax.set_ylabel('True Label',      color='#ccccdd', fontsize=11, labelpad=10)
    ax.tick_params(colors='#aaaacc', labelsize=9)

    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [SAVED] Confusion matrix disimpan: {save_path}")


def plot_metrics_comparison(
    model_names: list,
    accuracies: list,
    f1_scores: list,
    save_path: str,
) -> None:
    """Menyimpan bar chart perbandingan Accuracy vs Macro F1 Score."""
    x     = np.arange(len(model_names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#0f0f1e')
    ax.set_facecolor('#161625')

    bars_acc = ax.bar(
        x - width / 2, accuracies, width,
        label='Overall Accuracy',
        color='#00BCD4', alpha=0.88, zorder=3,
        edgecolor='#00E5FF', linewidth=0.8,
    )
    bars_f1 = ax.bar(
        x + width / 2, f1_scores, width,
        label='Macro F1 Score',
        color='#FF6B6B', alpha=0.88, zorder=3,
        edgecolor='#FF8A80', linewidth=0.8,
    )

    # Anotasi nilai di atas bar
    for bar in bars_acc:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{bar.get_height():.3f}',
            ha='center', va='bottom', color='#00E5FF',
            fontsize=10, fontweight='bold',
        )
    for bar in bars_f1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{bar.get_height():.3f}',
            ha='center', va='bottom', color='#FF8A80',
            fontsize=10, fontweight='bold',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, color='#ddddee', fontsize=12)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel('Score', color='#aaaacc', fontsize=12, labelpad=10)
    ax.set_title(
        'Perbandingan Performa Model\nOverall Accuracy vs Macro F1 Score',
        color='white', fontsize=14, fontweight='bold', pad=18,
    )
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='#555577', zorder=0)
    ax.tick_params(axis='y', colors='#aaaacc')
    ax.tick_params(axis='x', bottom=False)

    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    legend = ax.legend(
        facecolor='#1f1f35', edgecolor='#444466',
        labelcolor='#ddddee', fontsize=10,
    )

    # Anotasi penjelasan metrik
    ax.text(
        0.99, 0.06,
        '* Macro F1 penting untuk mendeteksi kelas minoritas\n'
        '  (Mining & Fire) pada distribusi imbalanced.',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=8, color='#8888aa', style='italic',
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [SAVED] Grafik metrik disimpan: {save_path}")


def plot_feature_importance(rf_model, save_path: str) -> None:
    """Menyimpan bar chart feature importance dari Random Forest."""
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_feat = [FEATURE_COLS[i] for i in indices]
    sorted_imp  = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f0f1e')
    ax.set_facecolor('#161625')

    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(sorted_feat)))
    bars = ax.barh(sorted_feat[::-1], sorted_imp[::-1], color=colors[::-1],
                   edgecolor='#333355', linewidth=0.5, zorder=3)

    ax.set_xlabel('Importance Score', color='#aaaacc', fontsize=11)
    ax.set_title(
        'Feature Importance – Random Forest\nFitur Sentinel-1 SAR & Sentinel-2 Optik',
        color='white', fontsize=13, fontweight='bold', pad=14,
    )
    ax.tick_params(colors='#aaaacc', labelsize=9)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#555577', zorder=0)

    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [SAVED] Feature importance disimpan: {save_path}")


# ─── Pipeline Utama ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Muat Data ─────────────────────────────────────────────────────────────
    print_section("1. Memuat Dataset")
    X, y, class_list = load_data(DATA_PATH)
    print(f"  Jumlah sampel : {len(X):,}")
    print(f"  Jumlah fitur  : {X.shape[1]}")
    print(f"  Kelas         : {class_list}")

    # Encode label ke integer untuk kebutuhan tertentu (CM ordering)
    le = LabelEncoder()
    le.fit(sorted(class_list))   # urutan alfabet agar konsisten

    # ── 2. Split Train / Test ─────────────────────────────────────────────────────
    print_section("2. Pembagian Data Train / Test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train : {len(X_train):,} sampel ({100*(1-TEST_SIZE):.0f}%)")
    print(f"  Test  : {len(X_test):,} sampel ({100*TEST_SIZE:.0f}%)")

    # ── 3. Normalisasi (diperlukan oleh MLP) ─────────────────────────────────────
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── 4. Model 1: Random Forest ─────────────────────────────────────────────────
    print_section("3. Training – Random Forest (class_weight='balanced')")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=4,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)          # RF tidak memerlukan normalisasi
    y_pred_rf = rf.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf  = f1_score(y_test, y_pred_rf, average='macro')
    cm_rf  = confusion_matrix(y_test, y_pred_rf, labels=le.classes_)

    print(f"  [OK] Accuracy   : {acc_rf:.4f}  ({acc_rf*100:.2f}%)")
    print(f"  [OK] Macro F1   : {f1_rf:.4f}  ({f1_rf*100:.2f}%)")

    # ── 5. Model 2: Neural Network (MLP) ──────────────────────────────────────────
    print_section("4. Training – Neural Network (MLP)")
    # Encode y ke integer untuk kompatibilitas MLPClassifier (sklearn bug dengan early_stopping + string labels)
    y_train_int = le.transform(y_train)
    y_test_int  = le.transform(y_test)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=64,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )
    mlp.fit(X_train_sc, y_train_int)
    y_pred_mlp_int = mlp.predict(X_test_sc)
    y_pred_mlp = le.inverse_transform(y_pred_mlp_int)  # decode ke string class names

    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    f1_mlp  = f1_score(y_test, y_pred_mlp, average='macro')
    cm_mlp  = confusion_matrix(y_test, y_pred_mlp, labels=le.classes_)

    print(f"  [OK] Accuracy   : {acc_mlp:.4f}  ({acc_mlp*100:.2f}%)")
    print(f"  [OK] Macro F1   : {f1_mlp:.4f}  ({f1_mlp*100:.2f}%)")    

    # ── 6. Laporan Lengkap ────────────────────────────────────────────────────────
    print_section("5. Laporan Klasifikasi Detail")
    print("\n--- Random Forest ---")
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
    print("\n--- Neural Network (MLP) ---")
    print(classification_report(y_test, y_pred_mlp, target_names=le.classes_))

    # ── 7. Ringkasan Perbandingan ─────────────────────────────────────────────────
    print_section("6. Ringkasan Perbandingan Model")
    header = f"{'Model':<25} {'Accuracy':>12} {'Macro F1':>12}"
    print(header)
    print("-" * len(header))
    print(f"{'Random Forest':<25} {acc_rf:>12.4f} {f1_rf:>12.4f}")
    print(f"{'Neural Network (MLP)':<25} {acc_mlp:>12.4f} {f1_mlp:>12.4f}")
    winner = "Random Forest" if f1_rf >= f1_mlp else "Neural Network (MLP)"
    print(f"\n  [WINNER] Model terbaik (Macro F1): {winner}")

    # ── 8. Simpan Visualisasi ─────────────────────────────────────────────────────
    print_section("7. Menyimpan Visualisasi")

    plot_confusion_matrix(
        cm_rf, le.classes_, model_name='Random Forest',
        save_path=os.path.join(RESULTS_DIR, 'deforestation-confusion-matrix.png'),
    )
    plot_metrics_comparison(
        model_names=['Random Forest', 'Neural Network\n(MLP)'],
        accuracies=[acc_rf,  acc_mlp],
        f1_scores =[f1_rf,   f1_mlp],
        save_path=os.path.join(RESULTS_DIR, 'deforestation-metrics-comparison.png'),
    )
    plot_feature_importance(
        rf_model=rf,
        save_path=os.path.join(RESULTS_DIR, 'deforestation-feature-importance.png'),
    )

    print("\n[DONE] Semua hasil berhasil disimpan di folder 'results/'")


if __name__ == '__main__':
    main()
