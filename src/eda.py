"""
eda.py
Eksplorasi Data (EDA) komprehensif untuk dataset Predictive Maintenance.
Menghasilkan statistik deskriptif, distribusi, korelasi, dan distribusi
kegagalan per mode dan tipe mesin.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

from .data_loader import NUMERIC_COLS, TARGET_COL, FAILURE_MODES

PALETTE  = "viridis"
FIG_DPI  = 150
STYLE    = "whitegrid"


def perform_eda(df: pd.DataFrame, output_dir: str = ".") -> None:
    """
    Menjalankan rangkaian analisis EDA lengkap.

    Parameters
    ----------
    df         : DataFrame hasil load_maintenance_data().
    output_dir : Direktori penyimpanan gambar output.
    """
    sns.set_theme(style=STYLE, palette=PALETTE)
    print("\n" + "="*60)
    print("   EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)

    _print_descriptive_stats(df)
    _plot_feature_distributions(df, output_dir)
    _plot_correlation_heatmap(df, output_dir)
    _plot_failure_analysis(df, output_dir)
    _plot_failure_by_type(df, output_dir)
    _print_failure_mode_summary(df)


def _print_descriptive_stats(df: pd.DataFrame) -> None:
    """Statistik deskriptif numerik."""
    print("\n[1/4] Statistik Deskriptif\n")
    desc = df[NUMERIC_COLS].describe().T
    desc["CV (%)"] = (desc["std"] / desc["mean"] * 100).round(2)
    desc["Skewness"] = df[NUMERIC_COLS].skew().round(4)
    desc["Kurtosis"] = df[NUMERIC_COLS].kurt().round(4)
    print(desc.to_string())

    # Tingkat kegagalan keseluruhan
    total   = len(df)
    failure = df[TARGET_COL].sum()
    print(f"\nTotal sampel       : {total:,}")
    print(f"Total kegagalan    : {failure:,}  ({failure/total*100:.2f}%)")
    print(f"Rasio kelas (imbal): {(total-failure)/failure:.1f} : 1  (Normal : Gagal)")


def _plot_feature_distributions(df: pd.DataFrame, out: str) -> None:
    """Distribusi setiap fitur numerik dengan KDE dan statistik."""
    print("\n[2/4] Membuat plot distribusi fitur...")
    n = len(NUMERIC_COLS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_COLS):
        ax = axes[i]
        data = df[col].dropna()
        sns.histplot(data, kde=True, ax=ax, color=sns.color_palette(PALETTE, n)[i])
        ax.axvline(data.mean(),   color="red",    ls="--", lw=1.2, label=f"Mean={data.mean():.1f}")
        ax.axvline(data.median(), color="orange", ls=":",  lw=1.2, label=f"Median={data.median():.1f}")
        ax.set_title(col, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    # Hapus subplot kosong jika ada
    for j in range(len(NUMERIC_COLS), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Distribusi Fitur Sensor – AI4I 2020 Predictive Maintenance",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{out}/eda_feature_distributions.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"   → Tersimpan: {path}")


def _plot_correlation_heatmap(df: pd.DataFrame, out: str) -> None:
    """Heatmap korelasi Pearson antara seluruh fitur numerik + target."""
    print("\n[3/4] Membuat heatmap korelasi...")
    cols = NUMERIC_COLS + [TARGET_COL]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        annot_kws={"size": 9}, ax=ax
    )
    ax.set_title("Matriks Korelasi Pearson – Fitur Sensor & Kegagalan Mesin",
                 fontweight="bold", pad=12)
    plt.tight_layout()
    path = f"{out}/eda_correlation_heatmap.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"   → Tersimpan: {path}")


def _plot_failure_analysis(df: pd.DataFrame, out: str) -> None:
    """Distribusi fitur: Normal vs Failure (overlapping KDE)."""
    print("\n[4/4] Membuat plot analisis kegagalan...")
    normal  = df[df[TARGET_COL] == 0]
    failure = df[df[TARGET_COL] == 1]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_COLS):
        ax = axes[i]
        sns.kdeplot(normal[col],  ax=ax, label="Normal",  fill=True, alpha=0.4, color="steelblue")
        sns.kdeplot(failure[col], ax=ax, label="Gagal",   fill=True, alpha=0.4, color="tomato")
        ax.set_title(col, fontweight="bold")
        ax.legend(fontsize=8)

    for j in range(len(NUMERIC_COLS), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Distribusi Fitur: Normal vs Kegagalan Mesin",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{out}/eda_failure_comparison.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"   → Tersimpan: {path}")


def _plot_failure_by_type(df: pd.DataFrame, out: str) -> None:
    """Tingkat kegagalan per tipe mesin (L/M/H) dan per mode kegagalan."""
    if "Type" not in df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel kiri: tingkat kegagalan per tipe mesin
    rate_by_type = df.groupby("Type")[TARGET_COL].mean() * 100
    rate_by_type.sort_values().plot(kind="bar", ax=ax1, color=sns.color_palette(PALETTE, 3))
    ax1.set_title("Tingkat Kegagalan per Tipe Mesin (%)", fontweight="bold")
    ax1.set_xlabel("Tipe Mesin"); ax1.set_ylabel("Tingkat Kegagalan (%)")
    ax1.tick_params(axis="x", rotation=0)
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.2f}%",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom", fontsize=9)

    # Panel kanan: distribusi mode kegagalan
    mode_counts = df[FAILURE_MODES].sum().sort_values(ascending=False)
    mode_counts.plot(kind="bar", ax=ax2, color=sns.color_palette("rocket", len(FAILURE_MODES)))
    ax2.set_title("Distribusi Mode Kegagalan (TWF/HDF/PWF/OSF/RNF)", fontweight="bold")
    ax2.set_xlabel("Mode Kegagalan"); ax2.set_ylabel("Jumlah Kejadian")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = f"{out}/eda_failure_by_type_mode.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"   → Tersimpan: {path}")


def _print_failure_mode_summary(df: pd.DataFrame) -> None:
    """Ringkasan statistik per mode kegagalan."""
    print("\n── Ringkasan Mode Kegagalan ──")
    mode_labels = {
        "TWF": "Tool Wear Failure",
        "HDF": "Heat Dissipation Failure",
        "PWF": "Power Failure",
        "OSF": "Overstrain Failure",
        "RNF": "Random Failure",
    }
    for mode, label in mode_labels.items():
        if mode in df.columns:
            n = df[mode].sum()
            pct = n / len(df) * 100
            print(f"  {mode} ({label:30s}): {n:3d}  ({pct:.2f}%)")