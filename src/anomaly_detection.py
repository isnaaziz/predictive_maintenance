"""
anomaly_detection.py
====================
Deteksi anomali multimetode untuk sistem Predictive Maintenance.
Mengimplementasikan:
  1. Z-Score Thresholding (Three-Sigma Rule)
  2. Interquartile Range (IQR) Method
  3. Isolation Forest (ML-based)
  4. Matriks evaluasi deteksi terhadap label kegagalan nyata
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from typing import Dict, Optional

from .data_loader import NUMERIC_COLS, TARGET_COL

FIG_DPI = 150

def detect_anomalies(
    df: pd.DataFrame,
    column: str = "Rotational speed [rpm]",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    contamination: float = 0.034,   # ≈ tingkat kegagalan actual (3.39%)
    output_dir: str = ".",
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Menjalankan tiga metode deteksi anomali dan membandingkan hasilnya.

    Parameters
    ----------
    df              : DataFrame sumber.
    column          : Kolom utama untuk Z-Score / IQR (univariat).
    z_threshold     : Batas Z-score (default = 3 → Three-Sigma Rule).
    iqr_multiplier  : Pengali IQR (default = 1.5 → Tukey's fence).
    contamination   : Proporsi anomali yang diharapkan (Isolation Forest).
    output_dir      : Direktori simpan plot.
    random_state    : Seed RNG untuk reproduktibilitas.

    Returns
    -------
    dict  Kunci: "zscore", "iqr", "isoforest" → DataFrame baris anomali.
    """
    print("\n" + "="*60)
    print("   DETEKSI ANOMALI – MULTIMETODE")
    print("="*60)

    results = {}
    results["zscore"]    = _zscore_detection(df, column, z_threshold, output_dir)
    results["iqr"]       = _iqr_detection(df, column, iqr_multiplier)
    results["isoforest"] = _isolation_forest_detection(
        df, contamination, random_state, output_dir
    )
    _compare_methods(df, results, output_dir)
    return results


# METODE 1: Z-SCORE

def _zscore_detection(
    df: pd.DataFrame,
    column: str,
    threshold: float,
    out: str,
) -> pd.DataFrame:
    """Deteksi anomali univariat menggunakan Z-Score."""
    print(f"\n[Metode 1] Z-Score Thresholding – Kolom: {column}")
    print(f"  Threshold |Z| > {threshold}  (Three-Sigma Rule)")

    data     = df[column].dropna()
    mu, sigma = data.mean(), data.std()
    z_scores  = np.abs(stats.zscore(data))

    anomaly_mask = z_scores > threshold
    anomalies    = df.loc[data.index[anomaly_mask]].copy()
    anomalies["z_score"] = z_scores[anomaly_mask]

    pct = len(anomalies) / len(df) * 100
    print(f"  Total data            : {len(df):,}")
    print(f"  Anomali terdeteksi    : {len(anomalies):,}  ({pct:.2f}%)")

    if TARGET_COL in df.columns:
        tp = anomalies[TARGET_COL].sum()
        precision = tp / len(anomalies) if len(anomalies) > 0 else 0
        recall    = tp / df[TARGET_COL].sum() if df[TARGET_COL].sum() > 0 else 0
        print(f"  Kegagalan nyata dalam anomali: {tp} / {df[TARGET_COL].sum()}")
        print(f"  Precision (anomali → gagal)  : {precision:.4f}")
        print(f"  Recall  (gagal → anomali)    : {recall:.4f}")

    _plot_zscore(data, mu, sigma, threshold, column, out)
    return anomalies


def _plot_zscore(data, mu, sigma, threshold, col, out):
    """Visualisasi distribusi dengan band anomali Z-score."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.linspace(data.min(), data.max(), 500)
    pdf = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, "b-", lw=2, label="Distribusi Normal Fit")

    lower = mu - threshold * sigma
    upper = mu + threshold * sigma
    ax.fill_between(x, pdf, where=(x < lower), alpha=0.4, color="red", label=f"Zona Anomali |Z|>{threshold}")
    ax.fill_between(x, pdf, where=(x > upper), alpha=0.4, color="red")
    ax.axvline(lower, color="red", ls="--", lw=1.2)
    ax.axvline(upper, color="red", ls="--", lw=1.2)
    ax.axvline(mu,    color="green", ls=":", lw=1.5, label=f"Mean = {mu:.0f}")
    ax.set_xlabel(col); ax.set_ylabel("Densitas")
    ax.set_title(f"Z-Score Anomaly Detection – {col}", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = f"{out}/anomaly_zscore.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"\n   → Plot tersimpan: {path}")


# METODE 2: IQR

def _iqr_detection(
    df: pd.DataFrame,
    column: str,
    multiplier: float,
) -> pd.DataFrame:
    """Deteksi anomali berdasarkan Interquartile Range (Tukey's Fence)."""
    print(f"\n[Metode 2] IQR Method – Kolom: {column}")
    print(f"  Multiplier: {multiplier}  (Tukey's Fence)")

    data  = df[column].dropna()
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR    = Q3 - Q1
    lower  = Q1 - multiplier * IQR
    upper  = Q3 + multiplier * IQR

    anomaly_mask = (data < lower) | (data > upper)
    anomalies    = df.loc[data.index[anomaly_mask]].copy()
    pct = len(anomalies) / len(df) * 100

    print(f"  Q1 / Q3               : {Q1:.2f} / {Q3:.2f}")
    print(f"  IQR                   : {IQR:.2f}")
    print(f"  Batas bawah           : {lower:.2f}")
    print(f"  Batas atas            : {upper:.2f}")
    print(f"  Anomali terdeteksi    : {len(anomalies):,}  ({pct:.2f}%)")

    if TARGET_COL in df.columns:
        tp = anomalies[TARGET_COL].sum()
        precision = tp / len(anomalies) if len(anomalies) > 0 else 0
        print(f"  Kegagalan nyata dalam anomali: {tp}")
        print(f"  Precision              : {precision:.4f}")

    return anomalies


# METODE 3: ISOLATION FOREST

def _isolation_forest_detection(
    df: pd.DataFrame,
    contamination: float,
    random_state: int,
    out: str,
) -> pd.DataFrame:
    """
    Deteksi anomali multivariat menggunakan Isolation Forest.
    Menggunakan seluruh NUMERIC_COLS sebagai fitur.
    """
    print(f"\n[Metode 3] Isolation Forest (Multivariat)")
    print(f"  Fitur  : {NUMERIC_COLS}")
    print(f"  Contamination: {contamination:.4f}  ({contamination*100:.2f}%)")

    feature_data = df[NUMERIC_COLS].dropna()
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    pred = model.fit_predict(feature_data)
    scores = model.decision_function(feature_data)  # semakin negatif = lebih anomali

    anomaly_mask = pred == -1
    anomalies = df.loc[feature_data.index[anomaly_mask]].copy()
    anomalies["if_score"] = scores[anomaly_mask]
    pct = len(anomalies) / len(df) * 100

    print(f"  Total data            : {len(df):,}")
    print(f"  Anomali terdeteksi    : {len(anomalies):,}  ({pct:.2f}%)")

    if TARGET_COL in df.columns:
        true_labels = df.loc[feature_data.index, TARGET_COL].values
        pred_binary = (pred == -1).astype(int)
        tp = int(np.sum((pred_binary == 1) & (true_labels == 1)))
        precision = tp / len(anomalies) if len(anomalies) > 0 else 0
        recall    = tp / true_labels.sum() if true_labels.sum() > 0 else 0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        try:
            auc = roc_auc_score(true_labels, -scores)
            print(f"  AUC-ROC               : {auc:.4f}")
        except Exception:
            pass

        print(f"  Kegagalan nyata dalam anomali: {tp} / {int(true_labels.sum())}")
        print(f"  Precision              : {precision:.4f}")
        print(f"  Recall                 : {recall:.4f}")
        print(f"  F1-Score               : {f1:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(true_labels, pred_binary,
                                    target_names=["Normal", "Gagal"], zero_division=0))

        _plot_confusion_matrix(
            confusion_matrix(true_labels, pred_binary),
            ["Normal", "Gagal"], "Isolation Forest", out
        )

    return anomalies


def _plot_confusion_matrix(cm, labels, method_name, out):
    """Visualisasi confusion matrix deteksi anomali vs label nyata."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    ax.set_title(f"Confusion Matrix – {method_name}", fontweight="bold")
    plt.tight_layout()
    path = f"{out}/anomaly_confusion_matrix.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"   → Confusion matrix tersimpan: {path}")


# PERBANDINGAN METODE

def _compare_methods(
    df: pd.DataFrame,
    results: Dict[str, pd.DataFrame],
    out: str,
) -> None:
    """Ringkasan perbandingan ketiga metode deteksi."""
    print("\n" + "─"*60)
    print("   RINGKASAN PERBANDINGAN METODE DETEKSI ANOMALI")
    print("─"*60)

    total_failures = df[TARGET_COL].sum() if TARGET_COL in df.columns else "N/A"
    print(f"\n  Total kegagalan nyata  : {total_failures}")
    print(f"  Total sampel           : {len(df):,}")
    print(f"\n  {'Metode':<25}  {'Anomali':>8}  {'Gagal dlm anomali':>18}  {'Precision':>10}")
    print(f"  {'-'*65}")

    for method, anom_df in results.items():
        n_anom    = len(anom_df)
        if TARGET_COL in anom_df.columns:
            tp    = anom_df[TARGET_COL].sum()
            prec  = tp / n_anom if n_anom > 0 else 0
            print(f"  {method:<25}  {n_anom:>8,}  {tp:>18}  {prec:>10.4f}")
        else:
            print(f"  {method:<25}  {n_anom:>8,}  {'N/A':>18}  {'N/A':>10}")