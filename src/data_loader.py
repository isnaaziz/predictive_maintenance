"""
data_loader.py
==============
Modul pemuat data untuk proyek Predictive Maintenance.
Mendukung pemuatan dari file lokal (CSV) maupun UCI ML Repository.

Dataset: AI4I 2020 Predictive Maintenance (UCI ML Repository ID=601)
Kolom fitur  : Air temperature [K], Process temperature [K],
               Rotational speed [rpm], Torque [Nm], Tool wear [min], Type
Kolom target : Machine failure, TWF, HDF, PWF, OSF, RNF

CATATAN KOMPATIBILITAS
----------------------
UCI ML Repository mengembalikan nama kolom TANPA satuan, misalnya:
  "Air temperature"   (bukan "Air temperature [K]")
  "Rotational speed"  (bukan "Rotational speed [rpm]")

File CSV dari Kaggle / unduhan langsung menggunakan nama DENGAN satuan.
Fungsi _normalise_columns() menangani kedua format secara otomatis.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]
TARGET_COL    = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NUMERIC_COLS  = [c for c in FEATURE_COLS if c != "Type"]
DATASET_ID    = 601

_COLUMN_NORMALISATION_MAP = {
    "Air temperature":     "Air temperature [K]",
    "Process temperature": "Process temperature [K]",
    "Rotational speed":    "Rotational speed [rpm]",
    "Torque":              "Torque [Nm]",
    "Tool wear":           "Tool wear [min]",
    "Type":            "Type",
    "Machine failure": "Machine failure",
    "TWF": "TWF", "HDF": "HDF",
    "PWF": "PWF", "OSF": "OSF", "RNF": "RNF",
}


def load_maintenance_data(
    csv_path=None,
    dataset_id=DATASET_ID,
    validate=True,
):
    """
    Memuat dataset Predictive Maintenance dari file CSV lokal atau UCI Repository.

    Parameters
    ----------
    csv_path   : Path ke file CSV lokal. Jika None, unduh dari UCI.
    dataset_id : ID dataset UCI ML Repository (default 601).
    validate   : Jika True, jalankan validasi skema dasar setelah muat.

    Returns
    -------
    pd.DataFrame  DataFrame bersih siap analisis.
    """
    if csv_path and os.path.exists(csv_path):
        df = _load_from_csv(csv_path)
    else:
        df = _load_from_uci(dataset_id)

    df = _clean_columns(df)
    df = _normalise_columns(df)
    _fix_dtypes(df)

    if validate:
        _validate_schema(df)

    print(f"[DataLoader] Dataset berhasil dimuat: {df.shape[0]:,} baris x {df.shape[1]} kolom")
    print(f"[DataLoader] Kolom aktif: {df.columns.tolist()}")
    print(f"[DataLoader] Distribusi kegagalan:\n{df[TARGET_COL].value_counts().to_string()}\n")
    return df


def _load_from_csv(path):
    """Muat dari CSV lokal."""
    print(f"[DataLoader] Memuat dari file lokal: {path}")
    return pd.read_csv(path)


def _load_from_uci(dataset_id):
    """Unduh dari UCI ML Repository menggunakan ucimlrepo."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "Package 'ucimlrepo' tidak ditemukan. "
            "Install dengan: pip install ucimlrepo"
        ) from exc

    print(f"[DataLoader] Mengunduh dataset UCI ID={dataset_id}...")
    dataset = fetch_ucirepo(id=dataset_id)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    print(f"[DataLoader] Kolom UCI asli: {df.columns.tolist()}")
    return df


def _clean_columns(df):
    """Strip whitespace nama kolom dan hapus kolom identifier serta baris kosong."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    drop_cols = [c for c in df.columns if c in ("UDI", "Product ID")]
    if drop_cols:
        print(f"[DataLoader] Menghapus kolom identifier: {drop_cols}")
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _normalise_columns(df):
    """
    Rename kolom UCI (tanpa satuan) ke format canonical (dengan satuan).

    Contoh transformasi:
        "Air temperature"   -> "Air temperature [K]"
        "Rotational speed"  -> "Rotational speed [rpm]"
        "Torque"            -> "Torque [Nm]"

    Kolom yang sudah menggunakan format canonical dibiarkan tanpa perubahan.
    """
    rename_map = {
        col: _COLUMN_NORMALISATION_MAP[col]
        for col in df.columns
        if col in _COLUMN_NORMALISATION_MAP
    }
    if rename_map:
        renamed = {k: v for k, v in rename_map.items() if k != v}
        if renamed:
            print(f"[DataLoader] Normalisasi nama kolom (UCI -> canonical): {renamed}")
        df = df.rename(columns=rename_map)
    return df


def _fix_dtypes(df):
    """Koreksi tipe data in-place: numerik untuk sensor, int untuk target."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [TARGET_COL] + FAILURE_MODES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def _validate_schema(df):
    """Validasi keberadaan kolom wajib dan proporsi nilai null."""
    required = NUMERIC_COLS + [TARGET_COL]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        available = df.columns.tolist()
        raise ValueError(
            f"[DataLoader] Kolom wajib tidak ditemukan: {missing}\n"
            f"  Kolom tersedia di DataFrame: {available}\n"
            f"  Kemungkinan penyebab:\n"
            f"    - File CSV menggunakan header berbeda dari standar AI4I 2020\n"
            f"    - UCI Repository mengembalikan format baru yang belum dipetakan\n"
            f"  Solusi: tambahkan mapping di _COLUMN_NORMALISATION_MAP di data_loader.py"
        )

    null_pct  = df[required].isnull().mean() * 100
    high_null = null_pct[null_pct > 5]
    if not high_null.empty:
        print(f"[DataLoader] Kolom dengan nilai null >5%:\n{high_null.to_string()}")

    print("[DataLoader] Validasi skema: OK")