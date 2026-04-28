"""
main.py
=======
Entry point utama untuk pipeline Predictive Maintenance – AI4I 2020.

Pipeline:
    1. Pemuatan & validasi data
    2. Eksplorasi Data (EDA)
    3. Pemodelan statistik (regresi + distribusi probabilitas)
    4. Deteksi anomali multimetode

Penggunaan:
    python main.py                         # Muat dari UCI repository
    python main.py --csv data/ai4i2020.csv # Muat dari file lokal
    python main.py --output results/       # Simpan output ke folder tertentu
"""

import os
import sys
import argparse
import time

# Tambahkan root ke sys.path agar import src.* berjalan
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_maintenance_data
from src.eda import perform_eda
from src.stats_models import analyze_regression, analyze_probability
from src.anomaly_detection import detect_anomalies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance Pipeline – AI4I 2020 Dataset"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path ke file CSV lokal (opsional; default: unduh dari UCI)"
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="Direktori output untuk plot dan laporan (default: output/)"
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Lewati tahap EDA"
    )
    parser.add_argument(
        "--skip-stats", action="store_true",
        help="Lewati tahap pemodelan statistik"
    )
    parser.add_argument(
        "--skip-anomaly", action="store_true",
        help="Lewati tahap deteksi anomali"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("\n" + "╔" + "═"*58 + "╗")
    print("║   PREDICTIVE MAINTENANCE PIPELINE – AI4I 2020           ║")
    print("╚" + "═"*58 + "╝")
    t_start = time.time()

    print("\n[STEP 1/4] Pemuatan & Validasi Data")
    df = load_maintenance_data(csv_path=args.csv, validate=True)

    if not args.skip_eda:
        print("\n[STEP 2/4] Eksplorasi Data (EDA)")
        perform_eda(df, output_dir=args.output)
    else:
        print("\n[STEP 2/4] EDA dilewati (--skip-eda)")

    if not args.skip_stats:
        print("\n[STEP 3/4] Pemodelan Statistik")
        analyze_regression(df, output_dir=args.output)
        analyze_probability(df, output_dir=args.output)
    else:
        print("\n[STEP 3/4] Pemodelan statistik dilewati (--skip-stats)")

    if not args.skip_anomaly:
        print("\n[STEP 4/4] Deteksi Anomali")
        anomaly_results = detect_anomalies(df, output_dir=args.output)
    else:
        print("\n[STEP 4/4] Deteksi anomali dilewati (--skip-anomaly)")

    elapsed = time.time() - t_start
    print("\n" + "╔" + "═"*58 + "╗")
    print(f"║  Pipeline selesai dalam {elapsed:.1f} detik{' '*(32 - len(f'{elapsed:.1f}'))}║")
    print(f"║  Output disimpan di: {args.output:<37}║")
    print("╚" + "═"*58 + "╝\n")


if __name__ == "__main__":
    main()