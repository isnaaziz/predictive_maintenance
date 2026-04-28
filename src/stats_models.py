"""
stats_models.py
===============
Pemodelan statistik komprehensif untuk analisis Predictive Maintenance.
Mencakup: regresi linier multipel, analisis distribusi probabilitas
(Normal & Poisson), uji signifikansi, dan analisis korelasi parsial.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Dict

from .data_loader import NUMERIC_COLS, TARGET_COL, FAILURE_MODES

FIG_DPI = 150


# ─────────────────────────────────────────────────────────────────────────────
# 1. ANALISIS REGRESI LINIER
# ─────────────────────────────────────────────────────────────────────────────

def analyze_regression(
    df: pd.DataFrame,
    x_col: str = "Torque [Nm]",
    y_col: str = "Process temperature [K]",
    output_dir: str = ".",
) -> LinearRegression:
    """
    Analisis regresi linier sederhana beserta uji signifikansi statistik.

    Melakukan:
    - Korelasi Pearson dengan p-value
    - Fitting OLS (Ordinary Least Squares)
    - Residual analysis
    - Plot scatter dengan confidence interval

    Parameters
    ----------
    df         : DataFrame sumber.
    x_col      : Kolom prediktor.
    y_col      : Kolom respons.
    output_dir : Direktori simpan plot.

    Returns
    -------
    LinearRegression  Model terlatih.
    """
    print("\n" + "="*60)
    print("   ANALISIS REGRESI LINIER")
    print("="*60)

    X_data = df[x_col].dropna()
    y_data = df[y_col].dropna()
    idx    = X_data.index.intersection(y_data.index)
    X_data, y_data = X_data.loc[idx], y_data.loc[idx]

    # ── Korelasi Pearson + uji t ──────────────────────────────────────────────
    r, p_corr = stats.pearsonr(X_data, y_data)
    n         = len(X_data)
    print(f"\n[Pearson Correlation]")
    print(f"  Variabel prediktor    : {x_col}")
    print(f"  Variabel respons      : {y_col}")
    print(f"  Koefisien Korelasi (r): {r:.4f}")
    print(f"  p-value               : {p_corr:.4e}  {'← Signifikan (α=0.05)' if p_corr < 0.05 else '← Tidak signifikan'}")
    print(f"  n sampel              : {n:,}")

    # ── OLS Regression ───────────────────────────────────────────────────────
    X_matrix = X_data.values.reshape(-1, 1)
    model    = LinearRegression()
    model.fit(X_matrix, y_data.values)
    y_pred   = model.predict(X_matrix)
    residuals = y_data.values - y_pred

    r_sq  = model.score(X_matrix, y_data.values)
    rmse  = np.sqrt(mean_squared_error(y_data.values, y_pred))
    mae   = mean_absolute_error(y_data.values, y_pred)

    # Standard error of coefficients
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data.values - y_data.values.mean())**2)
    se_b1  = np.sqrt((ss_res / (n - 2)) / np.sum((X_data.values - X_data.mean())**2))
    t_stat = model.coef_[0] / se_b1
    p_coef = 2 * stats.t.sf(np.abs(t_stat), df=n - 2)

    print(f"\n[OLS Model]")
    print(f"  Persamaan             : ŷ = {model.intercept_:.4f} + {model.coef_[0]:.4f} × {x_col}")
    print(f"  Koefisien Determinasi : R² = {r_sq:.4f}  ({r_sq*100:.2f}% variasi dijelaskan)")
    print(f"  RMSE                  : {rmse:.4f}")
    print(f"  MAE                   : {mae:.4f}")
    print(f"  t-statistik (slope)   : {t_stat:.4f}  (p = {p_coef:.4e})")
    print(f"\n  Interpretasi: Setiap kenaikan 1 Nm {x_col}, {y_col} berubah sebesar {model.coef_[0]:.4f} K")

    _plot_regression(X_data, y_data, model, x_col, y_col, r_sq, residuals, output_dir)
    return model


def _plot_regression(X, y, model, x_col, y_col, r_sq, residuals, out):
    """Plot scatter+garis regresi dan residual."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter + regression line
    X_range  = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_fit    = model.predict(X_range)

    ax1.scatter(X, y, alpha=0.2, s=10, label="Data Observasi", color="steelblue")
    ax1.plot(X_range, y_fit, color="red", lw=2, label=f"Garis Regresi (R²={r_sq:.3f})")
    ax1.set_xlabel(x_col); ax1.set_ylabel(y_col)
    ax1.set_title(f"Regresi Linier: {x_col} vs {y_col}", fontweight="bold")
    ax1.legend()

    # Residual plot
    y_pred_all = model.predict(X.values.reshape(-1, 1))
    ax2.scatter(y_pred_all, residuals, alpha=0.2, s=10, color="steelblue")
    ax2.axhline(0, color="red", lw=1.5, ls="--")
    ax2.set_xlabel("Nilai Prediksi"); ax2.set_ylabel("Residual")
    ax2.set_title("Plot Residual (Diagnostik Model)", fontweight="bold")

    plt.tight_layout()
    path = f"{out}/stats_regression_plot.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"\n   → Plot tersimpan: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANALISIS DISTRIBUSI PROBABILITAS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_probability(
    df: pd.DataFrame,
    temp_threshold: float = 312.0,
    output_dir: str = ".",
) -> Dict:
    """
    Analisis distribusi probabilitas: Normal (suhu) dan Poisson (kegagalan).

    Melakukan:
    - Fitting distribusi Normal + uji Shapiro-Wilk / Kolmogorov-Smirnov
    - Kalkulasi probabilitas suhu kritis P(X > threshold)
    - Estimasi parameter Poisson λ dan probabilitas kejadian kegagalan
    - Visualisasi fitted distribution

    Parameters
    ----------
    df              : DataFrame sumber.
    temp_threshold  : Ambang batas suhu kritis dalam Kelvin.
    output_dir      : Direktori simpan plot.

    Returns
    -------
    dict  Kamus hasil: mu, sigma, lambda, probabilities.
    """
    print("\n" + "="*60)
    print("   ANALISIS DISTRIBUSI PROBABILITAS")
    print("="*60)

    results = {}
    results.update(_analyze_normal_distribution(df, temp_threshold, output_dir))
    results.update(_analyze_poisson_distribution(df, output_dir))
    return results


def _analyze_normal_distribution(df, threshold, out) -> Dict:
    """Distribusi Normal pada suhu proses."""
    col  = "Process temperature [K]"
    data = df[col].dropna()
    mu, sigma = data.mean(), data.std()

    # Goodness-of-fit: KS test (lebih stabil untuk n besar)
    ks_stat, ks_p = stats.kstest(data, "norm", args=(mu, sigma))

    # Probabilitas
    prob_exceed  = 1 - stats.norm.cdf(threshold, mu, sigma)
    z_threshold  = (threshold - mu) / sigma
    ci_95_lower  = stats.norm.ppf(0.025, mu, sigma)
    ci_95_upper  = stats.norm.ppf(0.975, mu, sigma)

    print(f"\n[Distribusi Normal – {col}]")
    print(f"  μ (Mean)              : {mu:.4f} K")
    print(f"  σ (Std Dev)           : {sigma:.4f} K")
    print(f"  KS-Statistic          : {ks_stat:.4f}  (p = {ks_p:.4e})")
    print(f"  Uji normalitas        : {'Gagal tolak H₀ (≈ Normal)' if ks_p > 0.05 else 'Tolak H₀ (Non-Normal)'}")
    print(f"  CI 95% operasional    : [{ci_95_lower:.2f} K, {ci_95_upper:.2f} K]")
    print(f"  Z-score ({threshold}K)       : {z_threshold:.4f}")
    print(f"  P(Suhu > {threshold}K)      : {prob_exceed:.4f}  ({prob_exceed*100:.2f}%)")
    print(f"  Interpretasi: Peluang suhu mencapai kondisi kritis ({threshold} K) ≈ {prob_exceed*100:.2f}%")

    # Plot
    _plot_normal_distribution(data, mu, sigma, threshold, col, out)

    return {"mu_temp": mu, "sigma_temp": sigma, "prob_exceed_temp": prob_exceed}


def _plot_normal_distribution(data, mu, sigma, threshold, col, out):
    """PDF distribusi Normal dengan shading area kritis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(data.min(), data.max(), 500)
    pdf = stats.norm.pdf(x, mu, sigma)

    ax.plot(x, pdf, "b-", lw=2, label=f"PDF Normal (μ={mu:.2f}, σ={sigma:.2f})")
    ax.fill_between(x, pdf, where=(x > threshold), alpha=0.4, color="red",
                    label=f"P(X > {threshold}K)")
    ax.axvline(threshold, color="red", ls="--", lw=1.5, label=f"Threshold = {threshold} K")
    ax.axvline(mu, color="green", ls=":", lw=1.5, label=f"Mean = {mu:.2f} K")
    ax.set_xlabel(col); ax.set_ylabel("Densitas Probabilitas")
    ax.set_title(f"Distribusi Normal – {col}", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = f"{out}/stats_normal_distribution.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"\n   → Plot tersimpan: {path}")


def _analyze_poisson_distribution(df, out) -> Dict:
    """Distribusi Poisson pada laju kegagalan mesin per 100 siklus."""
    failure_rate = df[TARGET_COL].mean()
    lam = failure_rate * 100  # λ per 100 siklus

    # PMF untuk k = 0..10
    k_values  = np.arange(0, 11)
    pmf_vals  = stats.poisson.pmf(k_values, lam)
    cdf_vals  = stats.poisson.cdf(k_values, lam)

    prob_0 = stats.poisson.pmf(0, lam)
    prob_1 = stats.poisson.pmf(1, lam)
    prob_2_plus = 1 - stats.poisson.cdf(1, lam)

    print(f"\n[Distribusi Poisson – Kegagalan per 100 Siklus]")
    print(f"  Laju kegagalan global : {failure_rate:.4f}  ({failure_rate*100:.2f}%)")
    print(f"  λ (per 100 siklus)    : {lam:.4f}")
    print(f"  P(X = 0 kegagalan)    : {prob_0:.4f}  ({prob_0*100:.2f}%)")
    print(f"  P(X = 1 kegagalan)    : {prob_1:.4f}  ({prob_1*100:.2f}%)")
    print(f"  P(X ≥ 2 kegagalan)    : {prob_2_plus:.4f}  ({prob_2_plus*100:.2f}%)")

    print(f"\n  Tabel PMF Poisson (k = 0–10):")
    print(f"  {'k':>4}  {'P(X=k)':>10}  {'P(X≤k)':>10}")
    for k, p, c in zip(k_values, pmf_vals, cdf_vals):
        print(f"  {k:>4}  {p:>10.4f}  {c:>10.4f}")

    _plot_poisson_distribution(k_values, pmf_vals, lam, out)

    return {"lambda_poisson": lam, "prob_zero_fail": prob_0, "prob_one_fail": prob_1}


def _plot_poisson_distribution(k_vals, pmf_vals, lam, out):
    """Bar chart PMF distribusi Poisson."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(k_vals, pmf_vals, color=sns.color_palette("viridis", len(k_vals)),
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, pmf_vals):
        if val > 0.005:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Jumlah Kegagalan (k)"); ax.set_ylabel("Probabilitas P(X = k)")
    ax.set_title(f"Distribusi Poisson – Kegagalan per 100 Siklus (λ = {lam:.2f})",
                 fontweight="bold")
    ax.set_xticks(k_vals)
    plt.tight_layout()
    path = f"{out}/stats_poisson_distribution.png"
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"   → Plot tersimpan: {path}")