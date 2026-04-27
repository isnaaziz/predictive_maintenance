import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

def analyze_regression(df, x_col='Torque', y_col='Process temperature'):
    """Analyze Linear Regression and Correlation."""
    print(f"\n--- 2. Regresi-Korelasi ({x_col} vs {y_col}) ---")
    correlation = df[x_col].corr(df[y_col])
    print(f"Korelasi: {correlation:.4f}")
    
    X = df[[x_col]].values
    y = df[y_col].values
    
    model = LinearRegression()
    model.fit(X, y)
    r_sq = model.score(X, y)
    
    print(f"Koefisien Determinasi (R^2): {r_sq:.4f}")
    print(f"Slope (Kenaikan {y_col} per unit {x_col}): {model.coef_[0]:.4f}")
    
    # Plot Regression
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x_col, y=y_col, data=df, line_kws={"color": "red"}, scatter_kws={'alpha':0.3})
    plt.title(f'Regresi Linier: {x_col} vs {y_col} (R^2={r_sq:.2f})')
    plt.savefig('regression_plot.png')
    print("Saved regression_plot.png")
    # plt.show() # Disabled to prevent stalling
    return model

def analyze_probability(df):
    """Analyze failure probabilities using Normal and Poisson distributions."""
    print("\n--- 3. Distribusi Probabilitas ---")
    
    # 3a. Distribusi Normal (Process Temperature)
    mu = df['Process temperature'].mean()
    std = df['Process temperature'].std()
    print(f"Normal Distribution Suhu Proses: Mean={mu:.2f}, Std={std:.2f}")
    
    threshold = 312
    prob_high_temp = 1 - stats.norm.cdf(threshold, mu, std)
    print(f"Probabilitas Suhu Proses > {threshold}K: {prob_high_temp:.4f}")

    # 3b. Distribusi Poisson (Machine Failure)
    failure_rate = df['Machine failure'].mean()
    lam = failure_rate * 100 # average failures per 100 cycles
    print(f"Rata-rata kegagalan per 100 siklus (lambda): {lam:.2f}")
    
    prob_1_fail = stats.poisson.pmf(1, lam)
    print(f"Probabilitas tepat 1 kegagalan dalam 100 siklus: {prob_1_fail:.4f}")
