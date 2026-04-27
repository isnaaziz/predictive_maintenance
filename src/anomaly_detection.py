import numpy as np
from scipy import stats

def detect_anomalies(df, column='Rotational speed', threshold=3):
    """Detect anomalies using Z-score."""
    print(f"\n--- 4. Threshold Kegagalan (Z-Score) pada {column} ---")
    
    z_scores = np.abs(stats.zscore(df[column]))
    anomalies = df[z_scores > threshold]
    
    print(f"Jumlah data: {len(df)}")
    print(f"Jumlah anomali (Z > {threshold}): {len(anomalies)}")
    
    # Hubungan dengan kegagalan sebenarnya
    if 'Machine failure' in df.columns:
        failure_count = anomalies['Machine failure'].sum()
        print(f"Jumlah kegagalan yang terdeteksi dalam zona anomali: {failure_count}")
        
    return anomalies
