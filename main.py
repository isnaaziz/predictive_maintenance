from src.data_loader import load_maintenance_data
from src.eda import perform_eda
from src.stats_models import analyze_regression, analyze_probability
from src.anomaly_detection import detect_anomalies

def main():
    # 1. Load Data
    df = load_maintenance_data()
    
    # 2. EDA
    perform_eda(df)
    
    # 3. Statistical Modeling
    analyze_regression(df)
    analyze_probability(df)
    
    # 4. Anomaly Detection
    detect_anomalies(df)

if __name__ == "__main__":
    main()
