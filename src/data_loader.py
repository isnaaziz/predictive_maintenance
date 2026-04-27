from ucimlrepo import fetch_ucirepo 
import pandas as pd

def load_maintenance_data(dataset_id=601):
    """Fetch and prepare the AI4I 2020 Predictive Maintenance Dataset."""
    print(f"Fetching dataset ID {dataset_id}...")
    dataset = fetch_ucirepo(id=dataset_id) 
    
    X = dataset.data.features 
    y = dataset.data.targets 
    
    # Combine features and targets
    df = pd.concat([X, y], axis=1)
    
    # Cleanup column names (remove [K], [rpm], etc if present, or just use as is)
    # Based on inspection: ['Type', 'Air temperature', 'Process temperature', 
    # 'Rotational speed', 'Torque', 'Tool wear', 'Machine failure', ...]
    return df
