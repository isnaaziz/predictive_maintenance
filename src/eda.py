import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    """Perform basic Descriptive Statistics and Visualizations."""
    print("\n--- 1. Data Deskriptif (EDA) ---")
    print(df.describe())
    
    # Set style
    plt.style.use('ggplot')
    sns.set_palette("viridis")
    
    # Visualisasi Distribusi Suhu
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['Air temperature'], kde=True, color='skyblue')
    plt.title('Distribusi Air Temperature')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['Process temperature'], kde=True, color='salmon')
    plt.title('Distribusi Process Temperature')
    plt.tight_layout()
    plt.savefig('distribution_temp.png')
    print("Saved distribution_temp.png")
    # plt.show() # Disabled to prevent stalling in terminal
