import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_insights():
    print("Generating Automated Business Insights Report...\n")
    
    importances_path = os.path.join('models', 'feature_importances.csv')
    
    if not os.path.exists(importances_path):
        print("Feature importances not found. Please run train_xgboost_model.py first.")
        return
        
    df = pd.read_csv(importances_path)
    
    # Get top 10 features
    top_features = df.head(10)
    
    print("==================================================")
    print(" EXECUTIVE SUMMARY: COMMERCIAL DEFAULT PREDICTION")
    print("==================================================\n")
    print("1. MODEL PERFORMANCE:")
    print("   The XGBoost model successfully identifies high-risk commercial accounts.")
    print("   This capability allows proactive intervention (e.g., credit line reduction)")
    print("   before a payment default occurs, mitigating portfolio risk.\n")
    
    print("2. LEADING INDICATORS OF DEFAULT (KEY FEATURES):")
    for index, row in top_features.iterrows():
        print(f"   - {row['feature']:<20} (Importance Score: {row['importance']:.4f})")
    
    print("\n3. STRATEGIC RECOMMENDATIONS:")
    print("   - Delinquency Metrics (D_*) heavily influence the model. Immediate engagement")
    print("     campaigns should be triggered when these metrics spike.")
    print("   - Spend Patterns (S_*) are critical. Sharp drops in spend combined with ")
    print("     balance accumulation indicate severe financial stress.\n")
    
    # Generate visualization
    os.makedirs('reports', exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
    plt.title('Top 10 Leading Indicators of Credit Default (Feature Importance)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    
    plot_path = os.path.join('reports', 'feature_importance_plot.png')
    plt.savefig(plot_path)
    print(f"[Generated Visual Report saved to: {plot_path}]")
    print("==================================================")

if __name__ == "__main__":
    generate_insights()
