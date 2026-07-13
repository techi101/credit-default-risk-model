import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_amex_data(num_customers=5000, max_statements=13):
    """
    Generates a synthetic dataset mimicking the structure of the Kaggle Amex Default Prediction competition.
    This creates a time-series dataset of customer behaviors (spending, delinquency, balance, payments).
    """
    print(f"Generating synthetic data for {num_customers} customers...")
    
    np.random.seed(42)
    
    # Generate customer IDs
    customer_ids = [f"{i:016x}" for i in range(num_customers)]
    
    data = []
    labels = []
    
    # Feature names based on Amex dataset categories
    # P: Payment, D: Delinquency, B: Balance, S: Spend, R: Risk
    
    for cust_id in customer_ids:
        # Determine how many statements this customer has (1 to max_statements)
        n_statements = np.random.randint(1, max_statements + 1)
        
        # Determine if this customer defaults (target = 1)
        # We will make default more likely if certain random "risk factors" are high
        base_risk = np.random.rand()
        target = 1 if base_risk > 0.75 else 0
        labels.append({"customer_ID": cust_id, "target": target})
        
        # Start date
        start_date = datetime(2022, 1, 1) + timedelta(days=np.random.randint(0, 30))
        
        # Initialize some latent features that drift over time
        p_2 = np.random.normal(0.8, 0.1) if target == 0 else np.random.normal(0.4, 0.2) # Payment fraction
        d_39 = np.random.normal(0, 1) if target == 0 else np.random.normal(5, 3) # Delinquency metric
        b_1 = np.random.normal(0.1, 0.05) if target == 0 else np.random.normal(0.5, 0.3) # Balance
        s_3 = np.random.normal(0.2, 0.1) # Spend
        
        for i in range(n_statements):
            statement_date = start_date + timedelta(days=30 * i)
            
            # Add some noise/drift
            p_2 += np.random.normal(0, 0.02)
            d_39 += np.random.normal(0, 0.5)
            b_1 += np.random.normal(0, 0.05)
            s_3 += np.random.normal(0, 0.02)
            
            # Ensure boundaries
            p_2 = max(0, min(1, p_2))
            b_1 = max(0, b_1)
            d_39 = max(0, d_39)
            
            row = {
                "customer_ID": cust_id,
                "S_2": statement_date.strftime("%Y-%m-%d"),
                "P_2": p_2,
                "D_39": d_39,
                "B_1": b_1,
                "B_2": np.random.normal(0.5, 0.2),
                "R_1": np.random.exponential(0.1) if target == 0 else np.random.exponential(1.0),
                "S_3": s_3,
                "D_41": np.random.poisson(0.1) if target == 0 else np.random.poisson(1.5),
            }
            data.append(row)
            
    df = pd.DataFrame(data)
    labels_df = pd.DataFrame(labels)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    
    train_path = os.path.join('data', 'train_data.csv')
    labels_path = os.path.join('data', 'train_labels.csv')
    
    df.to_csv(train_path, index=False)
    labels_df.to_csv(labels_path, index=False)
    
    print(f"Dataset generated! Saved {len(df)} rows to {train_path}")
    print(f"Labels saved to {labels_path}")

if __name__ == "__main__":
    generate_synthetic_amex_data()
