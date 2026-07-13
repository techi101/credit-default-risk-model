import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import os
import joblib

def feature_engineer(df):
    """
    Engineers features by aggregating time-series data for each customer.
    Top Kaggle solutions relied heavily on these aggregations.
    """
    print("Engineering features...")
    
    # We drop the date column for aggregation
    num_features = [col for col in df.columns if col not in ['customer_ID', 'S_2']]
    
    # Aggregation functions
    agg_funcs = ['mean', 'std', 'min', 'max', 'last']
    
    # Group by customer and aggregate
    train_agg = df.groupby('customer_ID')[num_features].agg(agg_funcs)
    
    # Flatten multi-level columns
    train_agg.columns = ['_'.join(x) for x in train_agg.columns]
    train_agg.reset_index(inplace=True)
    
    return train_agg

def main():
    print("Loading data...")
    train_path = os.path.join('data', 'train_data.csv')
    labels_path = os.path.join('data', 'train_labels.csv')
    
    if not os.path.exists(train_path):
        print("Data not found! Please run generate_synthetic_data.py first.")
        return
        
    df = pd.read_csv(train_path)
    labels = pd.read_csv(labels_path)
    
    # Feature Engineering
    train_agg = feature_engineer(df)
    
    # Merge with labels
    train_data = train_agg.merge(labels, on='customer_ID', how='left')
    
    # Prepare X and y
    X = train_data.drop(['customer_ID', 'target'], axis=1)
    y = train_data['target']
    
    print(f"Dataset shape after feature engineering: {X.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training XGBoost model...")
    # Initialize XGBoost (parameters similar to those used in the Kaggle competition)
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        eval_metric='auc',
        random_state=42,
        use_label_encoder=False
    )
    
    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val) for X_val, y_val in [(X_train, y_train), (X_test, y_test)]],
        verbose=False
    )
    
    # Evaluate
    preds_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds_proba)
    print(f"\nValidation AUC Score: {auc:.4f}")
    
    preds_class = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, preds_class))
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/xgboost_default_model.pkl')
    
    # Save feature importances for the insights script
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importances.to_csv('models/feature_importances.csv', index=False)
    print("Model and feature importances saved to /models.")

if __name__ == "__main__":
    main()
