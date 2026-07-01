from data_preprocessing import load_invoice_data, split_data, scale_features, apply_labels
from modeling_evaluation import train_random_forest, evaluate_classifier
import joblib

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
]

TARGET = "flag_invoice"

import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def main():
    # Load data
    df = load_invoice_data()
    df = apply_labels(df)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Class distribution:\n{df['flag_invoice'].value_counts()}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    print(f"\nTrain set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")
    
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, 'models/scaler.pkl'
    )
    
    # Train and evaluate models
    grid_search = train_random_forest(X_train_scaled, y_train)
    
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )
    
    # Save best model
    joblib.dump(grid_search.best_estimator_, 'models/predict_flag_invoice.pkl')

if __name__ == "__main__":
    main()