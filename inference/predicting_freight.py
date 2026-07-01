import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "invoice flagging" / "models" / "predict_freight_model.pkl"
SCALER_PATH = BASE_DIR / "invoice flagging" / "models" / "freight_scaler.pkl"

def load_model(model_path: str = MODEL_PATH):
    """Load trained freight cost prediction model."""
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def load_scaler(scaler_path: str = SCALER_PATH):
    """Load feature scaler."""
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler

def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.
    
    Parameters
    ----------
    input_data : dict
        Dictionary containing invoice features
        Example: {"Quantity": [100], "Dollars": [18500]}
        
    Returns
    -------
    pd.DataFrame
        DataFrame with predicted freight cost
    """
    model = load_model()
    scaler = load_scaler()
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # ✓ SCALE the input data BEFORE prediction
    input_scaled = scaler.transform(input_df)
    
    # Predict
    predictions = model.predict(input_scaled).round()
    
    # Add predictions to dataframe
    input_df['Predicted_Freight'] = predictions
    return input_df

if __name__ == "__main__":
    # Example predictions
    batch_data = {
        "Quantity": [100, 250, 50],
        "Dollars": [18500, 9000, 5000]
    }
    
    prediction = predict_freight_cost(batch_data)
    print("Batch Predictions:")
    print(prediction)