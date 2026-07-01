import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained invoice flagging classifier model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def load_scaler(scaler_path: str = SCALER_PATH):
    """
    Load feature scaler.
    """
    with open(scaler_path, "rb") as f:
        scaler = joblib.load(f)
    return scaler

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for new vendor invoices.
    
    Parameters
    ----------
    input_data : dict
        Dictionary containing invoice features
        
    Returns
    -------
    pd.DataFrame
        DataFrame with predicted flag (0 = safe, 1 = flag for review)
    """
    model = load_model()
    scaler = load_scaler()
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # ✓ SCALE the input data BEFORE prediction
    input_scaled = scaler.transform(input_df)
    
    # Predict
    predictions = model.predict(input_scaled)
    
    # Add predictions to dataframe
    input_df['Predicted_Flag'] = predictions
    input_df['Risk_Level'] = input_df['Predicted_Flag'].map({
        0: 'Safe - No Review Needed',
        1: 'Flag - Review Required'
    })
    
    return input_df

if __name__ == "__main__":
    # Single prediction
    single_invoice = {
        "invoice_quantity": [100],
        "invoice_dollars": [18000],
        "Freight": [450],
        "total_item_quantity": [500],
        "total_item_dollars": [18500]
    }
    
    result = predict_invoice_flag(single_invoice)
    print(result)
    
    # Batch predictions
    batch_invoices = {
        "invoice_quantity": [100, 50, 200],
        "invoice_dollars": [18000, 9500, 35000],
        "Freight": [450, 200, 800],
        "total_item_quantity": [500, 250, 1000],
        "total_item_dollars": [18500, 9000, 35000]
    }
    
    results = predict_invoice_flag(batch_invoices)
    print(results)