# train_freight_model.py
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

os.makedirs('models', exist_ok=True)

def load_freight_data():
    conn = sqlite3.connect('../data/inventory.db')
    query = """
    SELECT
        vi.Quantity,
        vi.Dollars,
        vi.Freight
    FROM vendor_invoice vi
    WHERE vi.Freight IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Load and train
df = load_freight_data()
print(f"Training data: {df.shape[0]} samples")

X = df[['Quantity', 'Dollars']]
y = df['Freight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
rmse = (mean_squared_error(y_test, y_pred)) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

joblib.dump(model, 'models/predict_freight_model.pkl')
joblib.dump(scaler, 'models/freight_scaler.pkl')
print("✓ Model and scaler saved!")