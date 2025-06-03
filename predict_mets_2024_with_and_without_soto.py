import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# === Step 1: More realistic Mets 2024 projections ===
mets_2024_no_soto = {
    'war': 45.0,
    'pa': 6150,
    'hr': 200,
    'r': 755,
    'rbi': 735,
    'bb%': 0.088,
    'k%': 0.215,
    'iso': 0.170,
    'avg': 0.255,
    'obp': 0.325,
    'slg': 0.425,
    'woba': 0.322,
    'wrc+': 105,
    'off': 12.0,
    'def': 3.5,
    'bsr': 2.2,
    'era': 3.95,
    'whip': 1.29,
    'defeff': 0.704,
    'rtot': 10,
    'defrunssaved': 8
}

# Add Soto's WAR
mets_2024_with_soto = mets_2024_no_soto.copy()
mets_2024_with_soto['war'] += 6.5

# === Step 2: Load model and scaler ===
model = load_model("Trained_MLB_Model_2000_2024.h5")
with open("final_model_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# === Step 3: Build DataFrame and ensure feature order ===
input_df = pd.DataFrame([mets_2024_no_soto, mets_2024_with_soto])
input_df.index = ['No Soto', 'With Soto']

feature_order = [
    'war', 'pa', 'hr', 'r', 'rbi', 'bb%', 'k%', 'iso', 'avg',
    'obp', 'slg', 'woba', 'wrc+', 'off', 'def', 'bsr',
    'era', 'whip', 'defeff', 'rtot', 'defrunssaved'
]

X_scaled = scaler.transform(input_df[feature_order])

# === Step 4: Predict and display results ===
preds = model.predict(X_scaled).flatten()
input_df['Playoff Probability (%)'] = (preds * 100).round(2)

print("\n📊 2024 Mets Playoff Chances (Updated Estimates):")
print(input_df[['Playoff Probability (%)']])
