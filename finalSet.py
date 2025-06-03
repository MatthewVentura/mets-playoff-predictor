import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
df = pd.read_csv("Final_NN_Training_Dataset_2000_2024.csv")

# Clean percentage columns
df['bb%'] = df['bb%'].str.rstrip('%').astype(float)
df['k%'] = df['k%'].str.rstrip('%').astype(float)

# Define full feature set and target
features = [
    'war', 'pa', 'hr', 'r', 'rbi', 'bb%', 'k%', 'iso', 'avg',
    'obp', 'slg', 'woba', 'wrc+', 'off', 'def', 'bsr',
    'era', 'whip', 'defeff', 'rtot', 'defrunssaved'
]
target = 'made_playoffs'

# Drop rows with missing values
df_clean = df[features + [target]].dropna()

# Extract features and target
X = df_clean[features]
y = df_clean[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
with open("final_model_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build a deeper neural network to match added complexity
model = Sequential([
    Dense(128, activation='relu', input_shape=(len(features),)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[early_stop]
)

# Save the trained model
model.save("Trained_MLB_Model_2000_2024.h5")
