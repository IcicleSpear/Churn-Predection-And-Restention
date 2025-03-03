import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Check if the model file exists
if os.path.exists("lstm_model.h5"):
    print("✅ Model file found.")
else:
    print("❌ Model file NOT found. Ensure you have saved it correctly.")
    exit()

# Load the saved model
try:
    model = load_model("lstm_model.h5")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Display model architecture
model.summary()

# Load dataset (same as training data)
df = pd.read_csv("customer_churn.csv")

# Handle missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Select relevant features (must match training features)
df = df[["MonthlyCharges", "TotalCharges", "tenure", "Contract", "PaymentMethod", "Churn"]]

# Convert categorical variables to numerical
df = pd.get_dummies(df, columns=["Contract", "PaymentMethod"], drop_first=True)

# Convert "Churn" to binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Normalize numerical features
scaler = MinMaxScaler()
X = scaler.fit_transform(df.drop(columns=["Churn"]))
y = df["Churn"].values

# Reshape X for LSTM (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Select a sample input for testing
sample_input = X[0].reshape(1, X.shape[1], 1)

# Make a prediction
prediction = model.predict(sample_input)
print(f"🔍 Sample Prediction (Churn Probability): {prediction[0][0]:.4f}")

# Evaluate Model Accuracy
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")




