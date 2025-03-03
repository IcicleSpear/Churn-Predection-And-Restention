import os
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load dataset
df = pd.read_csv("customer_churn.csv")

# Handle missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Select relevant features
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

# Build the LSTM Model
def build_model(input_shape):
    model = Sequential([
        LSTM(32, input_shape=(input_shape, 1), return_sequences=True),  # First LSTM layer
        LSTM(16),  # Second LSTM layer
        Dense(1, activation='sigmoid')  # Output layer
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the initial LSTM model
model = build_model(X.shape[1])  # Use the number of features as the input shape
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Save the full model (architecture + weights)
model.save("lstm_model.h5")  # Save the entire model

# Genetic Algorithm for Feature Selection
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, lambda: [random.randint(0, 1) for _ in range(X.shape[1])])
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=20)  # Population size remains the same

# Genetic Algorithm Evaluation Function
def evaluate(individual):
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    
    # If no features are selected, return a low fitness score
    if len(selected_features) == 0:
        return (0.5,)
    
    # Select the columns based on the selected features
    X_selected = X[:, selected_features]  # Select the features based on GA

    # Reshape X_selected to match the input shape for LSTM
    X_selected = X_selected.reshape((X_selected.shape[0], X_selected.shape[1], 1))  # (samples, timesteps, features)

    # Rebuild the model with the new number of selected features
    model = build_model(X_selected.shape[1])  # Rebuild model with selected features

    # Train the model on the selected features for 5 epochs
    model.fit(X_selected, y, epochs=5, batch_size=32, verbose=0)

    # Evaluate the model on the selected features
    loss, accuracy = model.evaluate(X_selected, y, verbose=0)
    return (accuracy,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selBest)

# Run Genetic Algorithm for 10 generations
population = toolbox.population()
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)  # Only 10 generations

# Get the best feature set
best_individual = tools.selBest(population, k=1)[0]
selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
print(f"✅ Best Feature Set: {selected_features}")

# Train the final LSTM model using selected features
X_selected = X[:, selected_features]

# Reshape X_selected to match the input shape for LSTM (samples, timesteps, features)
X_selected = X_selected.reshape((X_selected.shape[0], X_selected.shape[1], 1))

# Rebuild the model with the selected features and train the final model
final_model = build_model(X_selected.shape[1])  # Rebuild model with selected features
final_model.fit(X_selected, y, epochs=15, batch_size=32, verbose=1)

# Evaluate the final model
final_loss, final_accuracy = final_model.evaluate(X_selected, y)
print(f"🎯 Final Model Accuracy: {final_accuracy * 100:.2f}%")  # Print final accuracy