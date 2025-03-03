import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# ✅ Load model once at startup
print("🔄 Loading Model...")
model = load_model("lstm_model.h5")
print("✅ Model Loaded Successfully!")

@app.route('/')
def home():
    return render_template("index.html")  # Serve the form page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # ✅ Accept JSON data from frontend

        # ✅ Extract features from JSON request
        features = np.array([[  
            float(data["MonthlyCharges"]),
            float(data["TotalCharges"]),
            float(data["tenure"]),
            int(data.get("Contract_OneYear", 0)),  
            int(data.get("Contract_TwoYear", 0)),  
            int(data.get("PaymentMethod_CreditCard", 0)),  
            int(data.get("PaymentMethod_ElectronicCheck", 0)),  
            int(data.get("PaymentMethod_MailedCheck", 0))  
        ]])

        # ✅ Reshape input for LSTM
        features = features.reshape((features.shape[0], features.shape[1], 1))  

        # ✅ Get prediction
        probability = model.predict(features)[0][0]
        prediction = "Yes" if probability > 0.5 else "No"

        return jsonify({
            "Churn Probability": round(float(probability), 4),  # ✅ Round for better readability
            "Prediction": prediction
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
