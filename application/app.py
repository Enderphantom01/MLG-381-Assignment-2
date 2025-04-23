from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load the model
notebook_path = Path().resolve()
model_path = notebook_path.parent / "MLG-381-Assignment-2" / "model" / 'churn_model.pkl'
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Create DataFrame with the same structure as training data
        input_data = {
            'gender_Male': [1 if data['gender'] == 'Male' else 0],
            'MultipleLines_No phone service': [1 if data['multipleLines'] == 'No phone service' else 0],
            'MultipleLines_Yes': [1 if data['multipleLines'] == 'Yes' else 0],
            'InternetService_Fiber optic': [1 if data['internetService'] == 'Fiber optic' else 0],
            'InternetService_No': [1 if data['internetService'] == 'No' else 0],
            'Contract_One year': [1 if data['contract'] == 'One year' else 0],
            'Contract_Two year': [1 if data['contract'] == 'Two year' else 0],
            'PaperlessBilling_Yes': [1 if data['paperlessBilling'] == 'Yes' else 0],
            'PaymentMethod_Credit card (automatic)': [1 if data['paymentMethod'] == 'Credit card' else 0],
            'PaymentMethod_Electronic check': [1 if data['paymentMethod'] == 'Electronic check' else 0],
            'PaymentMethod_Mailed check': [1 if data['paymentMethod'] == 'Mailed check' else 0],
            'hasFamily_Yes': [1 if data['hasFamily'] == 'Yes' else 0],
            'SeniorCitizen': [1 if data['seniorCitizen'] else 0],
            'AvgMonthlyCost': [float(data['monthlyCost'])]
        }
        
        # Create DataFrame
        df = pd.DataFrame(input_data)
        
        # Ensure all columns are present (fill missing with 0)
        expected_columns = [
            'gender_Male', 'MultipleLines_No phone service', 'MultipleLines_Yes',
            'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year',
            'Contract_Two year', 'PaperlessBilling_Yes', 
            'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check', 'hasFamily_Yes', 'SeniorCitizen', 'AvgMonthlyCost'
        ]
        
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training data
        df = df[expected_columns]
        
        # Make prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1]  # Probability of churn (Yes)
        
        return jsonify({
            'prediction': 'Yes' if prediction[0] == 'Yes' else 'No',
            'probability': float(probability),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)