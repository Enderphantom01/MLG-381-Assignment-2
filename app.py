from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from pathlib import Path

notebook_path = Path().resolve()
model_path = notebook_path.parent / "model" / 'churn_model.pkl'

app = Flask(__name__)

# Load the model
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_features)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)

