import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Model loading with fallback handling
MODEL_PATH = "Gradient_Boosting_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found. Please ensure the file is in the root directory.")

# Feature names derived from your model metadata
FEATURE_NAMES = [
    "Ship Mode", "Customer ID", "Customer Name", "Segment", "Country",
    "City", "State", "Region", "Category", "Sub-Category",
    "Product Name", "Quantity", "Discount", "Profit"
]

# Feature definitions for form rendering
FEATURES = [
    {"name": "Ship Mode", "type": "number", "default": 0, "step": "1", "hint": "Encoded integer"},
    {"name": "Customer ID", "type": "number", "default": 0, "step": "1", "hint": "Encoded integer"},
    {"name": "Customer Name", "type": "number", "default": 0, "step": "1", "hint": "Encoded integer"},
    {"name": "Segment", "type": "number", "default": 0, "step": "1", "hint": "Encoded integer"},
    {"name": "Country", "type": "number", "default": 0, "step": "1", "hint": "Encoded integer"},
    {"name": "City", "type": "number", "default": 0, "step": "1", "hint": "Encoded integer"},
    {"name": "State", "type": "number", "default": 0, "step": "1", "hint": "Encoded integer"},
    {"name": "Region", "type": "number", "default": 0, "step": "1", "hint": "Encoded integer"},
    {"name": "Category", "type": "number", "default": 0, "step": "1", "hint": "Encoded integer"},
    {"name": "Sub-Category", "type": "number", "default": 0, "step": "1", "hint": "Encoded integer"},
    {"name": "Product Name", "type": "number", "default": 0, "step": "1", "hint": "Encoded integer"},
    {"name": "Quantity", "type": "number", "default": 1, "step": "1", "hint": "Units quantity"},
    {"name": "Discount", "type": "number", "default": 0.0, "step": "0.01", "hint": "e.g., 0.15 for 15%"},
    {"name": "Profit", "type": "number", "default": 0.0, "step": "0.01", "hint": "Profit value"},
]

HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gradient Boosting Regressor Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.1);
            --primary: #8b5cf6;
            --primary-hover: #7c3aed;
            --accent: #06b6d4;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --input-border: rgba(255, 255, 255, 0.15);
            --input-focus: #8b5cf6;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 960px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header h1 {
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(to right, #c084fc, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 0.4rem;
            color: #e2e8f0;
        }

        .form-group input {
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 10px;
            padding: 0.65rem 0.85rem;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
        }

        .form-group input:focus {
            border-color: var(--input-focus);
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25);
        }

        .form-group span.hint {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        .actions {
            margin-top: 2rem;
            display: flex;
            justify-content: center;
        }

        .btn-submit {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 0.85rem 3rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 10px 15px -3px rgba(139, 92, 246, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 20px -3px rgba(139, 92, 246, 0.6);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--card-border);
            text-align: center;
            display: {% if prediction is not none or error is not none %}block{% else %}none{% endif %};
        }

        .result-box.success {
            border-color: rgba(52, 211, 153, 0.4);
            background: rgba(16, 185, 129, 0.1);
        }

        .result-box.error {
            border-color: rgba(248, 113, 113, 0.4);
            background: rgba(239, 68, 68, 0.1);
        }

        .result-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .result-value {
            font-size: 2rem;
            font-weight: 700;
            color: #34d399;
        }

        .error-message {
            color: #f87171;
            font-weight: 500;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Gradient Boosting Regressor</h1>
            <p>Enter model parameters to generate target predictions</p>
        </div>

        <form method="POST" action="/predict">
            <div class="grid-form">
                {% for feature in features %}
                <div class="form-group">
                    <label for="{{ feature.name }}">{{ feature.name }}</label>
                    <input 
                        type="{{ feature.type }}" 
                        step="{{ feature.step }}" 
                        id="{{ feature.name }}" 
                        name="{{ feature.name }}" 
                        value="{{ request.form.get(feature.name, feature.default) }}" 
                        required
                    >
                    <span class="hint">{{ feature.hint }}</span>
                </div>
                {% endfor %}
            </div>

            <div class="actions">
                <button type="submit" class="btn-submit">Predict Result</button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="result-box success">
            <div class="result-title">Predicted Value</div>
            <div class="result-value">{{ "%.4f"|format(prediction) }}</div>
        </div>
        {% endif %}

        {% if error %}
        <div class="result-box error">
            <div class="error-message">{{ error }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_LAYOUT, features=FEATURES, prediction=None, error=None)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(
            HTML_LAYOUT, 
            features=FEATURES, 
            prediction=None, 
            error="Model file 'Gradient_Boosting_model.pkl' not loaded on server."
        )

    try:
        # Parse inputs in order matching feature_names_in_
        input_values = []
        for feature in FEATURES:
            val = request.form.get(feature["name"])
            input_values.append(float(val))

        # Convert to 2D numpy array for prediction
        input_array = np.array([input_values])
        prediction = model.predict(input_array)[0]

        return render_template_string(
            HTML_LAYOUT, 
            features=FEATURES, 
            prediction=prediction, 
            error=None
        )

    except Exception as e:
        return render_template_string(
            HTML_LAYOUT, 
            features=FEATURES, 
            prediction=None, 
            error=f"Prediction Error: {str(e)}"
        )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API Endpoint for external requests"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        input_values = [float(data[feature["name"]]) for feature in FEATURES]
        prediction = model.predict(np.array([input_values]))[0]
        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
