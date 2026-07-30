from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Absolute path targeting the exact directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Gradient_Boosting_model.pkl")

feature_names = [
    "Ship Mode", "Customer ID", "Customer Name", "Segment", "Country", 
    "City", "State", "Region", "Category", "Sub-Category", 
    "Product Name", "Quantity", "Discount", "Profit"
]

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

model = load_model()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Superstore Sales Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #0d9488;
            --accent-hover: #14b8a6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: #334155;
            --border-color: #475569;
            --result-bg: #111827;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-main); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 2rem 1rem; }
        .container { background-color: var(--card-bg); border-radius: 12px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); width: 100%; max-width: 900px; padding: 2.5rem; border: 1px solid var(--border-color); }
        header { text-align: center; margin-bottom: 2rem; }
        header h1 { font-size: 1.8rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.5rem; }
        header p { color: var(--text-muted); font-size: 0.95rem; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.25rem; }
        .form-group { display: flex; flex-direction: column; }
        .form-group label { font-size: 0.85rem; font-weight: 600; margin-bottom: 0.4rem; color: var(--text-muted); }
        .form-group input { background-color: var(--input-bg); border: 1px solid var(--border-color); color: var(--text-main); padding: 0.65rem 0.85rem; border-radius: 6px; font-size: 0.95rem; outline: none; transition: border-color 0.2s; }
        .form-group input:focus { border-color: var(--accent-color); }
        .btn-submit { grid-column: 1 / -1; margin-top: 1rem; background-color: var(--accent-color); color: #ffffff; border: none; padding: 0.85rem; font-size: 1rem; font-weight: 600; border-radius: 6px; cursor: pointer; transition: background-color 0.2s; }
        .btn-submit:hover { background-color: var(--accent-hover); }
        .result-box { margin-top: 2rem; padding: 1.5rem; background-color: var(--result-bg); border-left: 4px solid var(--accent-color); border-radius: 6px; text-align: center; }
        .result-box h3 { font-size: 0.9rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.3rem; }
        .result-box .value { font-size: 1.75rem; font-weight: 700; color: var(--accent-hover); }
        .error-box { margin-top: 2rem; padding: 1rem; background-color: #7f1d1d; color: #fca5a5; border-radius: 6px; font-size: 0.9rem; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Gradient Boosting Model Inference</h1>
            <p>Provide numeric or encoded values for model predictions</p>
        </header>

        <form action="/predict" method="POST">
            <div class="form-grid">
                {% for feature in features %}
                <div class="form-group">
                    <label for="{{ feature }}">{{ feature }}</label>
                    <input type="number" step="any" id="{{ feature }}" name="{{ feature }}" value="{{ request.form.get(feature, '0') }}" required>
                </div>
                {% endfor %}
                <button type="submit" class="btn-submit">Calculate Prediction</button>
            </div>
        </form>

        {% if prediction_text %}
        <div class="result-box">
            <h3>Prediction Result</h3>
            <div class="value">${{ prediction_text }}</div>
        </div>
        {% endif %}

        {% if error_text %}
        <div class="error-box">
            {{ error_text }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE, features=feature_names)

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        model = load_model()

    if model is None:
        return render_template_string(
            HTML_TEMPLATE, 
            features=feature_names, 
            error_text="Error: Model file 'GradientBoosting_model.pkl' could not be loaded. Verify file path on deployment server."
        )

    try:
        input_values = [float(request.form.get(feature, 0)) for feature in feature_names]
        features_array = np.array([input_values])
        prediction = model.predict(features_array)[0]
        
        return render_template_string(
            HTML_TEMPLATE, 
            features=feature_names, 
            prediction_text=f"{prediction:,.2f}"
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE, 
            features=feature_names, 
            error_text=f"Prediction Error: {str(e)}"
        )

# Required for Vercel serverless execution
app = app
