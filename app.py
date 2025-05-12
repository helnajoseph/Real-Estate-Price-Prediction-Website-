from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import pandas as pd
import json

app = Flask(__name__)

# Load the model (we'll assume it's saved as a pickle file)
MODEL_PATH = 'model.pkl'

# Check if model exists
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
else:
    model = None
    print(f"Warning: Model file {MODEL_PATH} not found. Predictions will not work.")

# Load the list of locations from the model features
try:
    with open('model_features.json', 'r') as f:
        model_features = json.load(f)
    # The first 3 features are numeric (total_sqft, bath, bhk)
    # The rest are location features
    locations = model_features[3:]
    print(f"Loaded {len(locations)} locations from model_features.json")
except Exception as e:
    print(f"Error loading locations: {str(e)}")
    locations = []

@app.route('/')
def home():
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure model.pkl exists.'}), 400

    try:
        # Get data from form
        total_sqft = float(request.form.get('total_sqft', 0))
        bath = int(request.form.get('bath', 0))
        bhk = int(request.form.get('bhk', 0))
        location = request.form.get('location', '')

        print(f"Input: total_sqft={total_sqft}, bath={bath}, bhk={bhk}, location={location}")

        # Create a feature vector with zeros for all features
        x = np.zeros(len(model_features))

        # Set the numeric features
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk

        # Set the location feature (one-hot encoding)
        try:
            loc_index = model_features.index(location)
            x[loc_index] = 1
            print(f"Location '{location}' found at index {loc_index}")
        except ValueError:
            print(f"Warning: Location '{location}' not found in model features")

        # Make prediction
        prediction = model.predict([x])[0]

        # Format the prediction (assuming it's in lakhs for Bangalore)
        prediction_in_lakhs = round(float(prediction), 2)
        prediction_in_rupees = prediction_in_lakhs * 100000

        # Return prediction
        return jsonify({
            'prediction_lakhs': prediction_in_lakhs,
            'prediction_rupees': prediction_in_rupees,
            'success': True
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 400


if __name__ == '__main__':
    app.run(debug=True)
