"""
This script analyzes the Bangalore home prices model and updates the web application.
"""

import pickle
import os
import json
import numpy as np

def analyze_model(model_path='model.pkl'):
    """
    Analyze the model to extract feature information.
    
    Args:
        model_path: Path to the pickled model
        
    Returns:
        Dictionary with model information
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        model_info = {
            'type': type(model).__name__,
        }
        
        # Try to extract feature names
        if hasattr(model, 'feature_names_in_'):
            model_info['features'] = list(model.feature_names_in_)
        
        # For Bangalore home prices model, we might need to check for specific attributes
        # or structure based on how it was trained
        
        print(f"Model type: {model_info['type']}")
        
        # Check if it's a scikit-learn model
        if hasattr(model, 'predict'):
            print("Model has predict method (likely a scikit-learn model)")
        
        # Try to get more information about the model
        if hasattr(model, 'get_params'):
            params = model.get_params()
            print(f"Model parameters: {params}")
        
        # For Bangalore home prices, let's try to make a dummy prediction to see what features it expects
        try:
            # Create a dummy input with some reasonable values for Bangalore properties
            dummy_input = np.zeros((1, 5))  # Adjust the number of features as needed
            prediction = model.predict(dummy_input)
            print(f"Dummy prediction shape: {prediction.shape}")
            print(f"Dummy prediction value: {prediction}")
        except Exception as e:
            print(f"Could not make dummy prediction: {str(e)}")
        
        return model_info
    except Exception as e:
        print(f"Error analyzing model: {str(e)}")
        return None

def update_app_for_bangalore_model():
    """
    Update the web application to work with the Bangalore home prices model.
    """
    # Update the HTML template
    html_path = 'templates/index.html'
    
    try:
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        # Replace the form with Bangalore-specific fields
        form_start = html_content.find('<form id="prediction-form">')
        form_end = html_content.find('<div class="text-center">', form_start)
        
        bangalore_form = """
<form id="prediction-form">
    <div class="row mb-3">
        <div class="col-md-6">
            <div class="form-group">
                <label for="location">Location</label>
                <select class="form-control" id="location" name="location" required>
                    <option value="1">1st Block Jayanagar</option>
                    <option value="2">1st Phase JP Nagar</option>
                    <option value="3">2nd Phase Judicial Layout</option>
                    <option value="4">2nd Stage Nagarbhavi</option>
                    <option value="5">5th Block Hbr Layout</option>
                    <option value="6">5th Phase JP Nagar</option>
                    <option value="7">6th Phase JP Nagar</option>
                    <option value="8">7th Phase JP Nagar</option>
                    <option value="9">8th Phase JP Nagar</option>
                    <option value="10">9th Phase JP Nagar</option>
                    <option value="11">AECS Layout</option>
                    <option value="12">Abbigere</option>
                    <option value="13">Akshaya Nagar</option>
                    <!-- Add more locations as needed -->
                </select>
            </div>
        </div>
        <div class="col-md-6">
            <div class="form-group">
                <label for="sqft">Total Square Feet Area</label>
                <input type="number" class="form-control" id="sqft" name="sqft" required>
            </div>
        </div>
    </div>
    
    <div class="row mb-3">
        <div class="col-md-6">
            <div class="form-group">
                <label for="bath">Number of Bathrooms</label>
                <input type="number" class="form-control" id="bath" name="bath" required>
            </div>
        </div>
        <div class="col-md-6">
            <div class="form-group">
                <label for="bhk">BHK (Bedrooms, Hall, Kitchen)</label>
                <input type="number" class="form-control" id="bhk" name="bhk" required>
            </div>
        </div>
    </div>
"""
        
        new_html = html_content[:form_start] + bangalore_form + html_content[form_end:]
        
        with open(html_path, 'w') as f:
            f.write(new_html)
        
        print(f"Updated {html_path} with Bangalore-specific form fields")
        
        # Update app.py to handle Bangalore model predictions
        app_path = 'app.py'
        with open(app_path, 'r') as f:
            app_content = f.read()
        
        # Update the predict route to handle Bangalore model features
        predict_start = app_content.find('@app.route(\'/predict\'')
        predict_end = app_content.find('if __name__ ==', predict_start)
        
        bangalore_predict = """
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please ensure model.pkl exists.'}), 400
    
    try:
        # Get data from form
        location = int(request.form.get('location', 0))
        sqft = float(request.form.get('sqft', 0))
        bath = int(request.form.get('bath', 0))
        bhk = int(request.form.get('bhk', 0))
        
        # Prepare input for model
        # Note: Adjust this based on how your model expects input
        input_features = [[location, sqft, bath, bhk]]
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        
        # Convert to lakhs (if your model predicts in lakhs)
        # prediction_in_lakhs = prediction
        
        # Return prediction
        return jsonify({
            'prediction': round(float(prediction), 2),
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 400
"""
        
        new_app = app_content[:predict_start] + bangalore_predict + app_content[predict_end:]
        
        with open(app_path, 'w') as f:
            f.write(new_app)
        
        print(f"Updated {app_path} to handle Bangalore model predictions")
        
        return True
    except Exception as e:
        print(f"Error updating app for Bangalore model: {str(e)}")
        return False

if __name__ == "__main__":
    print("Analyzing Bangalore home prices model...")
    model_info = analyze_model()
    
    if model_info:
        print("\nUpdating web application for Bangalore home prices model...")
        update_app_for_bangalore_model()
        
        print("\nDone! You can now run the web application with:")
        print("python app.py")
    else:
        print("\nFailed to analyze model. Please check the model file.")
