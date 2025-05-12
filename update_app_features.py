"""
This script helps you update the web application to match your dataset features.
It analyzes your dataset and model, then suggests changes to the web form.
"""

import pandas as pd
import pickle
import os
import sys
import re
from bs4 import BeautifulSoup

def get_model_features(model_path='model.pkl'):
    """
    Extract feature names from a trained model.
    
    Args:
        model_path: Path to the pickled model
        
    Returns:
        List of feature names if available
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Try different ways to get feature names
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        elif hasattr(model, 'feature_names'):
            return list(model.feature_names)
        else:
            print("Could not automatically extract feature names from model.")
            return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def analyze_dataset(file_path):
    """
    Analyze a dataset to extract feature information.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Dictionary with feature information
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        elif file_extension == '.pkl':
            df = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        # Get feature information
        feature_info = {}
        for column in df.columns:
            if column == df.columns[-1]:  # Assume last column is target
                continue
                
            dtype = df[column].dtype
            unique_values = df[column].unique()
            
            if dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(dtype):
                if len(unique_values) <= 5:  # Likely categorical
                    feature_info[column] = {
                        'type': 'categorical',
                        'values': sorted(unique_values.tolist()),
                        'html_type': 'select'
                    }
                else:
                    feature_info[column] = {
                        'type': 'numeric',
                        'min': df[column].min(),
                        'max': df[column].max(),
                        'html_type': 'number'
                    }
            elif dtype == 'bool':
                feature_info[column] = {
                    'type': 'boolean',
                    'values': [True, False],
                    'html_type': 'select'
                }
            else:
                feature_info[column] = {
                    'type': 'text',
                    'html_type': 'text'
                }
        
        return feature_info
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")
        return None

def generate_form_html(feature_info):
    """
    Generate HTML form elements based on feature information.
    
    Args:
        feature_info: Dictionary with feature information
        
    Returns:
        HTML string with form elements
    """
    html = []
    
    # Group features into pairs for two-column layout
    features = list(feature_info.items())
    for i in range(0, len(features), 2):
        html.append('<div class="row mb-3">')
        
        # First feature in pair
        name, info = features[i]
        html.append('    <div class="col-md-6">')
        html.append('        <div class="form-group">')
        html.append(f'            <label for="{name}">{name.replace("_", " ").title()}</label>')
        
        if info['html_type'] == 'select':
            html.append(f'            <select class="form-control" id="{name}" name="{name}">')
            for val in info['values']:
                html.append(f'                <option value="{val}">{val}</option>')
            html.append('            </select>')
        elif info['html_type'] == 'number':
            html.append(f'            <input type="number" class="form-control" id="{name}" name="{name}" required>')
        else:
            html.append(f'            <input type="text" class="form-control" id="{name}" name="{name}" required>')
        
        html.append('        </div>')
        html.append('    </div>')
        
        # Second feature in pair (if exists)
        if i + 1 < len(features):
            name, info = features[i + 1]
            html.append('    <div class="col-md-6">')
            html.append('        <div class="form-group">')
            html.append(f'            <label for="{name}">{name.replace("_", " ").title()}</label>')
            
            if info['html_type'] == 'select':
                html.append(f'            <select class="form-control" id="{name}" name="{name}">')
                for val in info['values']:
                    html.append(f'                <option value="{val}">{val}</option>')
                html.append('            </select>')
            elif info['html_type'] == 'number':
                html.append(f'            <input type="number" class="form-control" id="{name}" name="{name}" required>')
            else:
                html.append(f'            <input type="text" class="form-control" id="{name}" name="{name}" required>')
            
            html.append('        </div>')
            html.append('    </div>')
        
        html.append('</div>')
    
    return '\n'.join(html)

def update_html_template(feature_info, template_path='templates/index.html'):
    """
    Update the HTML template with new form elements.
    
    Args:
        feature_info: Dictionary with feature information
        template_path: Path to the HTML template
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(template_path, 'r') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        form = soup.find('form', id='prediction-form')
        
        if form:
            # Remove existing form content (except submit button)
            for child in list(form.children):
                if child.name == 'div' and 'text-center' not in child.get('class', []):
                    child.decompose()
            
            # Add new form content
            form_html = generate_form_html(feature_info)
            form.insert(0, BeautifulSoup(form_html, 'html.parser'))
            
            # Write updated HTML
            with open(template_path, 'w') as f:
                f.write(str(soup))
            
            print(f"Successfully updated {template_path}")
            return True
        else:
            print(f"Could not find form element in {template_path}")
            return False
    except Exception as e:
        print(f"Error updating HTML template: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        
        print(f"Analyzing dataset: {dataset_path}")
        feature_info = analyze_dataset(dataset_path)
        
        if feature_info:
            print("\nExtracted feature information:")
            for name, info in feature_info.items():
                print(f"- {name}: {info['type']}")
            
            print("\nChecking model for feature names...")
            model_features = get_model_features()
            
            if model_features:
                print(f"Model features: {', '.join(model_features)}")
                
                # Filter feature_info to only include model features
                feature_info = {k: v for k, v in feature_info.items() if k in model_features}
            
            print("\nGenerating form HTML...")
            form_html = generate_form_html(feature_info)
            print(form_html)
            
            update = input("\nDo you want to update the HTML template with these features? (y/n): ")
            if update.lower() == 'y':
                update_html_template(feature_info)
        
    else:
        print("Usage: python update_app_features.py <path_to_dataset_file>")
        print("\nSupported file formats: .csv, .xlsx, .xls, .json, .pkl")
