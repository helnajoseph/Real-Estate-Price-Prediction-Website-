"""
This script evaluates the accuracy of the Bangalore home prices model.
It calculates various metrics and generates visualizations to assess model performance.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import json
import os

def load_model(model_path='model.pkl'):
    """Load the trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def load_test_data(data_path):
    """Load test data from a CSV file."""
    try:
        # Determine file type based on extension
        ext = os.path.splitext(data_path)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(data_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
            
        print(f"Test data loaded successfully from {data_path}")
        print(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return None

def prepare_features(df, model_features):
    """
    Prepare features for model evaluation.
    
    Args:
        df: DataFrame containing test data
        model_features: List of feature names expected by the model
        
    Returns:
        X: Feature matrix
        y: Target values
    """
    try:
        # Extract target variable (price)
        y = df['price'].values
        
        # Create feature matrix with zeros
        X = np.zeros((len(df), len(model_features)))
        
        # Set numeric features
        X[:, 0] = df['total_sqft'].values
        X[:, 1] = df['bath'].values
        X[:, 2] = df['bhk'].values
        
        # Set location features (one-hot encoding)
        for i, row in df.iterrows():
            loc = row['location']
            try:
                loc_index = model_features.index(loc)
                X[i, loc_index] = 1
            except ValueError:
                print(f"Warning: Location '{loc}' not found in model features")
        
        return X, y
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        return None, None

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred)
    }
    
    # Calculate additional metrics
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    
    return metrics

def perform_cross_validation(model, X, y, cv=5):
    """Perform k-fold cross-validation."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # R² scores
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    
    # RMSE scores
    neg_mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-neg_mse_scores)
    
    # MAE scores
    neg_mae_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    mae_scores = -neg_mae_scores
    
    cv_results = {
        'r2_mean': r2_scores.mean(),
        'r2_std': r2_scores.std(),
        'rmse_mean': rmse_scores.mean(),
        'rmse_std': rmse_scores.std(),
        'mae_mean': mae_scores.mean(),
        'mae_std': mae_scores.std(),
        'r2_scores': r2_scores.tolist(),
        'rmse_scores': rmse_scores.tolist(),
        'mae_scores': mae_scores.tolist()
    }
    
    return cv_results

def create_visualizations(y_true, y_pred, output_dir='static/images'):
    """Create visualizations for model evaluation."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scatter plot of predicted vs. actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Price (lakhs)')
    plt.ylabel('Predicted Price (lakhs)')
    plt.title('Predicted vs. Actual Home Prices')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/predicted_vs_actual.png')
    plt.close()
    
    # 2. Residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price (lakhs)')
    plt.ylabel('Residuals (lakhs)')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residual_plot.png')
    plt.close()
    
    # 3. Histogram of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals (lakhs)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residual_histogram.png')
    plt.close()
    
    # 4. Price distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(y_true, label='Actual', alpha=0.5, color='blue', kde=True)
    sns.histplot(y_pred, label='Predicted', alpha=0.5, color='red', kde=True)
    plt.xlabel('Price (lakhs)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Actual vs. Predicted Prices')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/price_distribution.png')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")
    
    return [
        'predicted_vs_actual.png',
        'residual_plot.png',
        'residual_histogram.png',
        'price_distribution.png'
    ]

def evaluate_model(test_data_path, model_path='model.pkl', output_file='static/model_evaluation.json'):
    """
    Evaluate the model and save results to a JSON file.
    
    Args:
        test_data_path: Path to the test data file
        model_path: Path to the model file
        output_file: Path to save the evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    # Load model and model features
    model = load_model(model_path)
    
    try:
        with open('model_features.json', 'r') as f:
            model_features = json.load(f)
    except Exception as e:
        print(f"Error loading model features: {str(e)}")
        return None
    
    # Load test data
    test_data = load_test_data(test_data_path)
    
    if model is None or test_data is None:
        print("Cannot evaluate model due to errors.")
        return None
    
    # Prepare features
    X, y = prepare_features(test_data, model_features)
    
    if X is None or y is None:
        print("Cannot evaluate model due to errors in feature preparation.")
        return None
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = calculate_metrics(y, y_pred)
    
    # Perform cross-validation
    cv_results = perform_cross_validation(model, X, y)
    
    # Create visualizations
    visualization_files = create_visualizations(y, y_pred)
    
    # Combine results
    evaluation_results = {
        'metrics': metrics,
        'cross_validation': cv_results,
        'visualizations': visualization_files
    }
    
    # Save results to JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"Evaluation results saved to {output_file}")
    
    return evaluation_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate the Bangalore home prices model')
    parser.add_argument('test_data_path', help='Path to the test data file')
    parser.add_argument('--model-path', default='model.pkl', help='Path to the model file')
    parser.add_argument('--output-file', default='static/model_evaluation.json', help='Path to save the evaluation results')
    
    args = parser.parse_args()
    
    results = evaluate_model(args.test_data_path, args.model_path, args.output_file)
    
    if results:
        print("\nModel Evaluation Results:")
        print(f"R² Score: {results['metrics']['r2']:.4f}")
        print(f"RMSE: {results['metrics']['rmse']:.4f} lakhs")
        print(f"MAE: {results['metrics']['mae']:.4f} lakhs")
        print(f"MAPE: {results['metrics']['mape']:.2f}%")
        
        print("\nCross-Validation Results:")
        print(f"Mean R²: {results['cross_validation']['r2_mean']:.4f} ± {results['cross_validation']['r2_std']:.4f}")
        print(f"Mean RMSE: {results['cross_validation']['rmse_mean']:.4f} ± {results['cross_validation']['rmse_std']:.4f} lakhs")
        print(f"Mean MAE: {results['cross_validation']['mae_mean']:.4f} ± {results['cross_validation']['mae_std']:.4f} lakhs")
