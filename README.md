# Real Estate Price Predictor Web Application

This is a web application that uses a machine learning model to predict real estate prices based on various features of a property.

## Setup Instructions

1. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Prepare your model**:
   - Make sure your trained model is saved as `model.pkl` in the root directory
   - The model should be compatible with the features defined in the web form

3. **Run the application**:
   ```
   python app.py
   ```

4. **Access the application**:
   - Open your web browser and go to `http://127.0.0.1:5000/`

## Features

- User-friendly web interface for inputting property details
- Real-time prediction using your trained model
- Responsive design that works on desktop and mobile devices

## Customizing the Application

If your model uses different features than the ones provided in the form, you'll need to:

1. Update the form in `templates/index.html` to match your model's required features
2. Adjust the data processing in the `/predict` route in `app.py`

## Troubleshooting

- If you see a "Model not loaded" error, make sure your model file exists at the specified path
- For other issues, check the Flask server logs for detailed error messages
