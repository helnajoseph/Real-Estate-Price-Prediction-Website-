This is a complete, end-to-end Data Science and Web Development project where we build a **Real Estate Price Prediction** website using **Python, Machine Learning, and Flask**. The app allows users to input property details such as area, location, number of bedrooms and bathrooms, and receive a predicted price using a trained ML model.

---

## 🚀 Project Workflow

### 📊 1. Data Analysis & Cleaning
- Used **Pandas** and **NumPy** to load and explore the Bangalore house prices dataset from Kaggle.
- Identified and removed:
  - Duplicates
  - Outliers based on area, price/sqft
  - Irrelevant or inconsistent entries

### 🧠 2. Model Building
- Applied **Feature Engineering**:
  - One-Hot Encoding for categorical variables (like location)
  - Combined similar locations with low frequency
- Used **Scikit-learn** to train a **Linear Regression** model.
- Performed:
  - **GridSearchCV** for hyperparameter tuning
  - **K-Fold Cross Validation** for model validation
- Saved the trained model using **joblib/pickle** as `model.pkl`.

### 🔧 3. Flask API Server
- Built a Flask backend (`app.py`) to:
  - Load the model and feature data
  - Accept HTTP requests at `/api/predict`
  - Return JSON responses with predicted prices
- Modularized the code:
  - `load_dataset.py`, `export_model.py`, `evaluate_model.py`, etc. for clean pipeline

### 🌐 4. Web Interface
- Created the UI using **HTML, CSS, JavaScript** inside the `templates/` and `static/` folders.
- The form collects input (location, area, bedrooms, bathrooms) and sends it to Flask using `fetch` API (AJAX).
- Displays the predicted price instantly on the web page.

---

## 🗂️ Project Structure

```

├── app.py                   # Flask server
├── model.pkl                # Trained ML model
├── requirements.txt         # Python dependencies
├── README.md                # Project overview
├── data/                    # (Optional) Sample data
├── templates/               # HTML files (frontend)
├── static/                  # CSS, JS files
├── explore\_dataset.py       # EDA scripts
├── load\_dataset.py          # Dataset loader
├── analyze\_model.py         # Model interpretation
├── export\_model.py          # Saves model to .pkl
├── evaluate\_model.py        # Accuracy, CV
├── model\_features.py        # List of model features
├── update\_app\_features.py   # Sync features for dropdown
├── sample\_test\_data.xlsx    # Manual test inputs

````

---

## 📦 Installation & Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/real-estate-price-predictor.git
cd real-estate-price-predictor
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Flask App

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## 🌍 Deployment Suggestions

For production deployment:

* Host on **AWS EC2** (Ubuntu server)
* Use **Gunicorn** as WSGI server
* Setup **Nginx** as reverse proxy
* Map `/api/` to your Flask app and serve frontend from `static/` and `templates/`

---

## 📚 Technologies Used

| Area           | Tools Used                       |
| -------------- | -------------------------------- |
| Language       | Python                           |
| Data Handling  | Pandas, NumPy                    |
| Visualization  | Matplotlib                       |
| Model Building | Scikit-learn (Linear Regression) |
| Model Tuning   | GridSearchCV, Cross Validation   |
| Backend Server | Flask                            |
| Frontend       | HTML, CSS, JavaScript            |
| Deployment     | AWS EC2, Nginx, Gunicorn         |

---

## ✨ Future Enhancements

* Add more ML models (e.g., Random Forest, XGBoost)
* Use real-time location APIs for dynamic dropdowns
* Add model performance dashboard
* Secure endpoints using authentication

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

* Dataset from [Kaggle: Bangalore Home Prices](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)



