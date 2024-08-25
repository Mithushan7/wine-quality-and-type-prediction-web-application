from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the models
wine_quality_model = joblib.load('RFC_wine_quality_model.pkl')
wine_type_model = joblib.load('RFC_wine_type_binary_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        # Define the feature names
        feature_names = [
                 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                        'pH', 'sulphates', 'alcohol'
            ]
        # Extract features from form
        features = [
            float(request.form['fixed acidity']),
            float(request.form['volatile acidity']),
            float(request.form['citric acid']),
            float(request.form['residual sugar']),
            float(request.form['chlorides']),
            float(request.form['free Sulfur Dioxide']),
            float(request.form['total sulfur dioxide']),
            float(request.form['density']),
            float(request.form['pH']),
            float(request.form['sulphates']),
            float(request.form['alcohol']),
            # Add more features if needed
        ]
        
        # Convert to numpy array
        features_df = pd.DataFrame([features], columns=feature_names)
        features_array = np.array([features])

        # Predict wine quality
        quality_prediction = wine_quality_model.predict(features_df)[0]
        threshold = 7
        wine_quality_threshold = 'Best' if quality_prediction >= 7 else 'Good'
        
        # Predict wine type
        type_prediction = wine_type_model.predict(features_df)[0]
        wine_type = 'Red' if type_prediction == 0 else 'White'

        return render_template('predict.html', quality=quality_prediction, wine_type=wine_type, wine_quality=wine_quality_threshold)
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
