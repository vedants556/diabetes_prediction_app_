from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load all models
try:
    models = {
        'logistic': joblib.load('logistic_model.joblib'),
        'decision_tree': joblib.load('decision_tree_model.joblib'),
        'knn': joblib.load('knn_model.joblib'),
        'random_forest': joblib.load('best_random_forest_model.joblib'),
        'gradient_boosting': joblib.load('gradient_boosting_model.joblib'),
        'svc': joblib.load('svc_model.joblib'),
        'naive_bayes': joblib.load('naive_bayes_model.joblib'),
        'adaboost': joblib.load('adaboost_model.joblib'),
        'xgboost': joblib.load('xgboost_model.joblib'),
        'extra_trees': joblib.load('extra_trees_model.joblib')
    }
except FileNotFoundError as e:
    raise RuntimeError("Model files not found. Please ensure they are in the correct location.") from e

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # The model is now always 'decision_tree'
        model_choice = 'random_forest'

        # Extract and validate input features from the form
        features = []
        fields = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                  'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
        
        for field in fields:
            value = request.form.get(field)
            if value is None:
                return render_template('index.html', prediction="Missing input for {field}.")
            try:
                features.append(float(value))
            except ValueError:
                return render_template('index.html', prediction=f"Invalid input for {field}. Please enter numeric values.")

        # Create feature array
        features_array = np.array([features])

        # Use the logistic regression model
        model = models[model_choice]

        # Make a prediction
        prediction = model.predict(features_array)

        # Return the result
        result = "1" if prediction[0] == 1 else "0"
        
        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction="error", error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
