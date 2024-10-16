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
        'random_forest': joblib.load('random_forest_model.joblib'),
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
        # Get the model choice from the form
        model_choice = request.form.get('model')
        if model_choice not in models:
            return render_template('index.html', prediction="Invalid model selected.")

        # Extract and validate input features from the form
        features = []
        fields = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                  'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
        
        for field in fields:
            value = request.form.get(field)
            if value is None:
                return render_template('index.html', prediction=f"Missing input for {field}.")
            try:
                features.append(float(value))
            except ValueError:
                return render_template('index.html', prediction=f"Invalid input for {field}. Please enter numeric values.")

        # Create feature array
        features_array = np.array([features])

        # Retrieve the selected model
        model = models[model_choice]

        # Make a prediction
        prediction = model.predict(features_array)

        # Return the result
        result = "You have diabetes." if prediction[0] == 1 else "You do not have diabetes."
        
        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
