import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Prepare features and target
X = data.drop(columns='Outcome')
y = data['Outcome']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train models and save them
models = {
    'logistic': LogisticRegression(max_iter=200),
    'decision_tree': DecisionTreeClassifier(),
    'knn': KNeighborsClassifier(),
    'random_forest': RandomForestClassifier(),
    'gradient_boosting': GradientBoostingClassifier(),
    'svc': SVC(),
    'naive_bayes': GaussianNB(),
    'adaboost': AdaBoostClassifier(),
    'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),  # Avoid deprecation warnings
    'extra_trees': ExtraTreesClassifier()
}

for name, model in models.items():
    model.fit(x_train, y_train)
    joblib.dump(model, f"{name}_model.joblib")
    print(f"{name.capitalize()} Model Accuracy: {accuracy_score(y_test, model.predict(x_test)) * 100:.2f}%")
