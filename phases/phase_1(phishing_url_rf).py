# -*- coding: utf-8 -*-

#connect with G-drive
from google.colab import drive
drive.mount('/content/drive')

#import libraries
import pandas as pd
import numpy as np

#loading the dataset and view first 5 rows
path = "/content/drive/MyDrive/phishing_project_datasets/phishing_url_features_large.csv"

df = pd.read_csv(path)

print(df.shape)
df.head()

#finding the number of null values
df.isnull().sum()

#feature/label splt
X = df.drop("Type", axis=1)
y = df["Type"]

print(X.shape)
print(y.shape)

#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

"""n_estimators - defines the number of decision trees to build before aggregating their predictions

random_state - It ensures that every time you run your code, the "random" processes follow the exact same sequence, leading to reproducible results.
"""

#train random foresting
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("Model trained")

#model evaluation - Random Forest
from sklearn.metrics import classification_report

y_pred = rf_model.predict(X_test)

print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = rf_model.predict(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred)
}
plt.bar(metrics.keys(), metrics.values())
plt.title("Random Forest Performance")
plt.ylim(0.9,1.0)
plt.ylabel("Score")
plt.show()

"""The purpose of this code is to find which URL features are most important for detecting phishing websites.

Random Forest automatically calculates something called: feature_importances_

This tells how much each feature contributed to the prediction.

Explainable AI Framework:
This means the system should not just predict phishing — it should explain why.

Feature importance helps show:
Which characteristics of the URL indicate phishing behavior.
"""

#feature importance
import pandas as pd

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance.head(10))

import joblib

model_path = "/content/drive/MyDrive/phishing_project_datasets/rf_phishing_model.pkl"

joblib.dump(rf_model, model_path)

print("Model saved in Google Drive")

import joblib

joblib.dump(X_test, "/content/drive/MyDrive/phishing_project_datasets/X_test_url.pkl")
joblib.dump(y_test, "/content/drive/MyDrive/phishing_project_datasets/y_test.pkl")

