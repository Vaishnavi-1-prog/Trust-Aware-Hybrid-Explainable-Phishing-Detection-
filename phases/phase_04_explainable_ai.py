# -*- coding: utf-8 -*-
!pip install shap -quiet
!pip install lime -quiet

# ==============================
# Phase 4 – Explainability (Fast)
# ==============================

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from google.colab import drive

drive.mount('/content/drive')

# ==============================
# Load Dataset
# ==============================

data_path = "/content/drive/MyDrive/phishing_project_datasets/phishing_url_features_large.csv"
df = pd.read_csv(data_path)

print("Dataset Loaded:", df.shape)

# Separate features and label
X = df.drop("Type", axis=1)
y = df["Type"]

# ==============================
# Load Random Forest Model
# ==============================

model_path = "/content/drive/MyDrive/phishing_project_datasets/rf_phishing_model.pkl"
rf_model = joblib.load(model_path)

print("Random Forest Model Loaded")

# ==============================
# Feature Importance
# ==============================

importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))

# ==============================
# Plot Feature Importance
# ==============================

plt.figure(figsize=(10,6))
plt.barh(feature_importance_df["Feature"][:10][::-1],
         feature_importance_df["Importance"][:10][::-1])

plt.xlabel("Importance Score")
plt.title("Top 10 Features Influencing Phishing Detection")
plt.show()



import os

folder_path = "/content/drive/MyDrive/phishing_project_datasets"

print(os.listdir(folder_path))

print(df.columns)

