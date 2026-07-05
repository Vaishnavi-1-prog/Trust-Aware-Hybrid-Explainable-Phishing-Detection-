# -*- coding: utf-8 -*-
!pip install tensorflow -quite

#import libraries
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

#connect with drive
from google.colab import drive
drive.mount('/content/drive')

#load dataset
df = pd.read_csv("/content/drive/MyDrive/phishing_project_datasets/phishing_email_dataset_CEAS.csv")

# Show first rows
df.head()

# Check dataset shape
print("Dataset shape:", df.shape)

# Check column names
print("\nColumns:")
print(df.columns)

# Check missing values
df.isnull().sum()

# Select email body as text input
texts = df['body'].astype(str)

# Select labels
labels = df['label']

# Check first few values
print(texts.head())
print("\nLabels:\n", labels.head())

#clean the email text

def clean_text(text):

    text = text.lower()  # convert to lowercase

    text = re.sub(r'http\S+', '', text)  # remove URLs

    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove special characters

    text = re.sub(r'\s+', ' ', text)  # remove extra spaces

    return text

# Apply cleaning
texts = texts.apply(clean_text)

# Check cleaned text
print(texts.head())

"""Neural networks cannot understand words directly, so we convert them into integer tokens.

it converts text to numbers , each number represent word index in the vocabulary
"""

#tokenization

# Maximum vocabulary size
max_words = 10000

# Create tokenizer
tokenizer = Tokenizer(num_words=max_words)

# Learn word index
tokenizer.fit_on_texts(texts)

# Convert text → sequences
sequences = tokenizer.texts_to_sequences(texts)

# Check example
print(sequences[:5])

"""Each email sequence will be converted to length = 200.

Example:

Before:

[45, 12, 89]

After padding:

[0,0,0,0,0,...,45,12,89]
"""

#padding the sequence

# maximum email length
max_len = 200

# apply padding
X = pad_sequences(sequences, maxlen=max_len)

# labels
y = labels

# check shape
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

"""Architecture of BiLSTM:
Input (Email Tokens)
        ->
Embedding Layer
        ->
Bidirectional LSTM
        ->
Dropout
        ->
Dense (Sigmoid)
        ->
Prediction (Phishing / Legitimate)


"""

# Build BiLSTM model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

model = Sequential()

# Embedding layer
model.add(Embedding(input_dim=10000, output_dim=128, input_shape=(200,)))

# Bidirectional LSTM layer
model.add(Bidirectional(LSTM(64)))

# Prevent overfitting
model.add(Dropout(0.5))

# Output layer (binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Show model architecture
model.summary()

"""| Parameter            | Meaning                                       |
| -------------------- | --------------------------------------------- |
| epochs=5             | model will learn from the dataset **5 times** |
| batch_size=32        | 32 emails processed at once                   |
| validation_split=0.1 | 10% of training data used for validation      |

"""

#Training the BiLSTM Model

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

"""The model outputs probabilities like:

0.98 → phishing
0.02 → legitimate

We convert them into 0 or 1 using the 0.5 threshold.
"""

#Make Predictions on Test Data

# Predict probabilities
y_pred_prob = model.predict(X_test)

# Convert probabilities to class labels
y_pred = (y_pred_prob > 0.5).astype(int)

print(y_pred[:10])

#evaluate the model - performance metrics of BiLSTM
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

#save the model

model.save("/content/drive/MyDrive/phishing_project_datasets/bilstm_email_model.h5")

print("BiLSTM model saved successfully.")

# ============================================
# SAVE TOKENIZER
# ============================================

import pickle

with open("/content/drive/MyDrive/phishing_project_datasets/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Tokenizer saved successfully.")

import joblib

joblib.dump(X, "/content/drive/MyDrive/phishing_project_datasets/X_test_seq.pkl")

#summary of BiLSTM
model.summary()

import matplotlib.pyplot as plt

# Create a figure with 2 plots side by side
plt.figure(figsize=(12,5))

# -------- Accuracy Plot --------
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('BiLSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train Accuracy','Validation Accuracy'])

# -------- Loss Plot --------
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('BiLSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train Loss','Validation Loss'])

plt.tight_layout()
plt.show()

