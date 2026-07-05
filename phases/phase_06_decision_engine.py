# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

# Phase 6 : Final Decision Engine
# Input values

# Hybrid model phishing probability (output from Phase 3 hybrid model)
hybrid_probability = 0.82

# Trust score from Phase 5
trust_score = 0.35

print("Hybrid Model Probability :", hybrid_probability)
print("Trust Score :", trust_score)

# Define weights


alpha = 0.7   # Hybrid model weight
beta = 0.3    # Trust score weight

print("Hybrid Weight (alpha) :", alpha)
print("Trust Weight (beta) :", beta)

# Calculate Final Risk Score


risk_final = (alpha * hybrid_probability) + (beta * (1 - trust_score))

print("Final Risk Score :", risk_final)

"""Final Classification Logic

We now classify the email based on the Final Risk Score.

| Risk Score    | Classification   |
| ------------- | ---------------- |
| **≥ 0.7**     | PHISHING         |
| **0.4 – 0.7** | SPAM (Ignore)    |
| **< 0.4**     | LEGITIMATE EMAIL |

"""

#Final Decision Logic

if risk_final >= 0.7:
    final_prediction = "PHISHING"

elif risk_final >= 0.4:
    final_prediction = "SPAM – Ignore"

else:
    final_prediction = "LEGITIMATE EMAIL"

print("Final Classification :", final_prediction)

"""This step will display all important values together:

Hybrid Model Probability

Trust Score

Final Risk Score

Final Classification
"""

#Final System Output

print("\n----- FINAL PHISHING DETECTION RESULT -----")

print("Hybrid Model Probability :", round(hybrid_probability, 3))
print("Trust Score              :", round(trust_score, 3))
print("Final Risk Score         :", round(risk_final, 3))

print("\nFinal Classification :", final_prediction)

print("-------------------------------------------")

