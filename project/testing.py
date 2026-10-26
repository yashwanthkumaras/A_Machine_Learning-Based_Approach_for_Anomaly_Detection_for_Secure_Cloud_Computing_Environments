import joblib
import numpy as np

# Load saved model and feature list
model = joblib.load("best_ids_model.pkl")
selected_features = joblib.load("selected_features.pkl")

print("\n📥 Enter values for the following features:")
user_input = []

# Collect user inputs for each feature
for feature in selected_features:
    while True:
        try:
            val = float(input(f"{feature}: "))
            user_input.append(val)
            break
        except ValueError:
            print("Please enter a valid number.")

# Convert to proper shape and predict
X_input = np.array(user_input).reshape(1, -1)
prediction = model.predict(X_input)[0]

# Output
print("\n✅ Prediction Result:")
print("Prediction:", "Attack" if prediction == 1 else "Normal ")
