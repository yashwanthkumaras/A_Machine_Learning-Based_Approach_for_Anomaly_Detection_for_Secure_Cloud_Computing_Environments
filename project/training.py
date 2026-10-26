import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# === Load Dataset ===
df = pd.read_csv('UNSW_NB15_training-set.csv')
print("Shape of dataset:", df.shape)

# Drop missing values and unwanted columns
df.dropna(inplace=True)
df.drop(['id', 'attack_cat', 'proto', 'service', 'state'], axis=1, inplace=True)

# Define features and target
X = df.drop('label', axis=1)
y = df['label']
print("Target classes:", y.unique())

# === Feature Selection: Top 20 Features ===
selector = SelectKBest(score_func=f_classif, k=20)
X_new = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
print("Selected Features:\n", selected_features)

# Save selected feature names and selector object
joblib.dump(selected_features, 'selected_features.pkl')
joblib.dump(selector, 'feature_selector.pkl')

# Use only selected features
X = X[selected_features]

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Define Models ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbor": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "MLP": MLPClassifier(max_iter=300)
}

# === Training and Evaluation ===
best_model = None
best_accuracy = 0

os.makedirs("reports", exist_ok=True)
os.makedirs("confusion_matrices", exist_ok=True)

# Classification reports file
with open("reports/classification_reports.txt", "w") as report_file:
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

        # Classification Report
        report = classification_report(y_test, y_pred)
        report_file.write(f"\n{name} Classification Report:\n")
        report_file.write(report)
        report_file.write("\n" + "-" * 60 + "\n")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format='d')
        plt.title(f"{name} Confusion Matrix")
        plt.savefig(f"confusion_matrices/{name.replace(' ', '_')}_confusion_matrix.png")
        plt.close()

        print(f"\n{name} Accuracy: {acc:.4f}")
        print(report)

# === Save Best Model ===
joblib.dump(best_model, 'best_ids_model.pkl')
print(f"\n✅ Best model saved: {best_model_name} with accuracy {best_accuracy:.4f}")
