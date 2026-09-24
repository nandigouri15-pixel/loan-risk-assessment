import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the CSV file
df = pd.read_csv(r"C:\Users\Sheela Nandi\Downloads\loan_data.csv")
print(df)
print(f"Dataset securely loaded. Shape: {df.shape}")
print(df.isnull().sum())

# Check target class distribution (1 = Approved/Default Risk, 0 = Rejected)
print("--- Target Column Distribution ---")
distribution = df['loan_status'].value_counts(normalize=True) * 100
print(f"Approved/Risk (1): {distribution[1]:.2f}%")
print(f"Rejected (0): {distribution[0]:.2f}%")

# Separate features (X) and target variable (y)
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# One-Hot Encode categorical strings (person_gender, person_education, loan_intent, etc.)
# drop_first=True prevents the 'dummy variable trap' (multicollinearity)
X = pd.get_dummies(X, drop_first=True)

print(f"Features converted successfully. New shape after encoding: {X.shape}")
print("\nEncoded Columns Preview:")
print(X.columns.tolist()[:10])  # Displays the first 10 engineered feature variables

# Train-Test Split (80% Training Data, 20% Testing Data)
# random_state ensures reproducible and identical outcomes when demonstrating to a recruiter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# Initialize Random Forest with 100 trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
print("Random Forest Classifier successfully trained on the loan dataset.")

# Generate structural predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("==============================================")
print(f"✨ Production Model Accuracy: {accuracy * 100:.2f}%")
print("==============================================")
print("\n📋 Advanced Evaluation Metrics (Classification Report):")
print(classification_report(y_test, y_pred))

print("\n📦 Confusion Matrix Details:")
print(f"True Negatives (Correct Rejections): {conf_matrix[0][0]}")
print(f"False Positives (Type I Error): {conf_matrix[0][1]}")
print(f"False Negatives (Type II Error): {conf_matrix[1][0]}")
print(f"True Positives (Correct Approvals): {conf_matrix[1][1]}")


# Extract feature importance values from our trained classifier
importances = model.feature_importances_
feature_names = X.columns
model_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Display numerical overview
print("\n--- Top 5 Most Crucial Variables for Credit Scoring ---")
print(model_importances.head(5))

# Plot the features beautifully
plt.figure(figsize=(10, 6))
model_importances.head(10).plot(kind='barh', color='skyblue')
plt.gca().invert_yaxis()  # Put the most important feature at the top
plt.title("Loan Approval Prediction Model - Top 10 Feature Importances", fontsize=14, fontweight='bold')
plt.xlabel("Relative Importance Score", fontsize=12)
plt.ylabel("Dataset Variables", fontsize=12)
plt.tight_layout()
plt.show()

import joblib

# Save the trained model and features as a dictionary file
model_artifacts = {
    'model': model,
    'features': list(X.columns)
}

joblib.dump(model_artifacts, 'loan_model_artifacts.pkl')
print("✅ Production artifacts securely saved to 'loan_model_artifacts.pkl'!")


