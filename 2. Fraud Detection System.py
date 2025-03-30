"""
नीचे Fraud Detection System का एक पूरा Python Code दिया गया है, जो Machine Learning का उपयोग करके Credit Card Fraud को डिटेक्ट करता है। इसमें हम Logistic Regression का उपयोग करेंगे, लेकिन आप इसे Random Forest या Neural Networks से भी Improve कर सकते हैं।

🚀 Step-by-Step Implementation
✅ Step 1: Dataset Load करें
✅ Step 2: Data Preprocessing करें
✅ Step 3: Model Train करें (Logistic Regression)
✅ Step 4: Model Evaluate करें
✅ Step 5: Real-time Fraud Detection करें

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
url = "https://www.kaggle.com/mlg-ulb/creditcardfraud/download"
df = pd.read_csv(url)  

# Display First 5 Rows
print(df.head())


# Check for missing values
print(df.isnull().sum())

# Balance Dataset (Fraud Cases कम होते हैं, इसलिए Undersampling करेंगे)
fraud_cases = df[df['Class'] == 1]
non_fraud_cases = df[df['Class'] == 0].sample(n=len(fraud_cases))  # Non-fraud Cases को कम करना

balanced_df = pd.concat([fraud_cases, non_fraud_cases])  # Fraud और Non-Fraud Cases को Balance किया

# Feature Selection
X = balanced_df.drop(columns=['Class'])  # Features
y = balanced_df['Class']  # Target Variable (0 = Not Fraud, 1 = Fraud)

# Data Splitting (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Scaling (Normalization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Logistic Regression Model Train करना
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction on Test Data
y_pred = model.predict(X_test)

# Accuracy Check
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))



# नया Transaction का Data लें
new_transaction = np.array([[1000, 0.2, -0.3, 1.2, 0.5, -1.0, 1.3, -0.2, 0.1, 0.3]])  # Example Data
new_transaction = scaler.transform(new_transaction)  # Scaling करें

# Prediction करें
fraud_prediction = model.predict(new_transaction)
if fraud_prediction[0] == 1:
    print("⚠️ Alert: Fraudulent Transaction Detected!")
else:
    print("✅ Transaction is Safe.")



