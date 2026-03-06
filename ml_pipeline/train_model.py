"""
Simple Model Training Module - Direct approach
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 Training Model - Direct Approach")
print("="*60)

# Load data directly
print("\n📊 Loading datasets...")
train_df = pd.read_csv('dataset/Training.csv')
test_df = pd.read_csv('dataset/Testing.csv')

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")
print(f"Training columns: {train_df.shape[1]}")
print(f"Testing columns: {test_df.shape[1]}")

# Fix column mismatch - use only common columns
common_columns = list(set(train_df.columns) & set(test_df.columns))
print(f"\nCommon columns: {len(common_columns)}")

# Remove 'prognosis' from features
feature_columns = [col for col in common_columns if col != 'prognosis']
print(f"Feature columns: {len(feature_columns)}")

# Prepare data
X_train = train_df[feature_columns]
y_train = train_df['prognosis']
X_test = test_df[feature_columns]
y_test = test_df['prognosis']

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Unique diseases in train: {y_train.nunique()}")
print(f"Unique diseases in test: {y_test.nunique()}")

# Train model
print("\n🧠 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/disease_model.pkl')

# Save symptoms
symptoms = list(X_train.columns)
with open('models/symptoms.json', 'w') as f:
    json.dump(symptoms, f, indent=2)

# Save diseases
diseases = list(model.classes_)
with open('models/diseases.json', 'w') as f:
    json.dump(diseases, f, indent=2)

print(f"\n✅ Model saved to models/disease_model.pkl")
print(f"✅ Symptoms saved: {len(symptoms)}")
print(f"✅ Diseases saved: {len(diseases)}")
print("\n" + "="*60)