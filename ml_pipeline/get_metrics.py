"""
get_metrics.py - Comprehensive Model Evaluation Script
Run this after training to see your model's actual performance metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)
import joblib
import json
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*70)
print("📊 AI HEALTHCARE TRIAGE ASSISTANT - MODEL METRICS")
print("="*70)

# Step 1: Load the trained model
print("\n🔍 Step 1: Loading trained model...")
try:
    model = joblib.load('models/disease_model.pkl')
    print("   ✅ Model loaded successfully")
    print(f"   📌 Model type: {type(model).__name__}")
except FileNotFoundError:
    print("   ❌ Model not found! Please train the model first.")
    print("   💡 Run: python ml_pipeline/train_model.py")
    exit(1)

# Step 2: Load test data
print("\n📂 Step 2: Loading test dataset...")
try:
    test_df = pd.read_csv('dataset/Testing.csv')
    print(f"   ✅ Test data loaded: {len(test_df)} samples")
    print(f"   📌 Test shape: {test_df.shape}")
except FileNotFoundError:
    print("   ❌ Testing.csv not found in dataset folder!")
    exit(1)

# Step 3: Load symptoms list
print("\n📋 Step 3: Loading symptoms list...")
try:
    with open('models/symptoms.json', 'r') as f:
        symptoms = json.load(f)
    print(f"   ✅ Symptoms loaded: {len(symptoms)} features")
except FileNotFoundError:
    print("   ⚠️  symptoms.json not found, using columns from test data")
    symptoms = list(test_df.columns[:-1])
    print(f"   ✅ Using {len(symptoms)} features from test data")

# Step 4: Prepare test data
print("\n🔄 Step 4: Preparing test data...")
X_test = test_df[symptoms]
y_test = test_df['prognosis']
print(f"   ✅ X_test shape: {X_test.shape}")
print(f"   ✅ y_test shape: {y_test.shape}")
print(f"   📌 Unique diseases in test: {y_test.nunique()}")

# Step 5: Make predictions
print("\n🤖 Step 5: Making predictions...")
y_pred = model.predict(X_test)
print(f"   ✅ Predictions completed for {len(y_pred)} samples")

# Step 6: Calculate metrics
print("\n📈 Step 6: Calculating performance metrics...")

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

# Calculate correct predictions
correct = (y_pred == y_test).sum()
total = len(y_test)
misclassified = total - correct
accuracy_percent = accuracy * 100

print("\n" + "="*70)
print("📊 MODEL PERFORMANCE SUMMARY")
print("="*70)
print(f"\n🎯 ACCURACY: {accuracy_percent:.2f}% ({correct}/{total})")
print(f"\n📊 WEIGHTED AVERAGE METRICS:")
print(f"   ✅ Precision: {precision_weighted*100:.2f}%")
print(f"   ✅ Recall:    {recall_weighted*100:.2f}%")
print(f"   ✅ F1-Score:  {f1_weighted*100:.2f}%")
print(f"\n📊 MACRO AVERAGE METRICS:")
print(f"   ✅ Precision: {precision_macro*100:.2f}%")
print(f"   ✅ Recall:    {recall_macro*100:.2f}%")
print(f"   ✅ F1-Score:  {f1_macro*100:.2f}%")
print("="*70)

# Step 7: Detailed analysis
print("\n🔍 Step 7: Detailed error analysis...")

if misclassified > 0:
    print(f"\n⚠️  Found {misclassified} misclassified samples:")
    
    # Find misclassified indices
    misclassified_indices = np.where(y_pred != y_test)[0]
    
    # Create error analysis table
    error_analysis = []
    for idx in misclassified_indices:
        error_analysis.append({
            'Index': idx,
            'Actual': y_test.iloc[idx],
            'Predicted': y_pred[idx]
        })
    
    # Display errors
    error_df = pd.DataFrame(error_analysis)
    print("\n📋 Misclassification Details:")
    for i, row in error_df.iterrows():
        print(f"   {i+1}. Actual: {row['Actual']} → Predicted: {row['Predicted']}")
    
    # Save error analysis
    error_df.to_csv('models/error_analysis.csv', index=False)
    print("\n   ✅ Error analysis saved to models/error_analysis.csv")
else:
    print("\n   ✅ No misclassifications found! Perfect score!")

# Step 8: Classification report
print("\n📋 Step 8: Generating detailed classification report...")
class_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
print(classification_report(y_test, y_pred, zero_division=0))

# Step 9: Save metrics to file
print("\n💾 Step 9: Saving metrics to file...")

metrics = {
    'timestamp': datetime.now().isoformat(),
    'model_type': type(model).__name__,
    'test_samples': int(total),
    'correct_predictions': int(correct),
    'misclassifications': int(misclassified),
    'accuracy': float(accuracy),
    'accuracy_percent': float(accuracy_percent),
    'precision_weighted': float(precision_weighted),
    'recall_weighted': float(recall_weighted),
    'f1_weighted': float(f1_weighted),
    'precision_macro': float(precision_macro),
    'recall_macro': float(recall_macro),
    'f1_macro': float(f1_macro),
    'classification_report': class_report
}

# Save as JSON
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("   ✅ Metrics saved to models/metrics.json")

# Save as readable text file
with open('models/metrics_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("AI HEALTHCARE TRIAGE ASSISTANT - MODEL METRICS REPORT\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*70 + "\n\n")
    f.write(f"Model Type: {type(model).__name__}\n")
    f.write(f"Test Samples: {total}\n")
    f.write(f"Correct Predictions: {correct}\n")
    f.write(f"Misclassifications: {misclassified}\n\n")
    f.write(f"ACCURACY: {accuracy_percent:.2f}%\n\n")
    f.write("WEIGHTED METRICS:\n")
    f.write(f"  Precision: {precision_weighted*100:.2f}%\n")
    f.write(f"  Recall:    {recall_weighted*100:.2f}%\n")
    f.write(f"  F1-Score:  {f1_weighted*100:.2f}%\n\n")
    f.write("MACRO METRICS:\n")
    f.write(f"  Precision: {precision_macro*100:.2f}%\n")
    f.write(f"  Recall:    {recall_macro*100:.2f}%\n")
    f.write(f"  F1-Score:  {f1_macro*100:.2f}%\n")
print("   ✅ Report saved to models/metrics_report.txt")

# Step 10: Confusion matrix (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\n📊 Step 10: Generating confusion matrix visualization...")
    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(y_test, y_pred)
    
    # Get unique disease labels
    labels = y_test.unique()
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels[:10] if len(labels) > 10 else labels,
                yticklabels=labels[:10] if len(labels) > 10 else labels)
    
    plt.title('Confusion Matrix (Top 10 Diseases)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    plt.savefig('static/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Confusion matrix saved to static/confusion_matrix.png")
except Exception as e:
    print(f"   ⚠️  Could not generate confusion matrix: {e}")

print("\n" + "="*70)
print("✅ METRICS CALCULATION COMPLETE!")
print("="*70)
print("\n📁 Files generated:")
print("   • models/metrics.json - JSON format for applications")
print("   • models/metrics_report.txt - Human-readable report")
print("   • models/error_analysis.csv - Misclassification details")
print("   • static/confusion_matrix.png - Visual confusion matrix")
print("\n🚀 Next step: Run 'python -m backend.app' to start the application")
print("="*70)