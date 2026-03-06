"""
Comprehensive Model Evaluation Module
Generates detailed metrics, visualizations, and reports
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc, precision_recall_curve)
import joblib
import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path='models/disease_model.pkl'):
        """Initialize evaluator with trained model"""
        logger.info("="*60)
        logger.info("INITIALIZING MODEL EVALUATOR")
        logger.info("="*60)
        
        # Load model
        self.model = joblib.load(model_path)
        logger.info(f"✅ Model loaded from {model_path}")
        
        # Load metadata
        with open('models/metadata.json', 'r') as f:
            self.metadata = json.load(f)
        logger.info(f"✅ Model: {self.metadata['model_name']}")
        logger.info(f"✅ Accuracy: {self.metadata['accuracy']:.4f}")
        
        # Load feature names
        with open('models/feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        # Load label mapping
        with open('models/label_mapping.json', 'r') as f:
            self.label_mapping = json.load(f)
        self.disease_names = list(self.label_mapping.keys())
        
        # Create output directory for plots
        os.makedirs('static', exist_ok=True)
        
    def load_test_data(self):
        """Load test data for evaluation"""
        logger.info("="*60)
        logger.info("LOADING TEST DATA")
        logger.info("="*60)
        
        # Load preprocessed data (using preprocessor)
        from preprocess_data import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        result = preprocessor.run_pipeline()
        
        if result is None:
            logger.error("❌ Failed to load test data")
            return False
        
        self.X_train, self.X_test, self.y_train, self.y_test, _, _ = result
        
        logger.info(f"✅ Test data shape: {self.X_test.shape}")
        return True
    
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        logger.info("="*60)
        logger.info("CALCULATING METRICS")
        logger.info("="*60)
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_prob = self.model.predict_proba(self.X_test)
        
        # Basic metrics
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        
        # Per-class metrics
        self.precision_macro = precision_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        self.recall_macro = recall_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        self.f1_macro = f1_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        
        self.precision_weighted = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        self.recall_weighted = recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        self.f1_weighted = f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        
        # Log metrics
        logger.info(f"Accuracy: {self.accuracy:.4f}")
        logger.info(f"Precision (macro): {self.precision_macro:.4f}")
        logger.info(f"Recall (macro): {self.recall_macro:.4f}")
        logger.info(f"F1-Score (macro): {self.f1_macro:.4f}")
        logger.info(f"Precision (weighted): {self.precision_weighted:.4f}")
        logger.info(f"Recall (weighted): {self.recall_weighted:.4f}")
        logger.info(f"F1-Score (weighted): {self.f1_weighted:.4f}")
        
        # Classification report
        self.class_report = classification_report(
            self.y_test, 
            self.y_pred, 
            target_names=self.disease_names,
            output_dict=True
        )
        
        # Save classification report
        with open('models/classification_report.json', 'w') as f:
            json.dump(self.class_report, f, indent=2)
        
        logger.info("✅ Classification report saved")
        
    def plot_confusion_matrix(self):
        """Plot and save confusion matrix"""
        logger.info("="*60)
        logger.info("GENERATING CONFUSION MATRIX")
        logger.info("="*60)
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(20, 16))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.disease_names[:20] if len(self.disease_names) > 20 else self.disease_names,
                   yticklabels=self.disease_names[:20] if len(self.disease_names) > 20 else self.disease_names)
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save
        path = 'static/confusion_matrix.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Confusion matrix saved to {path}")
        
        # Calculate per-class metrics from confusion matrix
        per_class = []
        for i, disease in enumerate(self.disease_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            per_class.append({
                'disease': disease,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv
            })
        
        # Save per-class metrics
        pd.DataFrame(per_class).to_csv('models/per_class_metrics.csv', index=False)
        logger.info("✅ Per-class metrics saved")
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        logger.info("="*60)
        logger.info("ANALYZING FEATURE IMPORTANCE")
        logger.info("="*60)
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # Get top N features
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(12, 8))
            
            # Create horizontal bar plot
            y_pos = np.arange(len(indices))
            plt.barh(y_pos, importances[indices])
            plt.yticks(y_pos, [self.feature_names[i].replace('_', ' ') for i in indices])
            
            plt.xlabel('Importance Score', fontsize=12)
            plt.title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            # Save
            path = 'static/feature_importance.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Feature importance plot saved to {path}")
            
            # Log top features
            logger.info("\nTop 10 Important Features:")
            for i in range(min(10, len(indices))):
                logger.info(f"  {i+1:2d}. {self.feature_names[indices[i]]:30s}: {importances[indices[i]]:.4f}")
            
            return importances
        else:
            logger.warning("⚠️  Model does not have feature_importances_ attribute")
            return None
    
    def analyze_errors(self):
        """Analyze misclassifications"""
        logger.info("="*60)
        logger.info("ERROR ANALYSIS")
        logger.info("="*60)
        
        # Find misclassified samples
        misclassified = np.where(self.y_test != self.y_pred)[0]
        
        if len(misclassified) > 0:
            logger.info(f"Total misclassifications: {len(misclassified)} out of {len(self.y_test)} ({len(misclassified)/len(self.y_test)*100:.2f}%)")
            
            # Create error matrix
            error_data = []
            for idx in misclassified[:20]:  # Show first 20
                actual = self.disease_names[self.y_test[idx]]
                predicted = self.disease_names[self.y_pred[idx]]
                error_data.append({
                    'actual': actual,
                    'predicted': predicted
                })
            
            # Create confusion pairs
            from collections import Counter
            error_pairs = Counter([(self.disease_names[self.y_test[i]], 
                                   self.disease_names[self.y_pred[i]]) 
                                  for i in misclassified])
            
            logger.info("\nMost common misclassifications:")
            for (actual, predicted), count in error_pairs.most_common(10):
                logger.info(f"  {actual} → {predicted}: {count} times")
            
            # Save error analysis
            error_df = pd.DataFrame(error_data)
            error_df.to_csv('models/error_analysis.csv', index=False)
            logger.info("✅ Error analysis saved")
            
            return len(misclassified)
        else:
            logger.info("✅ No misclassifications found!")
            return 0
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        logger.info("="*60)
        logger.info("GENERATING EVALUATION REPORT")
        logger.info("="*60)
        
        # Compile report
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.metadata['model_name'],
            'model_accuracy': self.metadata['accuracy'],
            'evaluation_metrics': {
                'accuracy': float(self.accuracy),
                'precision_macro': float(self.precision_macro),
                'recall_macro': float(self.recall_macro),
                'f1_macro': float(self.f1_macro),
                'precision_weighted': float(self.precision_weighted),
                'recall_weighted': float(self.recall_weighted),
                'f1_weighted': float(self.f1_weighted)
            },
            'n_classes': len(self.disease_names),
            'n_features': len(self.feature_names),
            'test_samples': len(self.y_test),
            'misclassifications': int(len(np.where(self.y_test != self.y_pred)[0])),
            'classification_report': self.class_report
        }
        
        # Save report
        with open('models/evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("✅ Evaluation report saved to models/evaluation_report.json")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("📊 EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Model: {report['model_name']}")
        logger.info(f"Test Accuracy: {report['evaluation_metrics']['accuracy']:.4f}")
        logger.info(f"F1-Score (macro): {report['evaluation_metrics']['f1_macro']:.4f}")
        logger.info(f"Total misclassifications: {report['misclassifications']}")
        logger.info("="*60)
        
        return report
    
    def run_evaluation_pipeline(self):
        """Run complete evaluation pipeline"""
        logger.info("\n" + "="*60)
        logger.info("🚀 STARTING MODEL EVALUATION PIPELINE")
        logger.info("="*60)
        
        # Step 1: Load test data
        if not self.load_test_data():
            return False
        
        # Step 2: Calculate metrics
        self.calculate_metrics()
        
        # Step 3: Plot confusion matrix
        self.plot_confusion_matrix()
        
        # Step 4: Plot feature importance
        self.plot_feature_importance()
        
        # Step 5: Analyze errors
        self.analyze_errors()
        
        # Step 6: Generate report
        report = self.generate_report()
        
        logger.info("\n" + "="*60)
        logger.info("✅ EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60 + "\n")
        
        return report

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_evaluation_pipeline()