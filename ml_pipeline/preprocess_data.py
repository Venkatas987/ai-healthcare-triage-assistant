"""
Advanced Data Preprocessing Module
Handles data validation, cleaning, feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        """Initialize the preprocessor"""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.symptom_categories = {}
        self.train_df = None
        self.test_df = None
        self.feature_names = []
        
    def load_data(self, train_path='dataset/Training.csv', test_path='dataset/Testing.csv'):
        """Load datasets from CSV files"""
        logger.info("="*60)
        logger.info("LOADING DATASETS")
        logger.info("="*60)
        
        try:
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)
            
            logger.info(f"✅ Training data: {self.train_df.shape[0]} rows, {self.train_df.shape[1]} columns")
            logger.info(f"✅ Testing data: {self.test_df.shape[0]} rows, {self.test_df.shape[1]} columns")
            
            # Basic validation
            assert self.train_df.shape[1] == self.test_df.shape[1], "Column count mismatch!"
            assert self.train_df.columns[-1] == 'prognosis', "Last column must be 'prognosis'!"
            
            return True
            
        except FileNotFoundError as e:
            logger.error(f"❌ File not found: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def explore_data(self):
        """Explore dataset statistics"""
        logger.info("="*60)
        logger.info("DATA EXPLORATION")
        logger.info("="*60)
        
        # Basic info
        logger.info(f"Features: {self.train_df.shape[1] - 1} symptoms")
        logger.info(f"Diseases: {self.train_df['prognosis'].nunique()}")
        
        # Check data types
        dtypes = self.train_df.dtypes.value_counts()
        logger.info(f"Data types: {dict(dtypes)}")
        
        # Check for missing values
        train_missing = self.train_df.isnull().sum().sum()
        test_missing = self.test_df.isnull().sum().sum()
        logger.info(f"Missing values - Train: {train_missing}, Test: {test_missing}")
        
        # Class distribution
        disease_counts = self.train_df['prognosis'].value_counts()
        logger.info(f"\nTop 10 diseases by frequency:")
        for disease, count in disease_counts.head(10).items():
            logger.info(f"  {disease}: {count} samples")
        
        # Most common symptoms
        symptom_cols = self.train_df.columns[:-1]
        symptom_sums = self.train_df[symptom_cols].sum().sort_values(ascending=False)
        logger.info(f"\nTop 10 most common symptoms:")
        for symptom, count in symptom_sums.head(10).items():
            percentage = (count / len(self.train_df)) * 100
            logger.info(f"  {symptom}: {count} ({percentage:.1f}%)")
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        logger.info("="*60)
        logger.info("HANDLING MISSING VALUES")
        logger.info("="*60)
        
        train_missing = self.train_df.isnull().sum().sum()
        test_missing = self.test_df.isnull().sum().sum()
        
        if train_missing > 0 or test_missing > 0:
            logger.warning(f"Missing values found - Train: {train_missing}, Test: {test_missing}")
            
            # Fill with 0 (since symptoms are binary)
            self.train_df = self.train_df.fillna(0)
            self.test_df = self.test_df.fillna(0)
            
            logger.info("✅ Filled missing values with 0")
        else:
            logger.info("✅ No missing values found")
    
    def validate_binary_features(self):
        """Ensure all symptom features are binary (0/1)"""
        logger.info("="*60)
        logger.info("VALIDATING BINARY FEATURES")
        logger.info("="*60)
        
        symptom_cols = self.train_df.columns[:-1]
        non_binary = []
        
        for col in symptom_cols:
            unique_values = self.train_df[col].unique()
            if not all(v in [0, 1] for v in unique_values):
                non_binary.append(col)
        
        if non_binary:
            logger.warning(f"Found {len(non_binary)} non-binary columns")
            # Convert to binary
            for col in non_binary:
                self.train_df[col] = (self.train_df[col] > 0).astype(int)
                self.test_df[col] = (self.test_df[col] > 0).astype(int)
            logger.info("✅ Converted all features to binary")
        else:
            logger.info("✅ All features are binary")
    
    def create_symptom_categories(self):
        """Group symptoms into medical categories for feature engineering"""
        logger.info("="*60)
        logger.info("CREATING SYMPTOM CATEGORIES")
        logger.info("="*60)
        
        # Define symptom categories based on medical knowledge
        self.symptom_categories = {
            'respiratory': ['cough', 'breathlessness', 'phlegm', 'runny_nose', 'sinus_pressure', 
                           'congestion', 'mucoid_sputum', 'rusty_sputum', 'chest_pain'],
            'gastrointestinal': ['stomach_pain', 'acidity', 'vomiting', 'indigestion', 'nausea', 
                                'abdominal_pain', 'diarrhoea', 'constipation', 'passage_of_gases'],
            'neurological': ['headache', 'dizziness', 'loss_of_balance', 'unsteadiness', 
                            'slurred_speech', 'weakness_of_one_body_side', 'loss_of_smell',
                            'altered_sensorium', 'coma'],
            'dermatological': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'ulcers_on_tongue',
                              'yellowish_skin', 'red_spots_over_body', 'dischromic_patches',
                              'bruising', 'skin_peeling'],
            'systemic': ['fatigue', 'weight_loss', 'weight_gain', 'fever', 'high_fever', 
                        'mild_fever', 'sweating', 'chills', 'shivering', 'malaise'],
            'urinary': ['burning_micturition', 'spotting_urination', 'bladder_discomfort',
                       'foul_smell_of_urine', 'continuous_feel_of_urine', 'polyuria'],
            'cardiovascular': ['fast_heart_rate', 'palpitations', 'chest_pain', 
                              'swollen_blood_vessels', 'prominent_veins_on_calf'],
            'musculoskeletal': ['joint_pain', 'muscle_wasting', 'muscle_weakness', 'knee_pain',
                               'hip_joint_pain', 'stiff_neck', 'swelling_joints', 'movement_stiffness']
        }
        
        # Add category columns to dataframe
        categories_added = 0
        for category, symptoms in self.symptom_categories.items():
            # Find which symptoms exist in our dataset
            existing_symptoms = [s for s in symptoms if s in self.train_df.columns]
            if existing_symptoms:
                self.train_df[f'category_{category}'] = self.train_df[existing_symptoms].sum(axis=1)
                self.test_df[f'category_{category}'] = self.test_df[existing_symptoms].sum(axis=1)
                categories_added += 1
                logger.info(f"✅ Created category '{category}' with {len(existing_symptoms)} symptoms")
        
        logger.info(f"✅ Added {categories_added} category features")
    
    def encode_target(self):
        """Encode disease labels to numeric values"""
        logger.info("="*60)
        logger.info("ENCODING TARGET VARIABLE")
        logger.info("="*60)
        
        # Combine all unique diseases
        all_diseases = pd.concat([
            self.train_df['prognosis'],
            self.test_df['prognosis']
        ]).unique()
        
        # Fit encoder
        self.label_encoder.fit(all_diseases)
        
        # Transform
        self.train_df['prognosis_encoded'] = self.label_encoder.transform(self.train_df['prognosis'])
        self.test_df['prognosis_encoded'] = self.label_encoder.transform(self.test_df['prognosis'])
        
        logger.info(f"✅ Encoded {len(all_diseases)} unique diseases")
        
        # Save label mapping
        label_mapping = dict(zip(
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_).tolist()
        ))
        
        os.makedirs('models', exist_ok=True)
        with open('models/label_mapping.json', 'w') as f:
            json.dump(label_mapping, f, indent=2)
        
        logger.info("✅ Saved label mapping to models/label_mapping.json")
    
    def check_class_balance(self):
        """Check if classes are balanced"""
        logger.info("="*60)
        logger.info("CLASS BALANCE CHECK")
        logger.info("="*60)
        
        class_counts = self.train_df['prognosis'].value_counts()
        
        min_count = class_counts.min()
        max_count = class_counts.max()
        ratio = max_count / min_count
        
        logger.info(f"Minimum samples per class: {min_count}")
        logger.info(f"Maximum samples per class: {max_count}")
        logger.info(f"Imbalance ratio: {ratio:.2f}")
        
        if ratio > 2:
            logger.warning("⚠️  Class imbalance detected! Consider using class weights.")
        else:
            logger.info("✅ Classes are reasonably balanced")
        
        return class_counts
    
    def prepare_features(self):
        """Prepare final feature sets for training"""
        logger.info("="*60)
        logger.info("PREPARING FEATURES")
        logger.info("="*60)
        
        # Original symptom columns
        symptom_cols = [col for col in self.train_df.columns 
                       if col not in ['prognosis', 'prognosis_encoded'] 
                       and not col.startswith('category_')]
        
        # Category features
        category_cols = [col for col in self.train_df.columns if col.startswith('category_')]
        
        # Combine all features
        self.feature_names = symptom_cols + category_cols
        logger.info(f"Total features: {len(self.feature_names)}")
        logger.info(f"  - Original symptoms: {len(symptom_cols)}")
        logger.info(f"  - Category features: {len(category_cols)}")
        
        # Create feature matrices
        X_train = self.train_df[self.feature_names].values
        y_train = self.train_df['prognosis_encoded'].values
        
        X_test = self.test_df[self.feature_names].values
        y_test = self.test_df['prognosis_encoded'].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"✅ Training data shape: {X_train_scaled.shape}")
        logger.info(f"✅ Testing data shape: {X_test_scaled.shape}")
        
        # Save feature names
        with open('models/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        # Save symptom names (original)
        with open('models/symptoms.json', 'w') as f:
            json.dump(symptom_cols, f, indent=2)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, symptom_cols
    
    def run_pipeline(self):
        """Run complete preprocessing pipeline"""
        logger.info("\n" + "="*60)
        logger.info("🚀 STARTING PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Load data
        if not self.load_data():
            return None
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Handle missing values
        self.handle_missing_values()
        
        # Step 4: Validate binary features
        self.validate_binary_features()
        
        # Step 5: Create symptom categories
        self.create_symptom_categories()
        
        # Step 6: Check class balance
        self.check_class_balance()
        
        # Step 7: Encode target
        self.encode_target()
        
        # Step 8: Prepare features
        X_train, X_test, y_train, y_test, symptom_cols = self.prepare_features()
        
        logger.info("="*60)
        logger.info("✅ PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info("="*60 + "\n")
        
        return X_train, X_test, y_train, y_test, symptom_cols, self.label_encoder

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_pipeline()