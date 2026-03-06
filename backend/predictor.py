"""
Disease Prediction Module
"""
import joblib
import numpy as np
import json
from .doctor_mapper import get_doctor_info

class DiseasePredictor:
    def __init__(self):
        """Initialize the predictor with trained model"""
        try:
            self.model = joblib.load('models/disease_model.pkl')
            with open('models/symptoms.json', 'r') as f:
                self.symptoms = json.load(f)
            with open('models/diseases.json', 'r') as f:
                self.diseases = json.load(f)
            print(f"✅ Model loaded: {len(self.diseases)} diseases, {len(self.symptoms)} symptoms")
        except FileNotFoundError as e:
            print(f"❌ Model files not found: {e}")
            print("Please run ml_pipeline/train_model.py first")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, selected_symptoms):
        """
        Predict disease based on selected symptoms
        
        Args:
            selected_symptoms: List of symptom names
            
        Returns:
            Dictionary with predictions and recommendations
        """
        # Create feature vector (all zeros)
        features = np.zeros(len(self.symptoms))
        
        # Mark selected symptoms as 1
        matched_count = 0
        for symptom in selected_symptoms:
            if symptom in self.symptoms:
                idx = self.symptoms.index(symptom)
                features[idx] = 1
                matched_count += 1
        
        # If no symptoms matched the model's feature set
        if matched_count == 0:
            return {
                'primary': {
                    'disease': 'Insufficient Symptoms',
                    'confidence': 0,
                    'specialist': 'General Physician',
                    'severity': 'Unknown',
                    'department': 'General Medicine'
                },
                'alternatives': [],
                'symptoms_count': len(selected_symptoms),
                'matched_count': 0,
                'warning': 'Selected symptoms not recognized by the model'
            }
        
        # Get prediction and probabilities
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3 = []
        
        for idx in top_3_idx:
            disease = self.diseases[idx]
            confidence = float(probabilities[idx] * 100)
            doctor_info = get_doctor_info(disease)
            
            top_3.append({
                'disease': disease,
                'confidence': round(confidence, 2),
                'specialist': doctor_info['specialist'],
                'severity': doctor_info['severity'],
                'department': doctor_info['department']
            })
        
        # Get symptom category analysis
        symptom_categories = self._analyze_categories(selected_symptoms)
        
        return {
            'primary': top_3[0],
            'alternatives': top_3[1:],
            'symptoms_count': len(selected_symptoms),
            'matched_count': matched_count,
            'symptoms_used': selected_symptoms,
            'categories': symptom_categories
        }
    
    def _analyze_categories(self, symptoms):
        """Analyze which body systems are affected"""
        categories = {
            'respiratory': 0,
            'gastrointestinal': 0,
            'neurological': 0,
            'dermatological': 0,
            'systemic': 0
        }
        
        # Keywords for each category
        keywords = {
            'respiratory': ['cough', 'breath', 'phlegm', 'nose', 'sinus', 'chest', 'wheeze'],
            'gastrointestinal': ['stomach', 'acidity', 'vomit', 'nausea', 'diarrhoea', 'constipation', 'abdomen'],
            'neurological': ['head', 'dizzy', 'balance', 'speech', 'weakness', 'numbness', 'vertigo'],
            'dermatological': ['itch', 'rash', 'skin', 'yellow', 'red', 'spot', 'lesion', 'nail'],
            'systemic': ['fatigue', 'fever', 'chills', 'sweat', 'weight', 'appetite', 'malaise']
        }
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for category, words in keywords.items():
                if any(word in symptom_lower for word in words):
                    categories[category] += 1
        
        # Determine primary system affected
        primary_system = max(categories, key=categories.get)
        if categories[primary_system] == 0:
            primary_system = 'general'
        
        return {
            'counts': categories,
            'primary_system': primary_system
        }
    
    def batch_predict(self, symptoms_list):
        """Predict for multiple symptom sets"""
        results = []
        for symptoms in symptoms_list:
            results.append(self.predict(symptoms))
        return results