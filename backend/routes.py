"""
API Routes Module
"""
from flask import Blueprint, request, jsonify, render_template
from .predictor import DiseasePredictor
from .doctor_mapper import get_doctor_info, get_severity_color
import json
import logging
from datetime import datetime

# Create blueprint
api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

# Load data
try:
    with open('models/symptoms.json', 'r') as f:
        all_symptoms = json.load(f)
    with open('models/diseases.json', 'r') as f:
        all_diseases = json.load(f)
    logger.info(f"✅ Loaded {len(all_symptoms)} symptoms and {len(all_diseases)} diseases")
except FileNotFoundError:
    logger.error("❌ Model files not found. Please run train_model.py first")
    all_symptoms = []
    all_diseases = []

# Initialize predictor
predictor = DiseasePredictor()

@api_bp.route('/')
def index():
    """Home page with symptom selector"""
    return render_template('index.html', symptoms=all_symptoms)

@api_bp.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({'error': 'Please select at least one symptom'}), 400
        
        # Get prediction
        result = predictor.predict(symptoms)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['symptoms_analyzed'] = len(symptoms)
        
        # Log prediction
        logger.info(f"Prediction: {result['primary']['disease']} "
                   f"({result['primary']['confidence']}%) - {len(symptoms)} symptoms")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

@api_bp.route('/api/symptoms')
@api_bp.route('/api/symptoms/<path:path>')
def get_symptoms(path=None):
    """Get all available symptoms"""
    try:
        # Group symptoms by first letter for better organization
        grouped = {}
        for symptom in all_symptoms:
            first_letter = symptom[0].upper() if symptom else '#'
            if first_letter not in grouped:
                grouped[first_letter] = []
            grouped[first_letter].append(symptom)
        
        # Sort each group
        for letter in grouped:
            grouped[letter].sort()
        
        return jsonify({
            'success': True,
            'symptoms': all_symptoms,
            'grouped': grouped,
            'count': len(all_symptoms)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/diseases')
def get_diseases():
    """Get all diseases with their information"""
    try:
        diseases_info = []
        for disease in all_diseases:
            info = get_doctor_info(disease)
            diseases_info.append({
                'name': disease,
                'specialist': info['specialist'],
                'severity': info['severity'],
                'department': info['department'],
                'color': get_severity_color(info['severity'])
            })
        
        return jsonify({
            'success': True,
            'diseases': diseases_info,
            'count': len(diseases_info)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor.model is not None,
        'symptoms_count': len(all_symptoms),
        'diseases_count': len(all_diseases)
    })

@api_bp.route('/result')
def result_page():
    """Result page for displaying prediction"""
    return render_template('result.html')

@api_bp.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Collect user feedback on predictions"""
    try:
        data = request.get_json()
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'predicted_disease': data.get('predicted_disease'),
            'actual_disease': data.get('actual_disease'),
            'was_correct': data.get('was_correct'),
            'symptoms': data.get('symptoms'),
            'rating': data.get('rating'),
            'comments': data.get('comments')
        }
        
        # Save feedback to file (in production, use database)
        import json
        import os
        
        feedback_file = 'models/feedback.json'
        existing = []
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                existing = json.load(f)
        
        existing.append(feedback)
        
        with open(feedback_file, 'w') as f:
            json.dump(existing, f, indent=2)
        
        return jsonify({'success': True, 'message': 'Feedback received'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500