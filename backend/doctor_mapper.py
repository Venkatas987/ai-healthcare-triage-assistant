"""
Doctor and Specialist Mapping Module
Contains comprehensive mapping of diseases to medical specialists
"""

# Complete mapping for all 41 diseases
DOCTOR_MAPPING = {
    'Fungal infection': {
        'specialist': 'Dermatologist', 
        'severity': 'Low', 
        'department': 'Dermatology',
        'description': 'Fungal infections of skin, nails, or mucous membranes',
        'warning_signs': 'Spreading rash, pain, discharge'
    },
    'Allergy': {
        'specialist': 'Allergist/Immunologist', 
        'severity': 'Low', 
        'department': 'Immunology',
        'description': 'Immune system reaction to harmless substances',
        'warning_signs': 'Difficulty breathing, swelling of face/lips'
    },
    'GERD': {
        'specialist': 'Gastroenterologist', 
        'severity': 'Medium', 
        'department': 'Gastroenterology',
        'description': 'Chronic acid reflux affecting esophagus',
        'warning_signs': 'Difficulty swallowing, weight loss, vomiting blood'
    },
    'Chronic cholestasis': {
        'specialist': 'Hepatologist', 
        'severity': 'High', 
        'department': 'Hepatology',
        'description': 'Reduced bile flow from liver',
        'warning_signs': 'Severe itching, dark urine, jaundice'
    },
    'Drug Reaction': {
        'specialist': 'Clinical Pharmacologist', 
        'severity': 'High', 
        'department': 'Clinical Pharmacology',
        'description': 'Adverse effects from medication',
        'warning_signs': 'Severe rash, difficulty breathing, fever'
    },
    'Peptic ulcer disease': {
        'specialist': 'Gastroenterologist', 
        'severity': 'Medium', 
        'department': 'Gastroenterology',
        'description': 'Sores in stomach or duodenum lining',
        'warning_signs': 'Black/tarry stools, vomiting blood, severe pain'
    },
    'AIDS': {
        'specialist': 'Infectious Disease Specialist', 
        'severity': 'Critical', 
        'department': 'Infectious Diseases',
        'description': 'Advanced HIV infection',
        'warning_signs': 'Recurrent infections, weight loss, night sweats'
    },
    'Diabetes': {
        'specialist': 'Endocrinologist', 
        'severity': 'High', 
        'department': 'Endocrinology',
        'description': 'High blood sugar levels',
        'warning_signs': 'Extreme thirst, frequent urination, confusion'
    },
    'Gastroenteritis': {
        'specialist': 'Gastroenterologist', 
        'severity': 'Medium', 
        'department': 'Gastroenterology',
        'description': 'Stomach and intestinal inflammation',
        'warning_signs': 'Severe dehydration, bloody stools, high fever'
    },
    'Bronchial Asthma': {
        'specialist': 'Pulmonologist', 
        'severity': 'High', 
        'department': 'Pulmonology',
        'description': 'Chronic airway inflammation',
        'warning_signs': 'Severe shortness of breath, blue lips'
    },
    'Hypertension': {
        'specialist': 'Cardiologist', 
        'severity': 'High', 
        'department': 'Cardiology',
        'description': 'High blood pressure',
        'warning_signs': 'Severe headache, chest pain, vision problems'
    },
    'Migraine': {
        'specialist': 'Neurologist', 
        'severity': 'Medium', 
        'department': 'Neurology',
        'description': 'Severe recurring headaches',
        'warning_signs': 'Sudden severe headache, fever, stiff neck'
    },
    'Cervical spondylosis': {
        'specialist': 'Orthopedist/Neurologist', 
        'severity': 'Medium', 
        'department': 'Orthopedics',
        'description': 'Age-related neck spine degeneration',
        'warning_signs': 'Loss of bladder/bowel control, weakness'
    },
    'Paralysis (brain hemorrhage)': {
        'specialist': 'Neurologist/Neurosurgeon', 
        'severity': 'Critical', 
        'department': 'Neurology',
        'description': 'Bleeding in brain causing paralysis',
        'warning_signs': 'Sudden severe headache, loss of consciousness'
    },
    'Jaundice': {
        'specialist': 'Hepatologist', 
        'severity': 'High', 
        'department': 'Hepatology',
        'description': 'Yellowing of skin and eyes',
        'warning_signs': 'Confusion, bleeding, severe abdominal pain'
    },
    'Malaria': {
        'specialist': 'Infectious Disease Specialist', 
        'severity': 'High', 
        'department': 'Infectious Diseases',
        'description': 'Mosquito-borne parasitic infection',
        'warning_signs': 'High fever, confusion, seizures'
    },
    'Chicken pox': {
        'specialist': 'Dermatologist/Infectious Disease', 
        'severity': 'Medium', 
        'department': 'Dermatology',
        'description': 'Viral infection with itchy blisters',
        'warning_signs': 'Difficulty breathing, severe headache'
    },
    'Dengue': {
        'specialist': 'Infectious Disease Specialist', 
        'severity': 'High', 
        'department': 'Infectious Diseases',
        'description': 'Mosquito-borne viral infection',
        'warning_signs': 'Severe abdominal pain, bleeding gums'
    },
    'Typhoid': {
        'specialist': 'Infectious Disease Specialist', 
        'severity': 'High', 
        'department': 'Infectious Diseases',
        'description': 'Bacterial infection from contaminated food/water',
        'warning_signs': 'Intestinal bleeding, perforation'
    },
    'hepatitis A': {
        'specialist': 'Hepatologist', 
        'severity': 'High', 
        'department': 'Hepatology',
        'description': 'Liver infection from contaminated food/water',
        'warning_signs': 'Acute liver failure symptoms'
    },
    'Hepatitis B': {
        'specialist': 'Hepatologist', 
        'severity': 'High', 
        'department': 'Hepatology',
        'description': 'Chronic liver infection from blood/body fluids',
        'warning_signs': 'Liver cirrhosis, liver cancer'
    },
    'Hepatitis C': {
        'specialist': 'Hepatologist', 
        'severity': 'High', 
        'department': 'Hepatology',
        'description': 'Chronic liver infection from blood exposure',
        'warning_signs': 'Liver failure, cirrhosis'
    },
    'Hepatitis D': {
        'specialist': 'Hepatologist', 
        'severity': 'High', 
        'department': 'Hepatology',
        'description': 'Requires hepatitis B for infection',
        'warning_signs': 'Severe acute hepatitis'
    },
    'Hepatitis E': {
        'specialist': 'Hepatologist', 
        'severity': 'High', 
        'department': 'Hepatology',
        'description': 'Liver infection from contaminated water',
        'warning_signs': 'Acute liver failure in pregnancy'
    },
    'Alcoholic hepatitis': {
        'specialist': 'Hepatologist', 
        'severity': 'High', 
        'department': 'Hepatology',
        'description': 'Liver inflammation from alcohol abuse',
        'warning_signs': 'Severe jaundice, ascites, confusion'
    },
    'Tuberculosis': {
        'specialist': 'Pulmonologist', 
        'severity': 'High', 
        'department': 'Pulmonology',
        'description': 'Bacterial infection mostly affecting lungs',
        'warning_signs': 'Coughing blood, chest pain, weight loss'
    },
    'Common Cold': {
        'specialist': 'General Physician', 
        'severity': 'Low', 
        'department': 'General Medicine',
        'description': 'Viral upper respiratory infection',
        'warning_signs': 'High fever, severe headache, difficulty breathing'
    },
    'Pneumonia': {
        'specialist': 'Pulmonologist', 
        'severity': 'High', 
        'department': 'Pulmonology',
        'description': 'Lung infection causing inflammation',
        'warning_signs': 'Difficulty breathing, chest pain, high fever'
    },
    'Dimorphic hemorrhoids(piles)': {
        'specialist': 'Proctologist', 
        'severity': 'Medium', 
        'department': 'Proctology',
        'description': 'Swollen veins in anus and rectum',
        'warning_signs': 'Heavy bleeding, severe pain, thrombosis'
    },
    'Heart attack': {
        'specialist': 'Cardiologist', 
        'severity': 'Critical', 
        'department': 'Cardiology',
        'description': 'Blocked blood flow to heart muscle',
        'warning_signs': 'Chest pain, shortness of breath, sweating'
    },
    'Varicose veins': {
        'specialist': 'Vascular Surgeon', 
        'severity': 'Medium', 
        'department': 'Vascular Surgery',
        'description': 'Enlarged, twisted veins',
        'warning_signs': 'Leg ulcers, bleeding, skin changes'
    },
    'Hypothyroidism': {
        'specialist': 'Endocrinologist', 
        'severity': 'Medium', 
        'department': 'Endocrinology',
        'description': 'Underactive thyroid gland',
        'warning_signs': 'Severe fatigue, depression, weight gain'
    },
    'Hyperthyroidism': {
        'specialist': 'Endocrinologist', 
        'severity': 'Medium', 
        'department': 'Endocrinology',
        'description': 'Overactive thyroid gland',
        'warning_signs': 'Rapid heartbeat, weight loss, anxiety'
    },
    'Hypoglycemia': {
        'specialist': 'Endocrinologist', 
        'severity': 'High', 
        'department': 'Endocrinology',
        'description': 'Low blood sugar',
        'warning_signs': 'Confusion, loss of consciousness, seizures'
    },
    'Osteoarthritis': {
        'specialist': 'Orthopedist/Rheumatologist', 
        'severity': 'Medium', 
        'department': 'Orthopedics',
        'description': 'Joint cartilage degeneration',
        'warning_signs': 'Severe joint pain, deformity'
    },
    'Arthritis': {
        'specialist': 'Rheumatologist', 
        'severity': 'High', 
        'department': 'Rheumatology',
        'description': 'Joint inflammation',
        'warning_signs': 'Joint deformity, organ involvement'
    },
    '(vertigo) Paroxysmal Positional Vertigo': {
        'specialist': 'ENT Specialist', 
        'severity': 'Medium', 
        'department': 'ENT',
        'description': 'Brief episodes of vertigo from head movements',
        'warning_signs': 'Hearing loss, neurological symptoms'
    },
    'Acne': {
        'specialist': 'Dermatologist', 
        'severity': 'Low', 
        'department': 'Dermatology',
        'description': 'Skin condition with pimples',
        'warning_signs': 'Severe cystic acne, scarring'
    },
    'Urinary tract infection': {
        'specialist': 'Urologist', 
        'severity': 'Medium', 
        'department': 'Urology',
        'description': 'Bacterial infection in urinary system',
        'warning_signs': 'Fever, back pain, blood in urine'
    },
    'Psoriasis': {
        'specialist': 'Dermatologist', 
        'severity': 'Medium', 
        'department': 'Dermatology',
        'description': 'Autoimmune skin condition',
        'warning_signs': 'Joint pain, widespread involvement'
    },
    'Impetigo': {
        'specialist': 'Dermatologist', 
        'severity': 'Low', 
        'department': 'Dermatology',
        'description': 'Contagious bacterial skin infection',
        'warning_signs': 'Fever, spreading infection'
    }
}

def get_doctor_info(disease):
    """
    Get doctor information for a disease
    
    Args:
        disease: Name of the disease
        
    Returns:
        Dictionary with specialist, severity, department, description, warning_signs
    """
    return DOCTOR_MAPPING.get(disease, {
        'specialist': 'General Physician',
        'severity': 'Medium',
        'department': 'General Medicine',
        'description': 'Please consult a doctor for accurate diagnosis',
        'warning_signs': 'Seek immediate medical attention if symptoms worsen'
    })

def get_severity_color(severity):
    """Get color code for severity level"""
    colors = {
        'Low': '#4caf50',      # Green
        'Medium': '#ff9800',    # Orange
        'High': '#f44336',      # Red
        'Critical': '#9c27b0',  # Purple
        'Unknown': '#757575'    # Gray
    }
    return colors.get(severity, '#757575')

def get_urgency_level(severity):
    """Get urgency level based on severity"""
    urgency = {
        'Low': 'Routine (within 1-2 weeks)',
        'Medium': 'Soon (within 2-3 days)',
        'High': 'Urgent (within 24 hours)',
        'Critical': 'Emergency (immediate care needed)'
    }
    return urgency.get(severity, 'Consult as soon as possible')

def get_specialties_by_department(department):
    """Get all specialists in a department"""
    specialists = []
    for disease, info in DOCTOR_MAPPING.items():
        if info['department'] == department and info['specialist'] not in specialists:
            specialists.append(info['specialist'])
    return specialists