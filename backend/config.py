"""
Configuration Management Module
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = False
    TESTING = False
    
    # Application
    APP_NAME = "AI Healthcare Triage Assistant"
    APP_VERSION = "2.0.0"
    
    # Model paths
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/disease_model.pkl')
    SYMPTOMS_PATH = os.getenv('SYMPTOMS_PATH', 'models/symptoms.json')
    DISEASES_PATH = os.getenv('DISEASES_PATH', 'models/diseases.json')
    
    # API Settings
    API_PREFIX = '/api'
    API_TITLE = 'Healthcare Triage API'
    API_DESCRIPTION = 'AI-powered disease prediction and specialist recommendation'
    
    # Rate Limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = "200 per day;50 per hour"
    RATELIMIT_STORAGE_URL = os.getenv('REDIS_URL', 'memory://')
    
    # Caching
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'SimpleCache')
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', 300))
    CACHE_THRESHOLD = 1000
    
    # Security
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Feature flags
    ENABLE_FEEDBACK = os.getenv('ENABLE_FEEDBACK', 'True').lower() == 'true'
    ENABLE_ANALYTICS = os.getenv('ENABLE_ANALYTICS', 'True').lower() == 'true'
    ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'True').lower() == 'true'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    FLASK_ENV = 'development'
    SESSION_COOKIE_SECURE = False  # Allow HTTP in development
    CACHE_TYPE = 'SimpleCache'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = 'production'
    
    # Override with production settings
    SECRET_KEY = os.getenv('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY must be set in production")
    
    # Use Redis for rate limiting and caching in production
    RATELIMIT_STORAGE_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_TYPE = 'RedisCache'
    CACHE_REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/1')
    
    # Stricter security
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '').split(',')
    if not CORS_ORIGINS or CORS_ORIGINS == ['']:
        raise ValueError("CORS_ORIGINS must be set in production")

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    FLASK_ENV = 'testing'
    SESSION_COOKIE_SECURE = False
    RATELIMIT_ENABLED = False
    CACHE_TYPE = 'NullCache'

# Configuration dictionary
CONFIG = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get the current configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    return CONFIG.get(env, CONFIG['default'])