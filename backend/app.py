"""
Main Flask Application
"""
from flask import Flask, jsonify
from flask_cors import CORS
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
import os
from .config import get_config
from .routes import api_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

# Load configuration
app.config.from_object(get_config())

# Initialize extensions
CORS(app)  # Enable CORS
cache = Cache(app)  # Enable caching

# FIXED: Limiter initialization
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=app.config.get('REDIS_URL', 'memory://')
)

# Register blueprints
app.register_blueprint(api_bp, url_prefix='/')

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(429)
def ratelimit_error(error):
    """Handle rate limit errors"""
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = app.config['DEBUG']
    app.run(host='0.0.0.0', port=port, debug=debug)