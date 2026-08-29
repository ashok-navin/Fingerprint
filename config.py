import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Load .env file
ENV_PATH = BASE_DIR / '.env'
load_dotenv(dotenv_path=ENV_PATH)

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb+srv://admin:admin123@cluster0.mongodb.net/?retryWrites=true&w=majority')
DB_NAME = os.getenv('DB_NAME', 'biometric_fingerprint_db')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'users')

# Storage Directories
UPLOAD_FOLDER = BASE_DIR / 'uploads'
FINGERPRINT_FOLDER = UPLOAD_FOLDER / 'fingerprints'
SAMPLE_FOLDER = UPLOAD_FOLDER / 'samples'
DATASET_FOLDER = BASE_DIR / os.getenv('DATASET_DIR', 'dataset')

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
FINGERPRINT_FOLDER.mkdir(exist_ok=True)
SAMPLE_FOLDER.mkdir(exist_ok=True)
DATASET_FOLDER.mkdir(exist_ok=True)

# Application Configuration
PORT = int(os.getenv('PORT', 5000))
DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 't')
SECRET_KEY = os.getenv('SECRET_KEY', 'biometric_fingerprint_secure_2026')
MATCH_THRESHOLD = float(os.getenv('MATCH_THRESHOLD', 0.55))
MATCHER_ENGINE = os.getenv('MATCHER_ENGINE', 'HYBRID_CNN_SIFT')
MAX_FINGERPRINTS_PER_USER = int(os.getenv('MAX_FINGERPRINTS_PER_USER', 10))
ENROLL_PASSWORD = os.getenv('ENROLL_PASSWORD', 'psr123')

# Model Settings
CNN_IMAGE_SIZE = (128, 128)
EMBEDDING_DIM = 256

