import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Load .env file if it exists
ENV_PATH = BASE_DIR / '.env'
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# Serverless Read-Only Filesystem Adaptor (Vercel / AWS Lambda)
IS_SERVERLESS = bool(os.getenv('VERCEL') or os.getenv('AWS_LAMBDA_FUNCTION_NAME'))
STORAGE_DIR = Path('/tmp') if IS_SERVERLESS else BASE_DIR

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb+srv://admin:admin123@cluster0.mongodb.net/?retryWrites=true&w=majority')
DB_NAME = os.getenv('DB_NAME', 'biometric_fingerprint_db')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'users')

# Storage Directories
UPLOAD_FOLDER = STORAGE_DIR / 'uploads'
FINGERPRINT_FOLDER = UPLOAD_FOLDER / 'fingerprints'
SAMPLE_FOLDER = UPLOAD_FOLDER / 'samples'
DATASET_FOLDER = STORAGE_DIR / os.getenv('DATASET_DIR', 'dataset')

# Safely create writable directories
for d in (UPLOAD_FOLDER, FINGERPRINT_FOLDER, SAMPLE_FOLDER, DATASET_FOLDER):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# Application Configuration
PORT = int(os.getenv('PORT', 5000))
DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
SECRET_KEY = os.getenv('SECRET_KEY', 'biometric_fingerprint_secure_2026')
MATCH_THRESHOLD = float(os.getenv('MATCH_THRESHOLD', 0.55))
MATCHER_ENGINE = os.getenv('MATCHER_ENGINE', 'HYBRID_CNN_SIFT')
MAX_FINGERPRINTS_PER_USER = int(os.getenv('MAX_FINGERPRINTS_PER_USER', 10))
ENROLL_PASSWORD = os.getenv('ENROLL_PASSWORD', 'psr123')

# Model Settings
CNN_IMAGE_SIZE = (128, 128)
EMBEDDING_DIM = 256
