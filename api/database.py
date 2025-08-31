import firebase_admin
from firebase_admin import credentials, firestore_async
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Global Firestore client
_db = None # Use a private name for the global variable

def initialize_firebase_db():
    global _db
    cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_PATH")
    print(f"Attempting to load Firebase credentials from: {cred_path}")
    if not cred_path:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_KEY_PATH environment variable not set.")
    if not os.path.exists(cred_path):
        raise FileNotFoundError(f"Firebase service account key file not found at: {cred_path}")
    
    cred = credentials.Certificate(cred_path)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    _db = firestore_async.client() # Use the async client
    print("Firestore client (_db) initialized.")

def get_firestore_db():
    if _db is None:
        raise RuntimeError("Firestore client not initialized. Call initialize_firebase_db() first.")
    return _db