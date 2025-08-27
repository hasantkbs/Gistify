from fastapi import FastAPI
import firebase_admin
from firebase_admin import credentials
import os
from dotenv import load_dotenv
from .routes import router as api_router

# Load environment variables from .env file
load_dotenv()

# Initialize Firebase Admin SDK
def initialize_firebase():
    cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_PATH")
    if not cred_path:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_KEY_PATH environment variable not set.")
    if not os.path.exists(cred_path):
        raise FileNotFoundError(f"Firebase service account key file not found at: {cred_path}")
    
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

app = FastAPI(title="Gistify API")

@app.on_event("startup")
async def startup_event():
    try:
        initialize_firebase()
        print("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        print(f"Error initializing Firebase Admin SDK: {e}")
        # Depending on the use case, you might want to exit the application
        # if Firebase initialization fails.

app.include_router(api_router)