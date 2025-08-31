from fastapi import FastAPI
import os
from dotenv import load_dotenv
from .routes import router as api_router
from .database import initialize_firebase_db # Import from new database file

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Gistify API")

@app.on_event("startup")
async def startup_event():
    try:
        initialize_firebase_db() # Call the function from the new database file
        print("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Firebase Admin SDK failed to initialize: {e}")
        raise # Re-raise the exception to prevent the application from starting

app.include_router(api_router)