import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model names for the summarization pipelines
ENGLISH_MODEL_NAME = os.getenv("ENGLISH_MODEL_NAME", "google/pegasus-cnn_dailymail")
TURKISH_MODEL_NAME = os.getenv("TURKISH_MODEL_NAME", "ozcangundes/mt5-small-turkish-summarization")
MT5_MODEL_NAME = os.getenv("MT5_MODEL_NAME", "google/mt5-small")
