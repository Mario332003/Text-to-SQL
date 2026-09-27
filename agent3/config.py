import os
from dotenv import load_dotenv

load_dotenv()

# LLM used only for turning the raw API response into a plain-English answer.
MODEL_NAME = "qwen3:8b-q4_K_M"

# Open-Meteo needs no API key/auth at all.
FORECAST_URL = os.environ.get("AGENT3_FORECAST_URL", "https://api.open-meteo.com/v1/forecast")
GEOCODING_URL = os.environ.get("AGENT3_GEOCODING_URL", "https://geocoding-api.open-meteo.com/v1/search")
API_TIMEOUT = float(os.environ.get("AGENT3_API_TIMEOUT", "10"))