import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("AGENT5_DATA_DIR", BASE_DIR / "data"))

MODEL_NAME = os.environ.get("AGENT5_MODEL_NAME", "qwen3:8b-q4_K_M")
TOP_K = int(os.environ.get("AGENT5_TOP_K", "6"))
CHUNK_SIZE = int(os.environ.get("AGENT5_CHUNK_SIZE", "1400"))
CHUNK_OVERLAP = int(os.environ.get("AGENT5_CHUNK_OVERLAP", "250"))
