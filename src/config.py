"""Central configuration for Hiver AI Support & Triage Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
GOLDEN_SET_PATH = DATA_DIR / "golden_eval_set.jsonl"
HUMAN_ANNOTATIONS_PATH = DATA_DIR / "human_annotations_sample.jsonl"
REPORT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "REPORT.md"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds & Constraints
MIN_INTENT_CONFIDENCE: float = float(os.getenv("MIN_INTENT_CONFIDENCE", "0.35"))
MIN_RETRIEVAL_SIMILARITY: float = float(os.getenv("MIN_RETRIEVAL_SIMILARITY", "0.40"))
FRUSTRATION_THRESHOLD: float = float(os.getenv("FRUSTRATION_THRESHOLD", "0.60"))
MAX_TWEET_CHARS: int = 280

# Embedding & LLM Configuration
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "mock"
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# Target Brand
TARGET_BRAND: str = "@AppleSupport"
