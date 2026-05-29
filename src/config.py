"""Central configuration for paths and model settings."""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
OCR_TEXT_DIR = DATA_DIR / "ocr_text"
CHUNKS_DIR = DATA_DIR / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GEMINI_MODEL = "gemini-1.5-flash"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# OCR settings
# Backends: "easyocr" (default) or "paddleocr"
OCR_BACKEND = os.getenv("OCR_BACKEND", "easyocr").lower()
OCR_LANG = "en"
OCR_USE_ANGLE_CLS = True
OCR_MIN_CONFIDENCE = 0.5
OCR_PARAGRAPH_GAP_MULTIPLIER = 1.3

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
