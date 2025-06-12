import os
from dotenv import load_dotenv

load_dotenv()

"""
Configuration settings for the RAG system.
"""

# Mode
DEBUG = True

# File paths
CHROMA_PATH = "chroma"
FAISS_INDEX_PATH = "faiss.index"
JSON_PATH = "docs/mdgspace_dataset.json"

# Embedding model configuration
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL")

# LLM configuration
LLM_API_URL = os.getenv("OPENROUTER_ENDPOINT")
HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEYS')}",
    "Content-Type": "application/json",
}

LLM_MODEL = "deepseek/deepseek-r1:free"
LLM_TEMPERATURE = 0.85
LLM_TOP_P = 0.9
LLM_TOP_K = 10

# Retrieval configuration
DEFAULT_RELEVANCE_THRESHOLD = 0.6
DEFAULT_RETRIEVAL_K = 10

# Hugging Face Creds
HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")