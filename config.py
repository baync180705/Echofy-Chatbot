"""
Configuration settings for the RAG system.
"""

# File paths
CHROMA_PATH = "chroma"
FAISS_INDEX_PATH = "faiss.index"
JSON_PATH = "docs/mdgspace_dataset.json"

# Embedding model configuration
DEFAULT_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"

# LLM configuration
LLM_API_URL = "http://127.0.0.1:11434/api/generate"
LLM_MODEL = "llama3.2:1b"
LLM_TEMPERATURE = 0.85
LLM_TOP_P = 0.9
LLM_TOP_K = 10
LLM_MAX_TOKENS = 256

# Retrieval configuration
DEFAULT_RELEVANCE_THRESHOLD = 0.4
DEFAULT_RETRIEVAL_K = 3
