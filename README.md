# Echofy: MDG Space Knowledge Assistant

## Introduction

Echofy is an intelligent question-answering system built on a Retrieval-Augmented Generation (RAG) workflow. Designed specifically for MDG Space, this chatbot processes knowledge base content, stores it in vector databases, and uses advanced embedding models combined with LLM technology to generate accurate, contextually relevant responses to user queries.

## Features

- **Dual Vector Storage**: Utilizes both Chroma and FAISS for efficient text retrieval
- **High-Quality Embeddings**: Powered by NoMic AI's embedding model for superior semantic understanding
- **Context-Aware Responses**: Generates answers using only the most relevant retrieved information
- **Relevance Scoring**: Filters results by confidence score to ensure high-quality responses
- **Modular Architecture**: Easily extendable and maintainable code structure

## Architecture

Echofy employs a modern RAG architecture with the following components:

- **Data Processing**: Loads and processes QA pairs from MDG Space dataset
- **Embedding Generation**: Converts text to semantic vector representations
- **Vector Storage**: Stores embeddings in both Chroma and FAISS for redundancy and performance
- **Similarity Search**: Finds the most relevant content based on user queries
- **Response Generation**: Uses a local LLM to generate natural language responses

## Technologies and Tools Used

Echofy is built using a modern tech stack that combines state-of-the-art tools for embedding, retrieval, and language generation:

- **[nomic-ai/nomic-embed-text-v1](https://docs.nomic.ai/docs/embedding-models/overview)**  
  Used for generating high-quality text embeddings with excellent semantic understanding.

- **[Llama3.2:1b](https://ollama.com/library/llama3)** (via [Ollama](https://ollama.com/))  
  Local LLM used for generating natural, context-aware responses based on the retrieved content.

- **[ChromaDB](https://docs.trychroma.com/)**  
  A fast and scalable open-source vector database used for storing and retrieving text embeddings.

- **[FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)**  
  Efficient similarity search library used in parallel with ChromaDB to improve redundancy and performance.

- **[LangChain](https://www.langchain.com/)**  
  Framework for developing applications powered by LLMs; used to manage chaining between embedding, retrieval, and generation components.

These tools work together to provide a robust Retrieval-Augmented Generation (RAG) pipeline, ensuring accurate and relevant answers based on the MDG Space knowledge base.

## Setup Instructions

### Prerequisites

Before you start, ensure you have the following installed:

- Python 3.8 or higher
- `pip` for package management
- Local LLM server (Ollama running llama3.2:1b model)

### Clone the Repository

First, clone the repository to your local machine:

```bash
# Using HTTPS
git clone https://github.com/baync180705/Echofy-Chatbot.git

#OR

# Using SSH
git clone git@github.com:baync180705/Echofy-Chatbot.git

cd echofy
```

#### Create a Virtual Environment

For Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Run the Project

```bash
# Run with a specific query
python main.py -q "What is MDG Space?"

# Or run interactively
python main.py
```

## Project Structure

```
echofy/
├── .gitignore                   # Git ignore file
├── config.py                    # Configuration settings
├── main.py                      # Entry point
├── docs/
│   └── mdgspace_dataset.json    # Dataset for MDG Space QA
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data/                    # Data handling
│   │   ├── __init__.py          # Package initialization
│   │   └── loader.py            # QA data loading
│   ├── embeddings/              # Text embedding
│   │   ├── __init__.py          # Package initialization
│   │   └── models.py            # Embedding models
│   ├── retrieval/               # Vector search
│   │   ├── __init__.py          # Package initialization
│   │   ├── chroma_service.py    # ChromaDB operations
│   │   └── faiss_service.py     # FAISS operations
│   ├── generation/              # Text generation
│   │   ├── __init__.py          # Package initialization
│   │   └── llm.py               # LLM interaction
│   └── utils/                   # Utilities
│       ├── __init__.py          # Package initialization
│       └── helpers.py           # Helper functions
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## Requirements

- sentence-transformers>=2.2.0
- langchain>=0.1.0
- langchain-chroma>=0.0.1
- faiss-cpu>=1.7.4
- chromadb>=0.4.15
- requests>=2.31.0
- numpy>=1.24.0

## Advanced Configuration

The system can be configured by modifying parameters in `config.py`:

- Change embedding models
- Adjust relevance thresholds
- Configure LLM parameters
- Update file paths
