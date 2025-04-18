"""
Data loading functionality.
"""

import json
from langchain_core.documents import Document
from config import JSON_PATH


def load_qa_pairs():
    """
    Load question-answer pairs from a JSON file and convert them to Document objects.
    
    Returns:
        list: A list of Document objects with question-answer content
    """
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as file:
            qa_pairs = json.load(file)
        
        documents = []
        for pair in qa_pairs:
            content = f"Question: {pair['question']}\nAnswer: {pair['answer']}"
            doc = Document(page_content=content, metadata={"source": "qa_dataset"})
            documents.append(doc)
        
        print(f"Loaded {len(documents)} QA pairs from {JSON_PATH}")
        return documents
    except FileNotFoundError:
        print(f"Error: File not found at {JSON_PATH}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {JSON_PATH}")
        return []
    except Exception as e:
        print(f"Error loading QA pairs: {str(e)}")
        return []
