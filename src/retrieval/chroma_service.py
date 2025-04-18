"""
ChromaDB operations for vector storage and retrieval.
"""

from langchain_chroma import Chroma


def load_chroma(persist_directory, embedding_function=None):
    """
    Load a ChromaDB collection from disk.
    
    Args:
        persist_directory (str): Directory where ChromaDB is stored
        embedding_function: Function to convert text to embeddings
        
    Returns:
        Chroma: ChromaDB collection
    """
    try:
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
    except Exception as e:
        print(f"Error loading ChromaDB: {str(e)}")
        raise


def save_to_chroma(persist_directory, chunks, embedding_function=None):
    """
    Save documents to a ChromaDB collection.
    
    Args:
        persist_directory (str): Directory to store ChromaDB
        chunks (list): List of Document objects to store
        embedding_function: Function to convert text to embeddings
    """
    try:
        # Create a new ChromaDB collection
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=persist_directory
        )

    except Exception as e:
        # Handle Exceptions
        print(f"Error saving to ChromaDB: {str(e)}")
        raise