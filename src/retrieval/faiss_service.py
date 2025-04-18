"""
FAISS operations for efficient similarity search.
"""

import os
import numpy as np
import faiss


def save_to_faiss(faiss_index_path, chunks, embedding_function):
    """
    Create and save a FAISS index for document embeddings.
    
    Args:
        faiss_index_path (str): Path to save the FAISS index
        chunks (list): List of Document objects to index
        embedding_function: Function to convert text to embeddings
    """
    try:
        # Extract texts from documents
        texts = [doc.page_content for doc in chunks]
        
        # Generate embeddings
        embeddings = embedding_function.embed_documents(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        d = embeddings_array.shape[1]  # embedding dimension
        index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity (normalized vectors)
        index.add(embeddings_array)
        
        # Save index to disk
        faiss.write_index(index, faiss_index_path)
        print(f"Saved FAISS index with {len(chunks)} vectors to {faiss_index_path}")
    except Exception as e:
        print(f"Error saving to FAISS: {str(e)}")
        raise


def search_faiss(faiss_index_path, query_text, embedding_function, k=3):
    """
    Search a FAISS index for similar documents.
    
    Args:
        faiss_index_path (str): Path to the FAISS index
        query_text (str): Query text to search for
        embedding_function: Function to convert text to embeddings
        k (int): Number of results to return
        
    Returns:
        list: List of (index, score) tuples for the closest documents
    """
    if not os.path.exists(faiss_index_path):
        print(f"FAISS index not found at {faiss_index_path}")
        return []
    
    try:
        # Load FAISS index
        index = faiss.read_index(faiss_index_path)
        
        # Generate query embedding
        query_embedding = embedding_function.embed_query(query_text)
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search for similar documents
        scores, indices = index.search(query_array, k)
        
        # Convert to list of tuples
        results = [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]
        
        return results
    except Exception as e:
        print(f"Error searching FAISS: {str(e)}")
        return []