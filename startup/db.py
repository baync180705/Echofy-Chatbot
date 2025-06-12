"""
This script initializes the database required for the RAG (Retrieval-Augmented Generation) pipeline. This file is to executed in the startup phase.

It uses the `StellaEmbeddings` model to generate embeddings and calls the 
`initialize_database` function to set up the database with the required data.
"""

from src.embeddings.models import StellaEmbeddings
from src.services.rag_pipeline import initialize_database

# Create an instance of the StellaEmbeddings model
embedding_function = StellaEmbeddings()

# Initialize the database with the embedding function
initialize_database(embedding_function)
