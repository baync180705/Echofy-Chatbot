"""
Main entry point for the RAG system.
"""

# Server setup libraries
from fastapi import FastAPI as server, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration and utility libraries
from src.embeddings.models import StellaEmbeddings
from src.generation.llm import generate_response
from src.services.rag_pipeline import create_prompt, initialize_database, retrieve_contexts
from src.utils.helpers import parse_arguments

app = server()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Query(BaseModel):
    prompt: str

@app.post("/query")
def handleQuery(query: Query):
    """
    Endpoint to handle query requests.
    
    Args:
        query (Query): The query object containing the prompt.
        
    Returns:
        dict: The response containing the response and sources.
    """
    try:
        response, contexts = orchestratePipeline(query_text=query.prompt)        
        
        return {"response": response, "sources": contexts, "status": 200}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def orchestratePipeline(query_text):
    """Main function orchestrating the retrieval and generation process."""

    # Initialize embedding function
    embedding_function = StellaEmbeddings()

    print(f"The query text is: {query_text}")
    
    # Initialize database if needed
    initialize_database(embedding_function)
    
    # Retrieve relevant contexts
    contexts = retrieve_contexts(query_text, embedding_function)
    
    if not contexts:
        print(f"\nUnable to find matching results for '{query_text}'")
        return
    
    print(f"Found {len(contexts)} relevant results. Generating response...")
    
    # Create prompt and generate response
    prompt = create_prompt(query_text, contexts)
    response = generate_response(prompt)
    
    print("\nResponse:")
    print(response)

    return response, contexts


if __name__ == "__main__":
    # Parse query
    query_text = parse_arguments()
    print("Received your prompt! Loading the database and finding matching results...")
    
    # Run main function
    orchestratePipeline(query_text=query_text)