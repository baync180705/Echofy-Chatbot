import json
import os

from config import CHROMA_PATH, DEFAULT_RELEVANCE_THRESHOLD, DEFAULT_RETRIEVAL_K, FAISS_INDEX_PATH
from src.data.loader import load_qa_pairs
from src.retrieval.chroma_service import load_chroma, save_to_chroma
from src.retrieval.faiss_service import save_to_faiss, search_faiss


def initialize_database(embedding_function):
    """Initialize the vector database if it doesn't exist."""
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
        documents = load_qa_pairs()
        save_to_chroma(persist_directory=CHROMA_PATH, chunks=documents, embedding_function=embedding_function)
        save_to_faiss(faiss_index_path=FAISS_INDEX_PATH, chunks=documents, embedding_function=embedding_function)


def retrieve_contexts(query_text, embedding_function, k=DEFAULT_RETRIEVAL_K, threshold=DEFAULT_RELEVANCE_THRESHOLD):
    """Retrieve and filter relevant contexts for the query."""
    # Load ChromaDB
    db = load_chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Search FAISS index
    faiss_results = search_faiss(
        faiss_index_path=FAISS_INDEX_PATH, 
        query_text=query_text, 
        embedding_function=embedding_function, 
        k=k
    )
    
    if not faiss_results:
        return []
    
    # Unpack results
    indices, scores = zip(*faiss_results)
    
    # Filter by threshold
    filtered_indices = []
    filtered_scores = []
    for i, score in enumerate(scores):
        if score >= threshold:
            filtered_indices.append(indices[i])
            filtered_scores.append(score)
        else:
            print(f"Filtered out result with score {score} (below threshold {threshold})")
    
    if not filtered_indices:
        return []
    
    # Get document metadata and content
    docs_metadata = db.get(limit=len(indices), offset=min(indices))
    doc_ids = docs_metadata.get("ids", [])
    results = db.get_by_ids(doc_ids)
    
    # Format results
    contexts = []
    for i, result in enumerate(results):
        if i < len(filtered_scores):
            contexts.append({
                "context": result.page_content,
                "relevance_score": str(filtered_scores[i])
            })
    
    return contexts


def create_prompt(query_text, contexts):
    """Create the prompt with retrieved contexts."""
    prompt = f"""
        You are an intelligent language model designed to answer questions using only a list of retrieved text passages, each tagged with a relevance score.

        Your objective is to:
        - **Carefully read** the provided contexts and select only the most relevant one(s) to answer the question.
        - **Do not rely on prior knowledge** or external information. Your answer must be grounded strictly in the given contexts.

        Guidelines:
        - The **higher the score**, the more importance that context holds in shaping your answer.
        - **Do not use prior knowledge** or any information outside the provided contexts.
        - **Respond in 2–3 short, clear sentences** forming a brief paragraph.
        - Your answer should be plain text—**never include the contexts or scores** in your response.
        - If the question is unrelated to MDG Space, respond with: "I can only provide information related to MDG Space. Please ask a question within this scope."
        - If the question is related to MDG Space, the answer **must directly address it using the provided context**—no exceptions.
        - The response **must always be confident** and **must never suggest uncertainty** or imply that it is "trying" to answer the question.
        - **Hallucination is strictly prohibited**—only use the information available in the retrieved context.

        Here is the list of contexts with relevance scores:

        {json.dumps(contexts, indent=4)}

        Now, answer the following question based only on the valid context(s):

        Question: {query_text}
    """
    return prompt
