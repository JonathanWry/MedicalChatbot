import base64
import os
from fastapi import HTTPException

import torch
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Literal, Optional
from BMRetriver import load_bmretriever_model, load_pubmed_documents, retrieve_relevant_documents, encode_documents
from chatgpt_requests import describeThroughDirectText, describeThroughKG, reset_openai_client, validate_openai_key
from config import OPENAI_API_KEY
from utils import encode_pdf_to_base64, clean_title



# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly define allowed methods
    allow_headers=["*"],  # Allow all headers
)
# Load retrieval model and documents at startup
retrieveModel, tokenizer = load_bmretriever_model()
documents = load_pubmed_documents(num_docs=10000)  # Load PubMed documents
# documents_embedding=encode_documents(documents, tokenizer, retrieveModel)
documents_embedding=torch.load("./embeddings/doc_embeddings.pt", map_location="cpu")



class QueryRequest(BaseModel):
    query: str
    mode: Literal["kg1","kg2", "direct"] = "direct"
    model: Optional[str] = "gpt-4o-mini"
    temperature: Optional[float] = 0.4
    top_p: Optional[float] = 0.3
    top_k: Optional[int] = 3
class APIKeyRequest(BaseModel):
    api_key: str

@app.get("/")
def health_check():
    return {"status": "Server is running"}


@app.post("/set_api_key")
def set_api_key(request: APIKeyRequest):
    if not request.api_key:
        raise HTTPException(status_code=400, detail="API key is required")

    # Validate API key before storing it
    if not validate_openai_key(request.api_key):
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key")

    try:
        reset_openai_client(request.api_key)
        return {"message": "API key updated successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set API key: {str(e)}")
@app.post("/query")
def process_query(request: QueryRequest):
    """
    Process user query using the retriever and return a response.
    - mode = "kg" -> Generates response using Knowledge Graph (`describeThroughKG`)
    - mode = "direct" -> Generates direct summary (`describeThroughDirectText`)
    """
    retrieved_texts = retrieve_relevant_documents(
        query=request.query,
        model=retrieveModel,
        tokenizer=tokenizer,
        doc_embeddings=documents_embedding,
        documents=documents,
        top_k=request.top_k
    )

    if not retrieved_texts:
        return {"error": "No relevant documents found"}
        # Extract titles from retrieved documents
    sources = []
    for doc_text in retrieved_texts:
        lines = doc_text.split("\n")
        for line in lines:
            if line.startswith("Title:"):
                found_title = line[len("Title:"):].strip()
                cleaned_title = clean_title(found_title)
                if cleaned_title:  # Only add if it's non-empty after cleaning
                    sources.append(cleaned_title)
                break  # Stop after first found title

    pdf_data = None
    if request.mode == "kg1":
        response, pdf_files = describeThroughKG(
            query=request.query,
            retrieved_texts=retrieved_texts,
            mode="ver1",  # or "ver2" if preferred
            ai_model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        if os.path.exists(pdf_files[0]):
            pdf_data = encode_pdf_to_base64(pdf_files[0])
    elif request.mode == "kg2":
        response, pdf_files = describeThroughKG(
            query=request.query,
            retrieved_texts=retrieved_texts,
            mode="ver2",  # or "ver2" if preferred
            ai_model=request.model,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        pdf_data = [encode_pdf_to_base64(pdf) for pdf in pdf_files if os.path.exists(pdf)]  # Encode multiple PDFs
    else:
        response = describeThroughDirectText(
            query=request.query,
            retrieved_texts=retrieved_texts,
            ai_model=request.model,
            top_k=request.top_k,
            temperature=request.temperature,
            top_p=request.top_p
        )

    return {
        "query": request.query,
        "sources": sources,
        "response": response,
        "pdfs": pdf_data
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
