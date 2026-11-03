import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import (
    process_youtube_videos,
    generate_embedding,
    generate_answers,
    get_persist_directory,
)

app = FastAPI(
    title="YouTube RAG API",
    version="1.0"
)

# Request schemas
class IngestRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    query: str
    persist_directory: str

@app.get("/")
def home():
    return {"status": "Online", "message": "YouTube RAG FastAPI Server is running!"}

@app.post("/ingest")
def ingest_youtube_source(request: IngestRequest):
    """
    Takes a YouTube video or playlist URL, processes transcripts,
    creates a unique local FAISS vector database, and returns the directory path.
    """
    try:
        if not request.url.strip():
            raise HTTPException(status_code=400, detail="URL cannot be empty.")
        
        # 1. Load transcripts using your existing pipeline function
        docs = process_youtube_videos(request.url)
        if not docs:
            raise HTTPException(status_code=400, detail="No transcripts found for this URL or playlist.")
        
        # 2. Create a unique collection folder
        collection_id = uuid.uuid4().hex
        persist_dir = get_persist_directory(collection_id)
        
        # 3. Generate embeddings and save to local FAISS index
        generate_embedding(docs, persist_directory=persist_dir)
        
        return {
            "status": "success",
            "message": f"Successfully ingested {len(docs)} documents.",
            "persist_directory": persist_dir
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Takes a user question and a persist_directory, then queries the vector database using Gemini.
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
        answer = generate_answers(request.query, persist_directory=request.persist_directory)
        
        return {
            "status": "success",
            "query": request.query,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))