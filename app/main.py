from fastapi import FastAPI
from pydantic import BaseModel
from app.rag.chunker import chunk_text

app = FastAPI(title="Mini RAG App")

class DocumentInput(BaseModel):
    text: str


@app.get("/health")
def health_check():
    return {"status" : "ok"}


@app.post("/ingest")
def ingest_document(doc: DocumentInput):
    chunks = chunk_text(
        text = doc.text,
        source = "user_input",
        title = "Uploaded Text"
        )
    
    return {
        "message": "Document Ingested and Chunked Successfully",
        "total_chunks": len(chunks),
        "chunks_preview": chunks[:2]
    }