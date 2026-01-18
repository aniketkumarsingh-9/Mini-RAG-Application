from fastapi import FastAPI
from pydantic import BaseModel
from app.rag.chunker import chunk_text
from app.rag.embedder import embed_texts
from app.rag.vector_store import upsert_embeddings
from app.rag.vector_store import search_embeddings
from dotenv import load_dotenv


load_dotenv()


app = FastAPI(title="Mini RAG App")

class DocumentInput(BaseModel):
    text: str

class QueryInput(BaseModel):
    query: str
    top_k: int = 5


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
    
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"]  for chunk in chunks]

    embeddings = embed_texts(texts)

    upsert_embeddings(embeddings, metadatas)
    
    return {
        "message": "Document Ingested, embedded, and stored successfully",
        "total_chunks": len(chunks)
    }

@app.post("/query")
def query_documents(q : QueryInput):
    query_embedding = embed_texts([q.query])[0]

    results = search_embeddings(
        query_vector=query_embedding,
        top_k=q.top_k
    )

    return {
        "query" : q.query,
        "results": results
    }