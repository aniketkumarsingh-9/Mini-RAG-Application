import logging

logging.basicConfig(
    level= logging.INFO,
    format= "%(asctime)s - %(levelname)s - %(message)s"
)


from fastapi import FastAPI
from pydantic import BaseModel
from app.rag.chunker import chunk_text
from app.rag.embedder import embed_texts
from app.rag.vector_store import clear_collection, upsert_embeddings
from app.rag.vector_store import search_embeddings
from app.rag.llm import generate_answer
from app.rag.vector_store import get_collection_stats

app = FastAPI(title="Mini RAG App")

class DocumentInput(BaseModel):
    text: str

class QueryInput(BaseModel):
    query: str
    top_k: int = 5

@app.get("/stats")
def stats():
    return get_collection_stats()

@app.get("/health")
def health_check():
    return {"status" : "ok"}


@app.post("/ingest")
def ingest_document(doc: DocumentInput):
    clear_collection()
    logging.info("Ingest request received")

    chunks = chunk_text(doc.text)

    logging.info(f"Chunks created: {len(chunks)}")

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    embeddings = embed_texts(texts)
   
    upsert_embeddings(embeddings, texts, metadatas)
    
    return {
        "chunk_ingested": len(chunks)
    }

@app.post("/query")
def query_documents(q : QueryInput):
    logging.info(f"Query received: {q.query}")

    query_embedding = embed_texts([q.query])[0]

    search_results = search_embeddings(query_vector= query_embedding, top_k= q.top_k)

    if not search_results:
        return {
            "query" : q.query,
            "answer": "No relevant information in the database.",
            "confidence": 0.0,
            "sources": []
        }
    

    contexts = list([r["content"] for r in search_results if r["content"]])

    answer = generate_answer(q.query, contexts)

    confidence = (
        sum(r["score"] for r in search_results) / len(search_results)
        if search_results else 0.0
    )

    return {
        "query" : q.query,
        "answer": answer,
        "confidence": round(confidence, 3),
        "sources": [
            {
                "content": r["content"],
                "score": round(r["score"], 3)
            }
            for r in search_results
        ]
    } 

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
