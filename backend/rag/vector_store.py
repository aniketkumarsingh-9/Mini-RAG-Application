import os
import uuid
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "documents"

client = QdrantClient(
    url = QDRANT_URL,
    api_key = QDRANT_API_KEY
)

def upsert_embeddings(
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[dict]
):
    points = []

    for vector, text, metadata in zip(embeddings, texts, metadatas):
        payload = {
            "content": text,
            **metadata
        }
        points.append(
            PointStruct(
                id = str(uuid.uuid4()),
                vector = vector,
                payload = payload
            )
        )

    client.upsert(
        collection_name = COLLECTION_NAME,
        points = points
    )

def search_embeddings(
        query_vector: List[float],
        top_k: int = 5
):
    
    response = client.query_points(
        collection_name = COLLECTION_NAME,
        query = query_vector,
        limit = top_k,
        with_payload = True
    )

    hits = response.points if hasattr(response, "points") else response

    results = []
    seen = set()

    for hit in hits:
        content = hit.payload.get("content")

        if content and content not in seen:
            seen.add(content)
            results.append({
                "content": content,
                "metadata": hit.payload,
                "score": hit.score
        })
    
    return results

def get_collection_stats():
    info = client.get_collection(COLLECTION_NAME)

    return {
        "collection": COLLECTION_NAME,
        "total_vectors": info.points_count,
        "vector_size": info.config.params.vectors.size,
        "status": "healthy"
    }

def clear_collection():
    return