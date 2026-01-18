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
        metadatas: List[dict]
):
    points = []

    for vector, metadata in zip(embeddings, metadatas):
        points.append(
            PointStruct(
                id = str(uuid.uuid4()),
                vector = vector,
                payload = metadata
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
    for hit in hits:
        results.append({
            "content": hit.payload.get("content"),
            "metadata": hit.payload,
            "score": hit.score
        })
    
    return results