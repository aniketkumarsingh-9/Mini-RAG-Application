from qdrant_client.models import VectorParams, Distance
from vector_store import client


COLLECTION_NAME = "documents"

def create__collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size = 384,
            distance=Distance.COSINE
        )
    )
    print("Collection created successfully")

if __name__ == "__main__":
    create__collection()