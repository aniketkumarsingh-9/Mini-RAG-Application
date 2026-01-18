from typing import List
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: List[str]) -> List[List[float]]:
    embeddings = model.encode(texts, convert_to_numpy = True)
    return embeddings.tolist()


if __name__ == "__main__":
    vectors = embed_texts(["Hello world"])
    print(len(vectors[0]))