import streamlit as st
import logging

import streamlit as st
from app.rag.chunker import chunk_text
from app.rag.embedder import embed_texts
from app.rag.vector_store import clear_collection, upsert_embeddings, search_embeddings
from app.rag.llm import generate_answer

import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Mini RAG App", layout="centered")
st.title("Mini RAG Application")

# -----------------------------
# Ingest Document
# -----------------------------
st.header("1. Ingest Document")

doc_text = st.text_area(
    "Write your document text here:",
    height=200
)

if st.button("Ingest Document"):
    clear_collection()
    chunks = chunk_text(doc_text)
    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = embed_texts(texts)
    upsert_embeddings(embeddings, texts, metadatas)
    st.success(f"Ingested {len(chunks)} chunks")


# -----------------------------
# Ask Question
# -----------------------------
st.header("2. Ask a Question")

query = st.text_input("Enter your question:")
top_k = st.slider("Top k Results", 1, 5, 3)

if st.button("Get Answer"):
    query_embedding = embed_texts([query])[0]
    results = search_embeddings(query_embedding, top_k)

    if not results:
        st.warning("No relevant info found")
    else:
        contexts = [r["content"] for r in results]
        answer = generate_answer(query, contexts)

        st.subheader("Answer")
        st.write(answer)

        confidence = sum(r["score"] for r in results) / len(results)
        st.subheader("Confidence")
        st.write(round(confidence, 3))
