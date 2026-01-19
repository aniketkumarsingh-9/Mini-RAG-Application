import streamlit as st
import logging

from app.rag.chunker import chunk_text
from app.rag.embedder import embed_texts
from app.rag.vector_store import (
    clear_collection,
    upsert_embeddings,
    search_embeddings,
    get_collection_stats,
)
from app.rag.llm import generate_answer

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
    if not doc_text.strip():
        st.warning("Please enter some text.")
    else:
        clear_collection()

        chunks = chunk_text(doc_text)
        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        embeddings = embed_texts(texts)
        upsert_embeddings(embeddings, texts, metadatas)

        st.success(f"Document ingested successfully! Chunks: {len(chunks)}")

# -----------------------------
# Ask Question
# -----------------------------
st.header("2. Ask a Question")

query = st.text_input("Enter your question:")
top_k = st.slider("Top k Results", 1, 5, 3)

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        query_embedding = embed_texts([query])[0]
        search_results = search_embeddings(query_embedding, top_k)

        if not search_results:
            st.subheader("Answer")
            st.write("No relevant information found.")
        else:
            contexts = [r["content"] for r in search_results if r["content"]]
            answer = generate_answer(query, contexts)

            confidence = sum(r["score"] for r in search_results) / len(search_results)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Confidence")
            st.write(round(confidence, 3))

            st.subheader("Sources")
            for i, r in enumerate(search_results, 1):
                st.markdown(f"**Source {i}:**")
                st.write(r["content"])
                st.caption(f"Score: {round(r['score'], 3)}")
