import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title= "Mini RAG App", layout= "centered")
st.title("Mini RAG Application")

st.header("1. Ingest Document")

doc_text = st.text_area(
    "Write Your document text here:", 
    height= 200
    
)

if st.button("Ingest Document"):
    if not doc_text.strip():
        st.warning("Please enter some text.")
    else:
        response = requests.post(
            f"{BACKEND_URL}/ingest",
            json={"text": doc_text}
        )

        if response.status_code == 200:
            st.success("Document ingested successfully!")
            st.json(response.json())
        else:
            st.error("Failed to ingest document.")


st.header("2. Ask a Question")

query = st.text_input("Enter your question:")
top_k = st.slider("Top k Results", 1, 5, 3)

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json= {"query": query, "top_k": top_k}
        )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Answer")
            st.write(data.get("answer"))

            sources = data.get("sources", [])
            if sources:
                confidence = max(src["score"] for src in sources)
                st.subheader("Confidence")
                st.write(round(confidence, 3))

            st.subheader("Sources")
            for i, src in enumerate(data.get("sources", []), start=1):
                st.markdown(f"**Source {i}:**")
                st.write(src["content"])
                st.caption(f"Score: {src['score']}")
        
        else:
            st.error("Failed to get answer.")
    