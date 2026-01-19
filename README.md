---
title: Mini RAG Application
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---




# Mini RAG Application

This project is a Mini Retrieval-Augmented Generation (RAG) application that combines
information retrieval with large language models to generate grounded and context-aware answers.

The system allows users to ingest custom documents and ask questions based on the ingested content.

## Features

- Document ingestion and chunking
- Vector embedding using Sentence Transformers
- Semantic search using Qdrant vector database
- Answer generation using FLAN-T5 model
- Confidence score based on similarity
- Simple Streamlit frontend

## Tech Stack

- Backend: FastAPI
- Frontend: Streamlit
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector Database: Qdrant Cloud
- LLM: google/flan-t5-base
- Language: Python


## Project Structure

mini-rag-app/
│
├── app/
│   ├── main.py
│   └── rag/
│       ├── chunker.py
│       ├── embedder.py
│       ├── vector_store.py
│       └── llm.py
│
├── ui/
│   └── app.py
│
├── requirements.txt
└── README.md

## How to Run Locally

### 1. Clone Repository
git clone https://github.com/aniketkumarsingh-9/Mini-RAG-Application.git
cd Mini-RAG-Application

### 2. Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Start Backend
uvicorn app.main:app --reload

### 5. Start Frontend
streamlit run ui/app.py


## Usage Flow

1. Ingest a document using the frontend
2. Ask a question related to the document
3. System retrieves relevant chunks
4. LLM generates an answer using retrieved context

## Resume Link

https://drive.google.com/file/d/1jdlaHWddv2_J0N3BSR7H9jy9DDFPUKZt/view?usp=drive_link

