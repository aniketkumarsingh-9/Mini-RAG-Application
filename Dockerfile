FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from transformers import pipeline; pipeline('text2text-generation', model='google/flan-t5-base')"

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY .env .env

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0"]