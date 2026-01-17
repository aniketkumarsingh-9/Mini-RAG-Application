from fastapi import FastAPI

app = FastAPI(title="Mini RAG App")

@app.get("/health")
def health_check():
    return {"status" : "ok"}