from typing import List
from transformers import pipeline

qa_pipeline = pipeline(
    "text2text-generation",
    model = "google/flan-t5-base",
    max_new_tokens = 128
    )

def generate_answer(query: str, contexts: List[str]) -> str:
    if not contexts:
        return "No relevant context found in the database."

    context_text = "\n".join(dict.fromkeys(contexts))

    prompt = (
        "You are a question answering system.\n"
        "Answer the QUESTION using the CONTEXT.\n"
        "Do NOT repeat the context.\n"
        "Answer in a full sentence.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{query}\n\n"
        "ANSWER:"
    )


    result = qa_pipeline(prompt)[0]["generated_text"].strip()

    if len(result.split()) < 6:
        return (
            "The context retrieved is insufficient to answer this question clearly."
        )
    
    return result

