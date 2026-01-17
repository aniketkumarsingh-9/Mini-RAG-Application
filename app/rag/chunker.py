from typing import List, Dict

def chunk_text(
        text: str,
        chunk_size: int = 1000,
        overlap: int = 100
) -> List[Dict]:
    
    chunks = []
    start = 0
    chunk_index = 0
    text_length = len(text)

    while start < len(text):
        end = min(start + chunk_size, text_length)
        chunk_content = text[start:end]

        chunks.append({
            "chunk_index": chunk_index,
            "content": chunk_content
        })

        chunk_index += 1

        if end == text_length:
            break

        start = end - overlap

    return chunks