from typing import List, Dict

def chunk_text(
        text: str,
        source: str = "user_input",
        title: str = "Uploaded Text",
        chunk_size: int = 1000,
        overlap: int = 100
) -> List[Dict]:
    
    chunks = []
    start = 0
    chunk_index = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        content = text[start:end]

        chunks.append({
            "content": content,
            "metadata": {
                "content": content,
                "source": source,
                "title": title,
                "chunk_index": chunk_index,
                "position": start
            }
        })

        chunk_index += 1

        if end == text_length:
            break

        start = end - overlap

    return chunks
