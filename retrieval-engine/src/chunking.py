def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0:
        raise ValueError("Overlap must be nonnegative integer.")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk_size.")

    chunks = []
    words = text.split()

    for i in range(0, len(words), chunk_size - overlap):
        if i + chunk_size < len(words):
            chunks.append(" ".join(words[i:i + chunk_size]))
        else:
            chunks.append(" ".join(words[i:]))
            break

    return chunks
