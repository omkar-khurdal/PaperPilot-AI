# from app.cleaning import clean_text


# def chunk_text(pages, chunk_size=500, overlap=100):

#     chunks = []
#     chunk_id = 1

#     for page in pages:

#         cleaned = clean_text(page["text"])

#         for i in range(0, len(cleaned), chunk_size - overlap):

#             chunk = cleaned[i:i + chunk_size]

#             # Ignore very small chunks
#             if len(chunk.strip()) < 50:
#                 continue

#             chunks.append({
#                 "chunk_id": chunk_id,
#                 "source": page["source"],
#                 "page": page["page"],
#                 "text": chunk
#             })

#             chunk_id += 1

#     return chunks


import re

from app.cleaning import clean_text


def split_into_sentences(text: str):
    """
    Simple sentence splitter.
    Good enough for local CPU RAG without extra heavy NLP dependencies.
    """
    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def split_long_sentence(sentence: str, max_len: int):
    """
    If a single sentence is too long, split it into smaller parts.
    """
    if len(sentence) <= max_len:
        return [sentence]

    parts = []
    start = 0

    while start < len(sentence):
        end = start + max_len
        parts.append(sentence[start:end].strip())
        start = end

    return [p for p in parts if p]


def chunk_text(pages, chunk_size=700, overlap=120, min_chunk_length=80):
    """
    Sentence-aware chunking.

    Why this is better than raw fixed slicing:
    - preserves semantic boundaries better
    - reduces broken sentences across chunks
    - improves retrieval quality

    Input page format:
    {
        "source": "file.pdf",
        "page": 3,
        "text": "..."
    }
    """
    chunks = []
    chunk_id = 1

    for page in pages:
        source = page["source"]
        page_num = page["page"]
        page_text = page["text"]

        sentences = split_into_sentences(page_text)

        # further split very long sentences
        processed_sentences = []
        for sent in sentences:
            processed_sentences.extend(split_long_sentence(sent, max_len=300))

        if not processed_sentences:
            continue

        current_sentences = []
        current_length = 0

        for sentence in processed_sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len + 1 <= chunk_size:
                current_sentences.append(sentence)
                current_length += sentence_len + 1
            else:
                chunk_text_value = " ".join(current_sentences).strip()

                if len(chunk_text_value) >= min_chunk_length:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "source": source,
                        "page": page_num,
                        "text": chunk_text_value
                    })
                    chunk_id += 1

                # overlap by characters from previous chunk tail
                if overlap > 0 and chunk_text_value:
                    overlap_text = chunk_text_value[-overlap:].strip()
                    current_sentences = [overlap_text, sentence] if overlap_text else [sentence]
                    current_length = sum(len(x) + 1 for x in current_sentences)
                else:
                    current_sentences = [sentence]
                    current_length = sentence_len + 1

        # flush final chunk
        final_chunk = " ".join(current_sentences).strip()
        if len(final_chunk) >= min_chunk_length:
            chunks.append({
                "chunk_id": chunk_id,
                "source": source,
                "page": page_num,
                "text": final_chunk
            })
            chunk_id += 1

    return chunks