import re
from collections import Counter

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# CPU-friendly embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def create_vector_store(embeddings):
    """
    Uses cosine similarity through normalized vectors + IndexFlatIP.
    """
    normalized_embeddings = normalize_embeddings(embeddings.astype("float32"))

    dimension = normalized_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(normalized_embeddings)

    return index


def tokenize(text: str):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def keyword_overlap_score(query: str, text: str) -> float:
    """
    Lightweight sparse retrieval signal.
    Very cheap alternative to BM25 for your CPU setup.
    """
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)

    if not query_tokens or not text_tokens:
        return 0.0

    query_counter = Counter(query_tokens)
    text_counter = Counter(text_tokens)

    overlap = 0
    for token, q_count in query_counter.items():
        overlap += min(q_count, text_counter.get(token, 0))

    return overlap / max(len(query_tokens), 1)

def token_set(text: str):
    return set(tokenize(text))


def rerank_score(query: str, chunk_text: str, base_score: float) -> float:
    """
    Cheap CPU-friendly reranker.
    Boosts chunks that share more direct lexical evidence with the query.
    """
    q_tokens = token_set(query)
    c_tokens = token_set(chunk_text)

    if not q_tokens or not c_tokens:
        return base_score

    overlap_count = len(q_tokens & c_tokens)
    overlap_ratio = overlap_count / max(len(q_tokens), 1)

    # extra boost for chunks that contain most query words
    direct_match_boost = 0.15 * overlap_ratio

    return base_score + direct_match_boost


def dense_search(question, chunks, index, dense_k=20):
    question = question.strip().lower()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    question_embedding = normalize_embeddings(question_embedding)

    distances, indices = index.search(question_embedding, dense_k)

    dense_results = {}

    for idx, score in zip(indices[0], distances[0]):
        if idx == -1:
            continue

        dense_results[idx] = float(score)

    return dense_results

def rerank_results(query: str, results: list, final_top_k: int = 7):
    """
    Cross-encoder reranking.
    Re-scores retrieved chunks using query-chunk relevance.
    """
    if not results:
        return []

    pairs = [(query, item["text"]) for item in results]
    rerank_scores = reranker.predict(pairs)

    reranked = []
    for item, r_score in zip(results, rerank_scores):
        new_item = item.copy()
        new_item["rerank_score"] = float(r_score)
        reranked.append(new_item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:final_top_k]



def search(question, chunks, index, top_k=7):
    """
    Hybrid retrieval:
    - dense semantic similarity
    - sparse keyword overlap
    - score fusion
    """
    if not chunks:
        return []

    dense_k = min(max(top_k * 4, 20), len(chunks))
    dense_scores = dense_search(question, chunks, index, dense_k=dense_k)

    candidate_indices = set(dense_scores.keys())

    # Add sparse candidates from all chunks.
    # With your dataset size, this is still fine on CPU.
    sparse_scores = {}
    for i, chunk in enumerate(chunks):
        sparse_score = keyword_overlap_score(question, chunk["text"])
        if sparse_score > 0:
            sparse_scores[i] = sparse_score
            candidate_indices.add(i)

    results = []

    for idx in candidate_indices:
        chunk = chunks[idx]

        dense_score = dense_scores.get(idx, 0.0)
        sparse_score = sparse_scores.get(idx, 0.0)

        # weighted fusion
        final_score = (0.75 * dense_score) + (0.25 * sparse_score)

        results.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "page": chunk["page"],
            "source": chunk["source"],
            "score": float(final_score),
            "dense_score": float(dense_score),
            "sparse_score": float(sparse_score)
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate near-identical source/page/text pairs
    deduped_results = []
    seen = set()

    for item in results:
        key = (item["source"], item["page"], item["text"][:120])
        if key in seen:
            continue
        seen.add(key)
        deduped_results.append(item)

    # take larger candidate pool first
    rerank_pool = deduped_results[:20]

    # rerank down to final top_k
    final_results = rerank_results(question, rerank_pool, final_top_k=top_k)

    return final_results


def should_answer(results):
    """
    Answer gating.
    Slightly relaxed to reduce false refusals when evidence is clearly present.
    """
    if not results:
        return False

    best_score = results[0]["score"]
    support_count = sum(1 for r in results[:5] if r["score"] >= 0.38)

    # strong support from best chunk
    if best_score >= 0.55:
        return True

    # moderate support backed by multiple chunks
    if best_score >= 0.42 and support_count >= 2:
        return True
    

    return False


def get_confidence_label(results):
    if not results:
        return "Low"

    best_score = results[0]["score"]

    if best_score >= 0.72:
        return "High"
    elif best_score >= 0.52:
        return "Medium"
    else:
        return "Low"