import numpy as np
from sentence_transformers import SentenceTransformer

# load same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def search(query, index, chunks, top_k=3):

    # Convert query to embedding
    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results