from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample sentences
sentences = [
    "AI is transforming healthcare.",
    "Machine learning improves medical diagnosis."
]

# Generate embeddings
embeddings = model.encode(sentences)

print("Number of vectors:", len(embeddings))
print("Dimension of each vector:", len(embeddings[0]))