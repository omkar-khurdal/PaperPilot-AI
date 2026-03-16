import os

from app.chunking import chunk_text
from app.embeddings import create_embeddings
from app.generator import REFUSAL_TEXT, generate_answer
from app.ingestion import extract_text_from_pdf
from app.memory import add_ai_message, add_user_message
from app.query_rewriter import rewrite_query
from app.vector_store import (create_vector_store, get_confidence_label,
                              search, should_answer)


def load_documents(folder_path="data"):
    print("Loading documents...")

    pages = []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            print(f"Reading {file}...")

            extracted_pages = extract_text_from_pdf(pdf_path)

            for page in extracted_pages:
                page["source"] = file

            pages.extend(extracted_pages)

    return pages


def build_pipeline():
    pages = load_documents("data")

    print("Chunking documents...")
    chunks = chunk_text(pages)

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)

    print("Building vector database...")
    index = create_vector_store(embeddings)

    return chunks, index


def main():
    chunks, index = build_pipeline()

    print("System ready! Ask questions about the documents.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        rewritten_query = rewrite_query(question)
        results = search(rewritten_query, chunks, index, top_k=7)
        confidence = get_confidence_label(results)

        if confidence == "Low" or not should_answer(results):
            answer = REFUSAL_TEXT
            confidence = "Low"
        else:
            answer = generate_answer(rewritten_query, results)

        # update memory after retrieval
        add_user_message(question)
        add_ai_message(answer)

        print("\nRewritten query:")
        print(rewritten_query)

        print("\nConfidence:")
        print(confidence)

        print("\nAnswer:\n")
        print(answer)

        print("\nSources:")
        for item in results:
            print(
                f"- {item['source']} | page {item['page']} | "
                f"final={item['score']:.4f} dense={item['dense_score']:.4f} sparse={item['sparse_score']:.4f}"
            )

        print("\n")


if __name__ == "__main__":
    main()