import ollama

REFUSAL_TEXT = "I could not find the answer in the indexed document context."


def build_context(context_chunks):
    blocks = []

    for i, chunk in enumerate(context_chunks, start=1):
        block = (
            f"[Source {i}] {chunk['source']} | Page: {chunk['page']}\n"
            f"{chunk['text']}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def generate_answer(question, context_chunks):
    if not context_chunks:
        return REFUSAL_TEXT

    context = build_context(context_chunks)

    system_prompt = """
You are a strictly document-grounded AI assistant.

Answer using ONLY the provided context.
Do NOT use outside knowledge.
Do NOT guess.
Do NOT invent examples, advantages, explanations, or comparisons unless they are clearly supported by the context.

Rules:
1. If the answer is clearly present in the context, answer directly.
2. If the context supports only part of the answer, give only that supported part.
3. If the answer is not clearly supported by the context, reply exactly:
I could not find the answer in the indexed document context.
4. For comparison questions, compare only what is explicitly supported in the context.
5. Keep the answer short, factual, and clean.
""".strip()

    user_prompt = f"""
Context:
{context}

Question:
{question}

Answer:
""".strip()

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0.0,
            "num_predict": 160,
        }
    )

    answer = response["message"]["content"].strip()

    if not answer:
        return REFUSAL_TEXT

    lowered = answer.lower().strip()
    if "could not find" in lowered and "context" in lowered:
        return REFUSAL_TEXT

    return answer