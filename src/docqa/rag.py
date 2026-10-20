"""Retrieve relevant chunks, then ask the LLM to answer using only those chunks."""
from . import config, store

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite passages inline like [1] or [2]. If the context does not contain the "
    "answer, say you could not find it in the documents. Do not use outside "
    "knowledge, and do not give legal or financial advice."
)


def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] ({p['source']}, page {p['page']})\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(question: str, top_k: int = config.TOP_K) -> dict:
    passages = store.search(question, top_k)
    if not passages:
        return {
            "answer": "No documents are indexed yet. Add files to data/raw and run the ingest script.",
            "sources": [],
        }

    sources = [
        {"id": i, "source": p["source"], "page": p["page"], "snippet": p["text"][:300]}
        for i, p in enumerate(passages, start=1)
    ]

    if not config.ANTHROPIC_API_KEY:
        return {
            "answer": "No ANTHROPIC_API_KEY set, so here are the most relevant passages instead.",
            "sources": sources,
        }

    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, passages)}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    return {"answer": text, "sources": sources}
