"""Vector store wrapper around Chroma (uses its built-in local embedding model)."""
from . import config
from .chunker import chunk_text
from .loader import Page


# Tests can set this to a lightweight offline embedding function.
# None means Chroma's default local model (downloaded on first use).
embedding_function = None


def _collection():
    import chromadb  # imported lazily so unit tests don't need it

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    kwargs = {} if embedding_function is None else {"embedding_function": embedding_function}
    return client.get_or_create_collection(config.COLLECTION, **kwargs)


def index_pages(pages: list[Page]) -> int:
    """Chunk and index pages. Re-indexing the same file replaces its old chunks."""
    col = _collection()
    for source in {p.source for p in pages}:
        col.delete(where={"source": source})

    ids, docs, metas = [], [], []
    for page in pages:
        for j, chunk in enumerate(chunk_text(page.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            ids.append(f"{page.source}-p{page.page}-c{j}")
            docs.append(chunk)
            metas.append({"source": page.source, "page": page.page})

    batch = 100
    for i in range(0, len(ids), batch):
        col.add(ids=ids[i:i + batch], documents=docs[i:i + batch], metadatas=metas[i:i + batch])
    return len(ids)


def search(question: str, top_k: int = config.TOP_K) -> list[dict]:
    col = _collection()
    if col.count() == 0:
        return []
    res = col.query(query_texts=[question], n_results=min(top_k, col.count()))
    return [
        {"text": doc, "source": meta["source"], "page": meta["page"], "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def count() -> int:
    return _collection().count()
