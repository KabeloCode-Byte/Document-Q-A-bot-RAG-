# SA Document Q&A (RAG)

Ask questions about a set of documents and get answers that **cite their sources** (file and page).
Built as a retrieval-augmented generation (RAG) pipeline: documents are chunked, embedded into a
local vector store, and the most relevant passages are given to an LLM that must answer from them.

```
PDF / TXT / MD  ->  loader  ->  chunker  ->  Chroma (embeddings)
                                                  |
question  ->  top-k retrieval  ->  prompt with numbered passages  ->  LLM  ->  answer + citations
                                                  |
                          FastAPI  (/ask, /health)  +  static chat page
```

## Features

- Page-aware ingestion of PDFs, `.txt` and `.md` files
- Overlapping chunks that prefer paragraph/sentence boundaries
- Local persistent vector store (Chroma) with a built-in local embedding model, so no embedding API is needed
- Answers grounded in retrieved text only, with inline `[1]`, `[2]` citations and source snippets
- Graceful fallback: with no API key it still returns the most relevant passages
- FastAPI backend and a small dark/light-mode web UI
- Unit and integration tests that run offline

## Quick start

```bash
git clone https://github.com/KabeloCode-Byte/sa-doc-qa.git
cd sa-doc-qa

python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then add your ANTHROPIC_API_KEY
```

1. Put your public documents in `data/raw/` (PDF, TXT or MD).
2. Index them:
   ```bash
   # Windows PowerShell:  $env:PYTHONPATH="src"     macOS/Linux:  export PYTHONPATH=src
   python -m docqa.ingest
   ```
3. Start the app:
   ```bash
   uvicorn docqa.api:app --reload
   ```
4. Open http://127.0.0.1:8000 and ask a question.

The first ingest downloads Chroma's small embedding model once, so it needs internet access.

## API

`POST /ask`

```json
{ "question": "What must a credit provider do before granting a loan?", "top_k": 5 }
```

Returns:

```json
{
  "answer": "... [1] ...",
  "sources": [{ "id": 1, "source": "act.pdf", "page": 12, "snippet": "..." }]
}
```

`GET /health` returns the number of indexed chunks.

## Configuration

Set in `.env` (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Needed for generated answers |
| `LLM_MODEL` | `claude-sonnet-5-5` | Model used to write the answer |
| `CHUNK_SIZE` | `900` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Characters shared between chunks |
| `TOP_K` | `5` | Passages retrieved per question |

## Tests

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest -q
```

The pipeline tests use a tiny hashing embedder, so they run without downloading any model.

## Suggested documents (public only)

- South African legislation, for example the National Credit Act
- A university handbook or rule book
- Published annual reports

Use public documents only. Do not index confidential or employer-internal material.

## Limitations

- Scanned PDFs with no text layer are not supported (they would need OCR)
- Retrieval is plain vector similarity; tables and multi-column layouts may extract poorly
- This is a study aid, not legal or financial advice. Always check the original document.

## Roadmap

- [ ] Hybrid search (BM25 + vectors) and a re-ranker
- [ ] A small evaluation set with retrieval hit-rate and answer-faithfulness metrics
- [ ] Streaming answers and a file-upload endpoint
- [ ] Dockerfile and CI workflow

## Project layout

```
src/docqa/   config, loader, chunker, store (Chroma), rag (retrieve + answer), api, ingest
static/      chat web page
tests/       chunker unit tests and offline pipeline tests
data/raw/    your documents (git-ignored)
```
