"""CLI: index every PDF/txt/md file in data/raw.

Usage:  python -m docqa.ingest
"""
from . import config, store
from .loader import load_directory


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    pages = load_directory(config.RAW_DIR)
    if not pages:
        print(f"No documents found in {config.RAW_DIR}. Add PDFs or .txt/.md files first.")
        return
    n = store.index_pages(pages)
    print(f"Indexed {n} chunks from {len({p.source for p in pages})} file(s).")


if __name__ == "__main__":
    main()
