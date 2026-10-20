"""Load PDFs and text files into (source, page, text) records."""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    source: str
    page: int
    text: str


def load_file(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [
            Page(path.name, i + 1, page.extract_text() or "")
            for i, page in enumerate(reader.pages)
        ]
    if suffix in {".txt", ".md"}:
        return [Page(path.name, 1, path.read_text(encoding="utf-8", errors="ignore"))]
    return []


def load_directory(directory: Path) -> list[Page]:
    pages: list[Page] = []
    for path in sorted(directory.iterdir()):
        if path.is_file():
            pages.extend(load_file(path))
    return pages
