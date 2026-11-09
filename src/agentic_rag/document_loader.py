
import logging
from pathlib import Path

from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Extensions that are plain text and can be read as-is.
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".json", ".jsonl", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs",
    ".rb", ".php", ".sql", ".sh", ".ps1", ".css",
}


def _load_pptx(path: Path) -> list[Document]:
    """One Document per slide."""
    from pptx import Presentation

    docs = []
    for number, slide in enumerate(Presentation(path).slides, start=1):
        texts = [shape.text_frame.text for shape in slide.shapes if shape.has_text_frame]
        text = "\n".join(t for t in texts if t.strip())
        if text:
            docs.append(Document(page_content=text, metadata={"source": str(path), "slide": number}))
    return docs


def _load_xlsx(path: Path) -> list[Document]:
    """One Document per sheet, each row written as comma-separated values."""
    from openpyxl import load_workbook

    docs = []
    workbook = load_workbook(path, read_only=True, data_only=True)
    for sheet in workbook.worksheets:
        rows = [
            ", ".join("" if cell is None else str(cell) for cell in row)
            for row in sheet.iter_rows(values_only=True)
            if any(cell is not None for cell in row)
        ]
        if rows:
            docs.append(Document(page_content="\n".join(rows), metadata={"source": str(path), "sheet": sheet.title}))
    workbook.close()
    return docs


def load_file(file_path: str | Path) -> list[Document]:
    """Load a single file, picking the right loader from its extension."""
    # Absolute path, so "data/a.txt" and "C:/.../data/a.txt" get the same
    # "source" and the vector store recognises a re-upload of the same file.
    path = Path(file_path).resolve()
    ext = path.suffix.lower()

    if ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()  # one Document per page
    elif ext == ".docx":
        docs = Docx2txtLoader(str(path)).load()
    elif ext == ".pptx":
        docs = _load_pptx(path)
    elif ext in {".xlsx", ".xlsm"}:
        docs = _load_xlsx(path)
    elif ext == ".csv":
        docs = CSVLoader(str(path), encoding="utf-8").load()  # one Document per row
    elif ext in {".html", ".htm"}:
        docs = BSHTMLLoader(str(path), open_encoding="utf-8", bs_kwargs={"features": "html.parser"}).load()
    elif ext in TEXT_EXTENSIONS:
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        # Unknown extension: try it as UTF-8 text, give up if it is binary.
        try:
            docs = TextLoader(str(path), encoding="utf-8").load()
        except RuntimeError:
            raise ValueError(f"Unsupported file type: {path.name}")

    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata["file_name"] = path.name
        doc.metadata["file_type"] = ext.lstrip(".")
    # Drop empty pages/rows (e.g. scanned PDF pages with no text layer).
    return [doc for doc in docs if doc.page_content.strip()]


def load_documents(path: str | Path, recursive: bool = True) -> list[Document]:
    """Load a file, or every file inside a folder.

    Files that fail to load are logged and skipped so one bad file
    doesn't stop the whole folder from loading.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file or folder: {path}")
    if path.is_file():
        return load_file(path)

    pattern = "**/*" if recursive else "*"
    documents = []
    for file in sorted(path.glob(pattern)):
        if not file.is_file() or file.name.startswith("."):
            continue
        try:
            docs = load_file(file)
            documents.extend(docs)
            logger.info("Loaded %s (%d documents)", file.name, len(docs))
        except Exception as e:
            logger.warning("Skipped %s: %s", file.name, e)
    return documents
    


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    target = sys.argv[1] if len(sys.argv) > 1 else "data"
    docs = load_documents(target)
    print(f"\nLoaded {len(docs)} documents from {target}")
    for doc in docs[:5]:
        print(f"- {doc.metadata}\n  {doc.page_content[:150]!r}")
