
import logging
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import Language, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000  # max characters per chunk
CHUNK_OVERLAP = 200  # characters shared between neighbouring chunks, so context isn't cut mid-thought

MARKDOWN_FILE_TYPES = {"md", "markdown"}
# Markdown is first cut into sections at these headings; the heading text goes
# into metadata (e.g. {"h1": "Guide", "h2": "Install"}) instead of the chunk.
MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# Code files get a splitter that prefers to cut between functions and classes.
LANGUAGE_BY_FILE_TYPE = {
    "py": Language.PYTHON,
    "js": Language.JS,
    "ts": Language.TS,
    "java": Language.JAVA,
    "c": Language.C,
    "cpp": Language.CPP,
    "h": Language.CPP,
    "cs": Language.CSHARP,
    "go": Language.GO,
    "rs": Language.RUST,
    "rb": Language.RUBY,
    "php": Language.PHP,
}


def _get_splitter(file_type: str, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    language = LANGUAGE_BY_FILE_TYPE.get(file_type)
    if language:
        return RecursiveCharacterTextSplitter.from_language(
            language, chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )
    # Tries paragraphs first, then lines, then sentences, then words.
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",  # keep the "." on the sentence it ends
        add_start_index=True,
    )


def _split_markdown_sections(doc: Document) -> list[Document]:
    """Cut a Markdown document at its headings, one Document per section.

    Note: start_index on Markdown chunks is relative to their section.
    """
    sections = MarkdownHeaderTextSplitter(MARKDOWN_HEADERS).split_text(doc.page_content)
    return [Document(page_content=s.page_content, metadata={**doc.metadata, **s.metadata}) for s in sections]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into chunks, keeping each chunk's original metadata.

    Every chunk also gets:
      - start_index: character position of the chunk within its document
      - chunk_index: position of the chunk within its source file
      - chunk_id:    UUID built from source + chunk_index, so re-loading the same
                     file gives the same ids (Qdrant needs UUID or integer ids)
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitters: dict[str, RecursiveCharacterTextSplitter] = {}
    chunks: list[Document] = []
    chunks_per_source: dict[str, int] = {}

    for doc in documents:
        file_type = doc.metadata.get("file_type", "")
        if file_type not in splitters:
            splitters[file_type] = _get_splitter(file_type, chunk_size, chunk_overlap)

        parts = _split_markdown_sections(doc) if file_type in MARKDOWN_FILE_TYPES else [doc]
        for chunk in splitters[file_type].split_documents(parts):
            source = chunk.metadata.get("source", "unknown")
            index = chunks_per_source.get(source, 0)
            chunks_per_source[source] = index + 1

            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{index}"))
            chunks.append(chunk)

    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


if __name__ == "__main__":
    import sys

    from agentic_rag.document_loader import load_documents

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    target = sys.argv[1] if len(sys.argv) > 1 else "data"
    chunks = chunk_documents(load_documents(target))
    print(f"\n{len(chunks)} chunks from {target}")
    for chunk in chunks[:5]:
        meta = chunk.metadata
        print(f"- {meta['file_name']} #{meta['chunk_index']} ({len(chunk.page_content)} chars)\n  {chunk.page_content[:150]!r}")
