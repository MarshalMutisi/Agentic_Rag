import os
import time
import uuid
from typing import List, Dict, Any, Callable, Iterable, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    KeywordIndexParams,
    KeywordIndexType,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from agentic_rag.embeddings import EmbeddingManager

load_dotenv()

# Roughly 1 KB of text per chunk, so 10,000 chunks is ~10 MB of text per user.
# Override with MAX_CHUNKS_PER_USER in .env.
DEFAULT_MAX_CHUNKS_PER_USER = 10_000

# Payload fields we filter on. Qdrant server/cloud needs an index on each to filter fast.
PAYLOAD_INDEXES = {
    # is_tenant=True tells Qdrant to store each user's points together.
    "user_id": KeywordIndexParams(type=KeywordIndexType.KEYWORD, is_tenant=True),
    "source": PayloadSchemaType.KEYWORD,
    "uploaded_at": PayloadSchemaType.INTEGER,
}


class StorageLimitExceeded(Exception):
    """Raised when an upload would take a user over their chunk limit"""


class VectorStore:
    """Stores chunk embeddings in Qdrant and searches them

    Multi-user: every chunk is tagged with the user_id of its owner, and
    searches only ever look at the calling user's chunks. Always pass the id
    of the *logged-in* user here, never a value the user can type in.

    A "document" is one source file; its chunks share the same "source" path.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_manager: Optional[EmbeddingManager] = None,
        path: str = "qdrant_data",
        max_chunks_per_user: Optional[int] = None,
    ):
        """
        Initialize the vector store

        Args:
            collection_name: Name of the Qdrant collection to use
            embedding_manager: Used to embed chunks and queries (a new one is created if not given)
            path: Folder for the local on-disk database. Ignored if QDRANT_URL is set in .env,
                  in which case it connects to that Qdrant server (or Qdrant Cloud) instead.
            max_chunks_per_user: Storage limit per user, in chunks (default: MAX_CHUNKS_PER_USER
                                 from .env, else 10,000)
        """
        self.collection_name = collection_name
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.max_chunks_per_user = max_chunks_per_user or int(
            os.getenv("MAX_CHUNKS_PER_USER", DEFAULT_MAX_CHUNKS_PER_USER)
        )

        url = os.getenv("QDRANT_URL")
        self.is_local = not url
        if url:
            print(f"Connecting to Qdrant at {url}")
            # The client's default timeout is only 5 seconds, too short to upload a
            # batch of vectors over a slow connection.
            self.client = QdrantClient(
                url=url, api_key=os.getenv("QDRANT_API_KEY"), timeout=int(os.getenv("QDRANT_TIMEOUT", "60"))
            )
        else:
            print(f"Using local Qdrant database in '{path}'")
            # Allow use from other threads (e.g. FastAPI's worker threads). Local mode
            # isn't safe for concurrent calls, so callers must take turns (the API uses a lock).
            self.client = QdrantClient(path=path, force_disable_check_same_thread=True)

        self._create_collection()

    def _create_collection(self):
        """Create the collection if it doesn't exist yet"""
        dimension = self.embedding_manager.dimension

        if self.client.collection_exists(self.collection_name):
            existing = self.client.get_collection(self.collection_name).config.params.vectors.size
            if existing != dimension:
                raise ValueError(
                    f"Collection '{self.collection_name}' stores {existing}-dim vectors but the model "
                    f"'{self.embedding_manager.model_name}' makes {dimension}-dim vectors. "
                    f"Use another collection_name, or delete the database folder to start over."
                )
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            print(f"Created collection '{self.collection_name}' ({dimension}-dim, cosine)")

        # Local mode ignores payload indexes (filters still work, just unindexed).
        if not self.is_local:
            for field_name, field_schema in PAYLOAD_INDEXES.items():
                self.client.create_payload_index(self.collection_name, field_name, field_schema)

    @staticmethod
    def _user_filter(user_id: str, *conditions: Condition) -> Filter:
        """Filter that matches only the given user's chunks (plus any extra conditions)"""
        if not user_id:
            raise ValueError("user_id is required")
        return Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id)), *conditions])

    @staticmethod
    def _source_condition(sources: Iterable[str]) -> FieldCondition:
        """Condition that matches chunks of any of the given documents"""
        return FieldCondition(key="source", match=MatchAny(any=list(sources)))

    def _delete(self, points_filter: Filter) -> int:
        """Delete every point matching the filter and return how many there were"""
        count = self.client.count(self.collection_name, count_filter=points_filter).count
        if count:
            self.client.delete(self.collection_name, points_selector=FilterSelector(filter=points_filter))
        return count

    def _upsert_with_retry(self, points: List[PointStruct], attempts: int = 3):
        """Upload one batch, retrying network hiccups (waits 2s, then 4s)"""
        for attempt in range(1, attempts + 1):
            try:
                self.client.upsert(collection_name=self.collection_name, points=points)
                return
            except Exception as e:
                if attempt == attempts:
                    raise
                print(f"Upload batch failed ({e}), retrying ({attempt}/{attempts - 1})...")
                time.sleep(2 ** attempt)

    def add_documents(
        self,
        chunks: List[Document],
        user_id: str,
        batch_size: int = 64,
        progress: Optional[Callable[[str, int, int], None]] = None,
    ):
        """
        Embed chunks and store them as belonging to user_id.

        Uploading a document this user already has replaces it: chunks of the
        old version that the new one doesn't overwrite are deleted, so an edited
        (e.g. shorter) file never leaves stale chunks behind.

        Args:
            chunks: Chunks from chunk_documents()
            user_id: Owner of these documents
            batch_size: How many points to send to Qdrant per request
            progress: Optional, called as progress(stage, done, total) with stage
                      "embedding" or "storing" (e.g. to show a progress bar in the UI)

        Raises:
            StorageLimitExceeded: If storing the chunks would take the user over
                                  max_chunks_per_user. Nothing is stored in that case.
            RuntimeError: If the upload to Qdrant keeps failing. The documents are
                          then removed entirely (including any earlier version).
        """
        self._user_filter(user_id)  # validates user_id
        if not chunks:
            print("No chunks to add")
            return

        # Chunks of documents being replaced are freed, so they don't count against the limit.
        sources = {chunk.metadata["source"] for chunk in chunks if "source" in chunk.metadata}
        replaced = self.count(user_id, self._source_condition(sources)) if sources else 0
        used_after = self.count(user_id) - replaced + len(chunks)
        if used_after > self.max_chunks_per_user:
            raise StorageLimitExceeded(
                f"User '{user_id}' would use {used_after} of {self.max_chunks_per_user} chunks. "
                f"Delete some documents first (see list_documents / delete_document)."
            )

        embeddings = self.embedding_manager.generate_embeddings(
            [chunk.page_content for chunk in chunks],
            progress_callback=(lambda done, total: progress("embedding", done, total)) if progress else None,
        )
        uploaded_at = int(time.time())

        points = [
            PointStruct(
                # user_id is part of the point id, so two users uploading the same
                # file get separate points instead of overwriting each other's.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{user_id}:{chunk.metadata.get('chunk_id') or uuid.uuid4()}")),
                vector=embedding.tolist(),
                payload={"text": chunk.page_content, **chunk.metadata, "user_id": user_id, "uploaded_at": uploaded_at},
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        try:
            for start in range(0, len(points), batch_size):
                self._upsert_with_retry(points[start:start + batch_size])
                if progress:
                    progress("storing", min(start + batch_size, len(points)), len(points))
        except Exception as e:
            # Don't leave a half-indexed document behind: remove these documents
            # completely, so the user can simply upload them again.
            if sources:
                self._delete(self._user_filter(user_id, self._source_condition(sources)))
            raise RuntimeError(f"Upload to the vector store failed, please try again ({e})") from e

        # Old chunks are only removed after the new ones are stored, so a failed
        # upload never leaves the user without the document.
        if sources:
            stale_filter = self._user_filter(user_id, self._source_condition(sources))
            stale_filter.must_not = [HasIdCondition(has_id=[point.id for point in points])]
            stale = self._delete(stale_filter)
            if stale:
                print(f"Removed {stale} outdated chunks from re-uploaded documents")

        print(f"Stored {len(points)} chunks for user '{user_id}' (using {self.count(user_id)}/{self.max_chunks_per_user})")

    def search(
        self, query: str, user_id: str, top_k: int = 5, score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Find the chunks most similar to a query, among user_id's documents only

        Args:
            query: The question or search text
            user_id: Whose documents to search
            top_k: Maximum number of results
            score_threshold: Drop results below this similarity (0-1), e.g. 0.3

        Returns:
            List of {"text", "score", "metadata"} dicts, best match first
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embedding_manager.embed_query(query).tolist(),
            query_filter=self._user_filter(user_id),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        ).points

        return [
            {
                "text": point.payload["text"],
                "score": point.score,
                "metadata": {k: v for k, v in point.payload.items() if k not in ("text", "user_id")},
            }
            for point in results
        ]

    def count(self, user_id: Optional[str] = None, *conditions: Condition) -> int:
        """Number of chunks stored for user_id, or in the whole collection if not given"""
        count_filter = self._user_filter(user_id, *conditions) if user_id is not None else None
        return self.client.count(self.collection_name, count_filter=count_filter).count

    def list_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List the documents a user has stored

        Returns:
            List of {"source", "file_name", "chunks", "uploaded_at"} dicts, sorted by file name.
            uploaded_at is a Unix timestamp.
        """
        documents: Dict[str, Dict[str, Any]] = {}
        offset = None
        while True:
            records, offset = self.client.scroll(
                self.collection_name,
                scroll_filter=self._user_filter(user_id),
                limit=1000,
                offset=offset,
                with_payload=["source", "file_name", "uploaded_at"],
            )
            for record in records:
                source = record.payload.get("source", "unknown")
                document = documents.setdefault(source, {
                    "source": source,
                    "file_name": record.payload.get("file_name", source),
                    "chunks": 0,
                    "uploaded_at": record.payload.get("uploaded_at"),
                })
                document["chunks"] += 1
            if offset is None:
                break
        return sorted(documents.values(), key=lambda document: document["file_name"])

    def get_usage(self, user_id: str) -> Dict[str, int]:
        """How much of their storage limit a user is using"""
        used = self.count(user_id)
        return {
            "documents": len(self.list_documents(user_id)),
            "chunks": used,
            "limit": self.max_chunks_per_user,
            "remaining": max(self.max_chunks_per_user - used, 0),
        }

    def delete_document(self, user_id: str, source: str) -> int:
        """
        Delete one of a user's documents

        Args:
            user_id: Owner of the document
            source: The document's "source" path, as shown by list_documents()

        Returns:
            Number of chunks deleted (0 if the user has no such document)
        """
        deleted = self._delete(self._user_filter(user_id, self._source_condition([source])))
        print(f"Deleted {deleted} chunks of '{source}' for user '{user_id}'")
        return deleted

    def delete_user_documents(self, user_id: str) -> int:
        """Delete all of one user's chunks (e.g. when they delete their account)"""
        deleted = self._delete(self._user_filter(user_id))
        print(f"Deleted all {deleted} chunks of user '{user_id}'")
        return deleted

    def delete_expired_documents(self, max_age_days: float, user_id: Optional[str] = None) -> int:
        """
        Delete documents uploaded more than max_age_days ago (a retention policy).
        Meant to be run periodically, e.g. from a daily scheduled job.

        Args:
            max_age_days: Documents older than this are deleted
            user_id: Only clean up this user's documents (default: every user's)

        Returns:
            Number of chunks deleted
        """
        cutoff = int(time.time() - max_age_days * 24 * 60 * 60)
        older = FieldCondition(key="uploaded_at", range=Range(lt=cutoff))
        expired_filter = self._user_filter(user_id, older) if user_id is not None else Filter(must=[older])
        deleted = self._delete(expired_filter)
        print(f"Deleted {deleted} chunks older than {max_age_days} days")
        return deleted

    def reset(self):
        """Delete every chunk of every user, keeping the (now empty) collection"""
        # Deleting points rather than the collection: in local mode on Windows,
        # delete_collection() can leave the locked database file behind, and
        # recreating the collection then brings the old points back.
        self.client.delete(self.collection_name, points_selector=FilterSelector(filter=Filter()))
        print(f"Reset collection '{self.collection_name}'")

    def close(self):
        """Close the database connection (releases the lock on the local folder)"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    import sys

    from agentic_rag.chunker import chunk_documents
    from agentic_rag.document_loader import load_documents

    target = sys.argv[1] if len(sys.argv) > 1 else "data"
    query = sys.argv[2] if len(sys.argv) > 2 else "What is this document about?"
    user_id = sys.argv[3] if len(sys.argv) > 3 else "demo-user"

    with VectorStore() as store:
        store.add_documents(chunk_documents(load_documents(target)), user_id=user_id)

        print(f"\nUsage for {user_id}: {store.get_usage(user_id)}")
        for document in store.list_documents(user_id):
            print(f"- {document['file_name']}: {document['chunks']} chunks")

        print(f"\nTop results for {user_id}: {query!r}")
        for result in store.search(query, user_id=user_id, top_k=3):
            print(f"- {result['score']:.3f} {result['metadata']['file_name']}: {result['text'][:100]!r}")
