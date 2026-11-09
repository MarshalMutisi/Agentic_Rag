"""FastAPI backend for the agentic RAG system.

Run with:  uv run uvicorn agentic_rag.api:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""

import json
import os
import tempfile
import threading
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentic_rag.agent import build_agent
from agentic_rag.auth import authenticate, create_token, create_user, get_current_user
from agentic_rag.chunker import chunk_documents
from agentic_rag.document_loader import load_file
from agentic_rag.vector_store import VectorStore

load_dotenv()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")).resolve()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")
FRONTEND_DIR = os.getenv("FRONTEND_DIR")  # folder with the built frontend (set in the Docker image)

# Source metadata that is safe and useful to show in the browser
# (leaves out e.g. "source", the file's path on the server).
PUBLIC_SOURCE_FIELDS = ("number", "source_type", "label", "file_name", "page", "slide", "sheet", "row", "title", "url")

resources: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model and connect to Qdrant once, at startup.
    store = VectorStore(
        collection_name=os.getenv("QDRANT_COLLECTION", "documents"),
        path=os.getenv("QDRANT_PATH", "qdrant_data"),
    )
    resources["store"] = store
    resources["agent"] = build_agent(store)
    # Local Qdrant can't handle concurrent calls, so requests take turns there.
    resources["store_lock"] = threading.Lock() if store.is_local else nullcontext()
    yield
    store.close()


app = FastAPI(title="Agentic RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Credentials(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def _user_dir(user: Dict[str, Any]) -> Path:
    return UPLOAD_DIR / user["id"]


def _safe_file_name(file_name: str) -> str:
    """Strip any folder parts, so a file name like "../../x" can't escape the user's folder"""
    name = Path(file_name or "").name.strip()
    if not name or name.startswith(".") or len(name) > 200:
        raise ValueError("Invalid file name")
    return name


def _documents_response(store: VectorStore, user: Dict[str, Any]) -> Dict[str, Any]:
    documents = [
        {"file_name": document["file_name"], "chunks": document["chunks"], "uploaded_at": document["uploaded_at"]}
        for document in store.list_documents(user["id"])
    ]
    return {"documents": documents, "usage": store.get_usage(user["id"])}


# ---------- Auth ----------

@app.post("/auth/register")
def register(credentials: Credentials):
    try:
        user = create_user(credentials.username, credentials.password)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return {"access_token": create_token(user), "user": user}


@app.post("/auth/login")
def login(credentials: Credentials):
    user = authenticate(credentials.username, credentials.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong username or password")
    return {"access_token": create_token(user), "user": user}


@app.get("/auth/me")
def me(user: Dict[str, Any] = Depends(get_current_user)):
    return user


# ---------- Documents ----------

@app.get("/documents")
def list_documents(user: Dict[str, Any] = Depends(get_current_user)):
    with resources["store_lock"]:
        return _documents_response(resources["store"], user)


def _store_file(
    raw_name: str, content: bytes, user: Dict[str, Any], report: Callable[[str, int, int], None]
) -> Dict[str, Any]:
    """Load, chunk, embed and store one uploaded file. Never raises: errors go in the result."""
    store: VectorStore = resources["store"]
    result = {"file_name": raw_name, "status": "error", "chunks": 0}
    try:
        file_name = _safe_file_name(raw_name)
        result["file_name"] = file_name
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError(f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

        # Work on a temporary copy (deleted automatically), so a failed upload
        # never replaces a good earlier version of the file.
        user_dir = _user_dir(user)
        user_dir.mkdir(parents=True, exist_ok=True)
        final_path = user_dir / file_name
        with tempfile.TemporaryDirectory(dir=user_dir, prefix=".upload-") as temp_dir:
            temp_path = Path(temp_dir) / file_name
            temp_path.write_bytes(content)

            report("reading", 0, 0)
            documents = load_file(temp_path)
            if not documents:
                raise ValueError("No text found in this file (scanned PDFs need OCR)")
            for document in documents:
                # Record the final location, so re-uploading this file replaces its chunks.
                document.metadata["source"] = str(final_path)

            report("chunking", 0, 0)
            chunks = chunk_documents(documents)
            with resources["store_lock"]:
                store.add_documents(chunks, user_id=user["id"], progress=report)
            os.replace(temp_path, final_path)
        result.update(status="stored", chunks=len(chunks))
    except Exception as e:  # includes StorageLimitExceeded
        result["error"] = str(e)
    return result


@app.post("/documents")
def upload_documents(files: List[UploadFile] = File(...), user: Dict[str, Any] = Depends(get_current_user)):
    """
    Upload one or more files into the user's knowledge base. Each file is handled
    on its own: one bad file doesn't stop the others. Uploading a file with the
    same name as an existing one replaces it.

    Streams server-sent events, so the browser can show a progress bar:
      progress: {"file_name", "file_index", "file_count", "stage", "done", "total"}
                stage is "reading", "chunking", "embedding" or "storing"
      done:     {"results", "documents", "usage"}
      error:    {"message"}
    """
    # Read the files now: FastAPI closes the uploads once this function returns,
    # before the streamed response is sent.
    uploads = [(upload.filename, upload.file.read(MAX_UPLOAD_BYTES + 1)) for upload in files]
    events: Queue = Queue()

    def work():
        try:
            results = []
            for index, (raw_name, content) in enumerate(uploads):
                def report(stage: str, done: int, total: int, index=index, raw_name=raw_name):
                    events.put(("progress", {
                        "file_name": raw_name, "file_index": index, "file_count": len(uploads),
                        "stage": stage, "done": done, "total": total,
                    }))
                results.append(_store_file(raw_name, content, user, report))
            with resources["store_lock"]:
                events.put(("done", {"results": results, **_documents_response(resources["store"], user)}))
        except Exception as e:
            print(f"Upload failed: {e!r}")
            events.put(("error", {"message": "Something went wrong while uploading. Please try again."}))

    # The work runs in its own thread and reports into the queue; the response
    # streams each report to the browser as soon as it arrives.
    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            event, data = events.get()
            yield _sse(event, data)
            if event in ("done", "error"):
                return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/documents/{file_name}")
def delete_document(file_name: str, user: Dict[str, Any] = Depends(get_current_user)):
    try:
        path = _user_dir(user) / _safe_file_name(file_name)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    store: VectorStore = resources["store"]
    with resources["store_lock"]:
        deleted = store.delete_document(user["id"], str(path))
        path.unlink(missing_ok=True)
        if not deleted:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
        return {"deleted_chunks": deleted, **_documents_response(store, user)}


# ---------- Chat ----------

def _sse(event: str, data: Dict[str, Any]) -> str:
    """Format one server-sent event"""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/chat")
def chat(request: ChatRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """
    Ask the agent a question. Streams server-sent events:
      step:   {"node", "text"}  each decision the agent makes, as it happens
      answer: {"answer", "source_type", "sources"}
      error:  {"message"}
    """
    def events():
        try:
            with resources["store_lock"]:
                state = {"question": request.question, "user_id": user["id"], "steps": []}
                for update in resources["agent"].stream(state, stream_mode="updates"):
                    for node, changes in update.items():
                        for step in changes.get("steps", []):
                            yield _sse("step", {"node": node, "text": step})
                        if "answer" in changes:  # from "generate" or "answer_directly"
                            sources = [
                                {key: source[key] for key in PUBLIC_SOURCE_FIELDS if key in source}
                                for source in changes["sources"]
                            ]
                            yield _sse("answer", {
                                "answer": changes["answer"],
                                "source_type": changes["source_type"],
                                "sources": sources,
                            })
        except Exception as e:
            print(f"Chat failed: {e!r}")
            yield _sse("error", {"message": "Something went wrong while answering. Please try again."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# In the Docker image the built frontend is served from here too, so the whole app
# is one process on one port. Must stay last: "/" catches whatever isn't matched above.
if FRONTEND_DIR:
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
