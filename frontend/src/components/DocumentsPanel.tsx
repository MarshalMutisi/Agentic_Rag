"use client";

import { useEffect, useRef, useState } from "react";

import { DocumentInfo, UploadProgress, UploadResult, Usage, api, uploadDocuments } from "@/lib/api";

// Share of the bar for sending the files; the rest is the server's work on each file.
const SENDING_SHARE = 0.15;
// Where each server stage starts and ends within one file's share of the bar.
// Embedding is by far the slowest step, so it gets most of the room.
const STAGE_RANGES: Record<Exclude<UploadProgress["stage"], "uploading">, [number, number]> = {
  reading: [0, 0.05],
  chunking: [0.05, 0.1],
  embedding: [0.1, 0.85],
  storing: [0.85, 1],
};

function progressPercent(progress: UploadProgress): number {
  const fraction = progress.total > 0 ? progress.done / progress.total : 0;
  if (progress.stage === "uploading") return SENDING_SHARE * fraction * 100;
  const [start, end] = STAGE_RANGES[progress.stage];
  const fileFraction = start + (end - start) * fraction;
  return (SENDING_SHARE + (1 - SENDING_SHARE) * ((progress.fileIndex + fileFraction) / progress.fileCount)) * 100;
}

function progressLabel(progress: UploadProgress): string {
  const megabytes = (bytes: number) => (bytes / (1024 * 1024)).toFixed(1);
  switch (progress.stage) {
    case "uploading":
      return progress.total > 0 ? `Sending ${megabytes(progress.done)} / ${megabytes(progress.total)} MB` : "Sending…";
    case "reading":
      return "Reading the file…";
    case "chunking":
      return "Splitting into chunks…";
    case "embedding":
      return `Embedding ${progress.done} / ${progress.total} chunks`;
    case "storing":
      return `Saving ${progress.done} / ${progress.total} chunks`;
  }
}

// Matches what the backend's document loader can read.
const ACCEPTED = ".pdf,.docx,.pptx,.xlsx,.xlsm,.csv,.html,.htm,.txt,.md,.markdown,.json,.jsonl,.xml,.yaml,.yml,.rst,.log,.py,.js,.ts,.java,.c,.cpp,.cs,.go,.rs,.rb,.php,.sql";

export default function DocumentsPanel() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [failures, setFailures] = useState<UploadResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  function applyResponse(response: { documents: DocumentInfo[]; usage: Usage }) {
    setDocuments(response.documents);
    setUsage(response.usage);
  }

  useEffect(() => {
    api.listDocuments()
      .then(applyResponse)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  async function upload(files: File[]) {
    if (!files.length || uploading) return;
    setUploading(true);
    setProgress({ stage: "uploading", fileIndex: 0, fileCount: files.length, done: 0, total: 0 });
    setError(null);
    setFailures([]);
    try {
      const response = await uploadDocuments(files, setProgress);
      applyResponse(response);
      setFailures(response.results.filter((result) => result.status === "error"));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
      setProgress(null);
    }
  }

  async function remove(fileName: string) {
    if (!window.confirm(`Delete "${fileName}" from your knowledge base?`)) return;
    setDeleting(fileName);
    setError(null);
    try {
      applyResponse(await api.deleteDocument(fileName));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setDeleting(null);
    }
  }

  const usedPercent = usage ? Math.min(100, (usage.chunks / usage.limit) * 100) : 0;
  const barColor = usedPercent > 95 ? "bg-danger" : usedPercent > 80 ? "bg-warning" : "bg-accent";

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="border-b border-line p-4">
        <h2 className="mb-3 text-sm font-semibold">Knowledge base</h2>

        {uploading && progress ? (
          <div className="rounded-xl border border-line px-4 py-4" role="status" aria-live="polite">
            <div className="mb-1 flex items-baseline justify-between gap-2 text-sm">
              <span className="truncate font-medium" title={progress.fileName}>
                {progress.fileName ?? (progress.fileCount === 1 ? "Uploading file" : `Uploading ${progress.fileCount} files`)}
              </span>
              <span className="shrink-0 text-xs tabular-nums text-muted">{Math.round(progressPercent(progress))}%</span>
            </div>
            <div
              className="h-2 overflow-hidden rounded-full bg-line"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(progressPercent(progress))}
              aria-label="Upload progress"
            >
              <div
                className="h-full rounded-full bg-accent transition-[width] duration-300"
                style={{ width: `${Math.max(2, progressPercent(progress))}%` }}
              />
            </div>
            <p className="mt-2 text-xs text-muted">
              {progressLabel(progress)}
              {progress.fileCount > 1 && progress.stage !== "uploading" && ` · file ${progress.fileIndex + 1} of ${progress.fileCount}`}
            </p>
          </div>
        ) : (
        <button
          type="button"
          onClick={() => fileInput.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            upload(Array.from(e.dataTransfer.files));
          }}
          disabled={uploading}
          className={`flex w-full flex-col items-center gap-1 rounded-xl border-2 border-dashed px-4 py-5 text-center transition ${
            dragging ? "border-accent bg-accent/5" : "border-line hover:border-accent/60"
          } disabled:opacity-60`}
        >
          <svg aria-hidden className="h-6 w-6 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0-4 4m4-4 4 4M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
          </svg>
          <span className="text-sm font-medium">Drop files or click to upload</span>
          <span className="text-xs text-muted">PDF, Word, PowerPoint, Excel, CSV, text…</span>
        </button>
        )}
        <input
          ref={fileInput}
          type="file"
          multiple
          accept={ACCEPTED}
          className="hidden"
          onChange={(e) => {
            upload(Array.from(e.target.files ?? []));
            e.target.value = ""; // allow picking the same file again
          }}
        />

        {usage && (
          <div className="mt-4">
            <div className="mb-1 flex justify-between text-xs text-muted">
              <span>Storage</span>
              <span>
                {usage.chunks.toLocaleString()} / {usage.limit.toLocaleString()} chunks
              </span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-line">
              <div className={`h-full rounded-full ${barColor} transition-all`} style={{ width: `${usedPercent}%` }} />
            </div>
          </div>
        )}
      </div>

      {(error || failures.length > 0) && (
        <div className="space-y-2 border-b border-line p-4">
          {error && <p role="alert" className="rounded-lg bg-danger/10 px-3 py-2 text-xs text-danger">{error}</p>}
          {failures.map((failure) => (
            <p key={failure.file_name} role="alert" className="rounded-lg bg-danger/10 px-3 py-2 text-xs text-danger">
              <span className="font-medium">{failure.file_name}:</span> {failure.error}
            </p>
          ))}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {loading ? (
          <p className="p-2 text-sm text-muted">Loading…</p>
        ) : documents.length === 0 ? (
          <p className="p-2 text-sm text-muted">No documents yet. Upload some to ask questions about them.</p>
        ) : (
          <ul className="space-y-0.5">
            {documents.map((document) => (
              <li key={document.file_name} className="group flex items-center gap-3 rounded-lg px-2 py-2 hover:bg-background">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-background text-[10px] font-semibold uppercase text-muted group-hover:bg-surface">
                  {document.file_name.split(".").pop()?.slice(0, 4)}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm" title={document.file_name}>{document.file_name}</p>
                  <p className="text-xs text-muted">
                    {document.chunks} {document.chunks === 1 ? "chunk" : "chunks"}
                    {document.uploaded_at && ` · ${new Date(document.uploaded_at * 1000).toLocaleDateString()}`}
                  </p>
                </div>
                <button
                  onClick={() => remove(document.file_name)}
                  disabled={deleting === document.file_name}
                  aria-label={`Delete ${document.file_name}`}
                  className="rounded-md p-1.5 text-muted opacity-100 transition hover:bg-danger/10 hover:text-danger disabled:opacity-50 md:opacity-0 md:group-hover:opacity-100 md:focus:opacity-100"
                >
                  <svg aria-hidden className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 7h16M10 11v6m4-6v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3" />
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
