// Client for the FastAPI backend (src/agentic_rag/api.py).

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const TOKEN_KEY = "agentic-rag-token";

export type User = { id: string; username: string };

export type DocumentInfo = { file_name: string; chunks: number; uploaded_at: number | null };
export type Usage = { documents: number; chunks: number; limit: number; remaining: number };
export type DocumentsResponse = { documents: DocumentInfo[]; usage: Usage };
export type UploadResult = { file_name: string; status: "stored" | "error"; chunks: number; error?: string };

export type SourceType = "knowledge_base" | "web" | "knowledge_base+web" | "direct" | "none";
export type Source = {
  number: number;
  source_type: "knowledge_base" | "web";
  label: string;
  file_name?: string;
  title?: string;
  url?: string;
};
export type ChatAnswer = { answer: string; source_type: SourceType; sources: Source[] };

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

// localStorage can throw (private mode, blocked storage), so every access is guarded.
export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {}
}

export function clearToken() {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {}
}

function authHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return headers;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail) && body.detail[0]?.msg) return body.detail[0].msg;
  } catch {}
  return `Request failed (${response.status})`;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, headers: authHeaders(init.headers) });
  } catch {
    throw new ApiError(0, `Can't reach the server at ${API_URL}. Is the backend running?`);
  }
  if (!response.ok) throw new ApiError(response.status, await errorMessage(response));
  return response.json();
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

type AuthResponse = { access_token: string; user: User };

export const api = {
  login: (username: string, password: string) => postJson<AuthResponse>("/auth/login", { username, password }),
  register: (username: string, password: string) => postJson<AuthResponse>("/auth/register", { username, password }),
  me: () => request<User>("/auth/me"),

  listDocuments: () => request<DocumentsResponse>("/documents"),
  deleteDocument: (fileName: string) =>
    request<DocumentsResponse>(`/documents/${encodeURIComponent(fileName)}`, { method: "DELETE" }),
};

type ServerEvent = { event: string; data: any }; // eslint-disable-line @typescript-eslint/no-explicit-any

/**
 * Split complete server-sent events ("event: <name>\ndata: <json>\n\n") off the
 * front of a buffer. Returns the parsed events and the unfinished remainder.
 */
function takeEvents(buffer: string): { events: ServerEvent[]; rest: string } {
  const events: ServerEvent[] = [];
  let end;
  while ((end = buffer.indexOf("\n\n")) !== -1) {
    const rawEvent = buffer.slice(0, end);
    buffer = buffer.slice(end + 2);

    let event = "message";
    let data = "";
    for (const line of rawEvent.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7);
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (data) events.push({ event, data: JSON.parse(data) });
  }
  return { events, rest: buffer };
}

export type UploadStage = "uploading" | "reading" | "chunking" | "embedding" | "storing";
export type UploadProgress = {
  stage: UploadStage;
  fileName?: string;
  fileIndex: number;
  fileCount: number;
  done: number;
  total: number;
};
export type UploadResponse = DocumentsResponse & { results: UploadResult[] };

/**
 * Upload files, reporting progress: first the bytes being sent, then the server's
 * stages (reading, chunking, embedding, storing) as it streams them back.
 * Uses XMLHttpRequest because fetch can't report upload progress.
 */
export function uploadDocuments(files: File[], onProgress: (progress: UploadProgress) => void): Promise<UploadResponse> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/documents`);
    authHeaders().forEach((value, key) => xhr.setRequestHeader(key, value));

    let seen = 0;
    let buffer = "";
    let finished = false;

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        onProgress({ stage: "uploading", fileIndex: 0, fileCount: files.length, done: e.loaded, total: e.total });
      }
    };

    // The response arrives bit by bit; read whatever is new each time.
    function readNewEvents() {
      if (xhr.status !== 200) return;
      buffer += xhr.responseText.slice(seen);
      seen = xhr.responseText.length;
      const { events, rest } = takeEvents(buffer);
      buffer = rest;
      for (const { event, data } of events) {
        if (event === "progress") {
          onProgress({
            stage: data.stage,
            fileName: data.file_name,
            fileIndex: data.file_index,
            fileCount: data.file_count,
            done: data.done,
            total: data.total,
          });
        } else if (event === "done") {
          finished = true;
          resolve(data);
        } else if (event === "error") {
          finished = true;
          reject(new ApiError(500, data.message));
        }
      }
    }

    xhr.onprogress = readNewEvents;
    xhr.onload = () => {
      if (xhr.status !== 200) {
        let message = `Upload failed (${xhr.status})`;
        try {
          const detail = JSON.parse(xhr.responseText).detail;
          if (typeof detail === "string") message = detail;
        } catch {}
        reject(new ApiError(xhr.status, message));
        return;
      }
      readNewEvents();
      if (!finished) reject(new ApiError(0, "The upload ended unexpectedly. Please try again."));
    };
    xhr.onerror = () => reject(new ApiError(0, `Can't reach the server at ${API_URL}. Is the backend running?`));
    xhr.send(form);
  });
}

type ChatHandlers = {
  onStep: (text: string) => void;
  onAnswer: (answer: ChatAnswer) => void;
  onError: (message: string) => void;
};

/** Ask the agent a question, calling the handlers as server-sent events arrive. */
export async function streamChat(question: string, handlers: ChatHandlers, signal?: AbortSignal) {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ question }),
      signal,
    });
  } catch (e) {
    if (signal?.aborted) throw e;
    throw new ApiError(0, `Can't reach the server at ${API_URL}. Is the backend running?`);
  }
  if (!response.ok || !response.body) throw new ApiError(response.status, await errorMessage(response));

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const { events, rest } = takeEvents(buffer + value);
    buffer = rest;
    for (const { event, data } of events) {
      if (event === "step") handlers.onStep(data.text);
      else if (event === "answer") handlers.onAnswer(data);
      else if (event === "error") handlers.onError(data.message);
    }
  }
}
