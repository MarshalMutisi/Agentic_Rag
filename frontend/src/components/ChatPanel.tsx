"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

import { ApiError, ChatAnswer, Source, SourceType, clearToken, streamChat } from "@/lib/api";

type Message =
  | { id: number; role: "user"; text: string }
  | { id: number; role: "assistant"; steps: string[]; answer?: ChatAnswer; error?: string; pending: boolean };

const SOURCE_BADGES: Record<SourceType, { label: string; className: string }> = {
  knowledge_base: { label: "From your documents", className: "bg-kb/10 text-kb" },
  web: { label: "From the web", className: "bg-web/10 text-web" },
  "knowledge_base+web": { label: "Documents + web", className: "bg-both/10 text-both" },
  direct: { label: "General answer, no search needed", className: "bg-accent/10 text-accent" },
  none: { label: "Answer not found", className: "bg-line text-muted" },
};

// One example per path the agent can take: your documents, the web, a direct answer.
const EXAMPLES = [
  "What are the key points in my documents?",
  "What is retrieval augmented generation?",
  "Write a short thank-you email to my team",
];

export default function ChatPanel() {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const nextId = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // Stop a running answer if the user leaves the page.
  useEffect(() => () => abort.current?.abort(), []);

  function updateAssistant(id: number, update: (message: Extract<Message, { role: "assistant" }>) => Partial<Message>) {
    setMessages((current) =>
      current.map((message) =>
        message.id === id && message.role === "assistant" ? ({ ...message, ...update(message) } as Message) : message,
      ),
    );
  }

  async function ask(question: string) {
    question = question.trim();
    if (!question || busy) return;

    const assistantId = nextId.current + 1;
    nextId.current += 2;
    setMessages((current) => [
      ...current,
      { id: assistantId - 1, role: "user", text: question },
      { id: assistantId, role: "assistant", steps: [], pending: true },
    ]);
    setInput("");
    setBusy(true);
    abort.current = new AbortController();

    try {
      await streamChat(
        question,
        {
          onStep: (text) => updateAssistant(assistantId, (message) => ({ steps: [...message.steps, text] })),
          onAnswer: (answer) => updateAssistant(assistantId, () => ({ answer })),
          onError: (error) => updateAssistant(assistantId, () => ({ error })),
        },
        abort.current.signal,
      );
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) {
        clearToken();
        router.replace("/login");
        return;
      }
      if (!abort.current?.signal.aborted) {
        updateAssistant(assistantId, () => ({ error: e instanceof Error ? e.message : "Something went wrong" }));
      }
    } finally {
      updateAssistant(assistantId, () => ({ pending: false }));
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
          {messages.length === 0 && (
            <div className="pt-12 text-center sm:pt-20">
              <h2 className="text-xl font-semibold tracking-tight">What do you want to know?</h2>
              <p className="mx-auto mt-2 max-w-md text-sm text-muted">
                Ask about your uploaded documents. If they don&apos;t have the answer, I&apos;ll search the web.
                General tasks like writing or maths are answered directly.
              </p>
              <div className="mt-6 flex flex-wrap justify-center gap-2">
                {EXAMPLES.map((example) => (
                  <button
                    key={example}
                    onClick={() => ask(example)}
                    className="rounded-full border border-line bg-surface px-3 py-1.5 text-sm text-muted transition hover:border-accent/50 hover:text-foreground"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) =>
            message.role === "user" ? (
              <div key={message.id} className="flex justify-end">
                <p className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-accent px-4 py-2.5 text-sm text-white">
                  {message.text}
                </p>
              </div>
            ) : (
              <AssistantMessage key={message.id} message={message} />
            ),
          )}
          <div ref={bottom} />
        </div>
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
        className="border-t border-line bg-surface px-4 py-3"
      >
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask(input);
              }
            }}
            rows={1}
            maxLength={2000}
            placeholder="Ask a question…"
            aria-label="Question"
            className="max-h-40 min-h-10 flex-1 resize-none rounded-xl border border-line bg-background px-3.5 py-2.5 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
          />
          <button
            type="submit"
            disabled={busy || !input.trim()}
            className="h-10 rounded-xl bg-accent px-4 text-sm font-medium text-white transition hover:bg-accent/90 disabled:opacity-50"
          >
            {busy ? "…" : "Ask"}
          </button>
        </div>
      </form>
    </div>
  );
}

function AssistantMessage({ message }: { message: Extract<Message, { role: "assistant" }> }) {
  const { answer, steps, error, pending } = message;
  const badge = answer ? SOURCE_BADGES[answer.source_type] : null;

  return (
    <div className="space-y-3">
      {pending && !answer ? (
        <ol className="space-y-1.5 rounded-xl border border-line bg-surface px-4 py-3 text-sm text-muted">
          {steps.map((step, i) => (
            <li key={i} className="flex gap-2">
              <span className="text-kb">✓</span>
              <span>{step}</span>
            </li>
          ))}
          <li className="flex items-center gap-2">
            <span className="h-3 w-3 animate-spin rounded-full border-2 border-accent border-t-transparent" />
            <span>{steps.length === 0 ? "Reading your question…" : "Working…"}</span>
          </li>
        </ol>
      ) : (
        steps.length > 0 && (
          <details className="text-xs text-muted">
            <summary className="cursor-pointer select-none hover:text-foreground">How this was answered ({steps.length} steps)</summary>
            <ol className="mt-2 list-decimal space-y-1 pl-5">
              {steps.map((step, i) => <li key={i}>{step}</li>)}
            </ol>
          </details>
        )
      )}

      {error && <p role="alert" className="rounded-xl bg-danger/10 px-4 py-3 text-sm text-danger">{error}</p>}

      {answer && badge && (
        <div className="rounded-2xl rounded-bl-md border border-line bg-surface px-4 py-3">
          <span className={`mb-2 inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${badge.className}`}>{badge.label}</span>
          <div className="markdown text-sm leading-relaxed">
            <ReactMarkdown
              components={{
                a: ({ href, children }) =>
                  href?.startsWith("#cite-") ? (
                    <a
                      href={href}
                      className="mx-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded bg-accent/10 px-1 align-super text-[10px] font-semibold text-accent no-underline hover:bg-accent/20"
                    >
                      {children}
                    </a>
                  ) : (
                    <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>
                  ),
              }}
            >
              {linkCitations(answer.answer, message.id)}
            </ReactMarkdown>
          </div>
          {answer.sources.length > 0 && <SourceList sources={answer.sources} messageId={message.id} />}
        </div>
      )}
    </div>
  );
}

/** Turn "[1]" citations into links to the matching source below the answer. */
function linkCitations(text: string, messageId: number): string {
  return text.replace(/\[(\d+)\](?!\()/g, (_, n) => `[${n}](#cite-${messageId}-${n})`);
}

function SourceList({ sources, messageId }: { sources: Source[]; messageId: number }) {
  return (
    <div className="mt-3 border-t border-line pt-3">
      <p className="mb-2 text-xs font-medium text-muted">Sources</p>
      <ol className="space-y-1.5">
        {sources.map((source) => (
          <li key={source.number} id={`cite-${messageId}-${source.number}`} className="flex scroll-mt-4 gap-2 text-xs target:rounded target:bg-accent/10">
            <span className="mt-px flex h-4 min-w-4 shrink-0 items-center justify-center rounded bg-background px-1 font-semibold text-muted">
              {source.number}
            </span>
            {source.source_type === "web" && source.url ? (
              <a href={source.url} target="_blank" rel="noopener noreferrer" className="min-w-0 truncate text-web hover:underline" title={source.url}>
                {source.title || source.url}
                <span className="text-muted"> · {safeHostname(source.url)}</span>
              </a>
            ) : (
              <span className="min-w-0 truncate">
                <span className="text-kb">Document:</span> {source.label.replace(/^knowledge base: /, "")}
              </span>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

function safeHostname(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}
