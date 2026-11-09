"use client";

import { useRouter } from "next/navigation";
import { useEffect, useLayoutEffect, useState } from "react";

import ChatPanel from "@/components/ChatPanel";
import DocumentsPanel from "@/components/DocumentsPanel";
import { ApiError, User, api, clearToken, getToken } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showDocuments, setShowDocuments] = useState(false);

  // Next.js keeps this page alive (just hidden) after navigating to /login, and shows it
  // again after the next login. Forget the user when it's hidden: that also discards the
  // chat and document list, so the next person to log in never sees them.
  useLayoutEffect(() => () => {
    setUser(null);
    setError(null);
    setShowDocuments(false);
  }, []);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    api.me().then(setUser).catch((e) => {
      if (e instanceof ApiError && e.status === 401) {
        clearToken();
        router.replace("/login");
      } else {
        setError(e.message);
      }
    });
  }, [router]);

  function logout() {
    clearToken();
    router.replace("/login");
  }

  if (error) {
    return (
      <main className="flex flex-1 items-center justify-center px-4">
        <p role="alert" className="max-w-md rounded-lg bg-danger/10 px-4 py-3 text-sm text-danger">{error}</p>
      </main>
    );
  }

  if (!user) {
    return <main className="flex flex-1 items-center justify-center text-sm text-muted">Loading…</main>;
  }

  return (
    <div className="flex h-dvh flex-col">
      <header className="flex items-center justify-between gap-3 border-b border-line bg-surface px-4 py-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-sm font-semibold text-white">R</div>
          <span className="font-semibold tracking-tight">Agentic RAG</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowDocuments((shown) => !shown)}
            className="rounded-lg border border-line px-3 py-1.5 text-sm md:hidden"
          >
            {showDocuments ? "Chat" : "Documents"}
          </button>
          <span className="hidden text-sm text-muted sm:inline">{user.username}</span>
          <button onClick={logout} className="rounded-lg px-3 py-1.5 text-sm text-muted hover:bg-background hover:text-foreground">
            Log out
          </button>
        </div>
      </header>

      {/* key: a different user always gets freshly created panels, never another user's state */}
      <div key={user.id} className="flex min-h-0 flex-1">
        {/* On phones the panels share the screen and the header button switches between them. */}
        <aside className={`${showDocuments ? "flex" : "hidden"} w-full flex-col border-r border-line bg-surface md:flex md:w-80 md:shrink-0`}>
          <DocumentsPanel />
        </aside>
        <main className={`${showDocuments ? "hidden" : "flex"} min-w-0 flex-1 flex-col md:flex`}>
          <ChatPanel />
        </main>
      </div>
    </div>
  );
}
