"use client";

import { useRouter } from "next/navigation";
import { useEffect, useLayoutEffect, useState } from "react";

import { api, getToken, setToken } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Next.js keeps this page alive (just hidden) after logging in, and shows it again on
  // logout. Clear the form when it's hidden, so the previous user's username and password
  // are never left filled in, and the button isn't stuck on "Please wait…".
  useLayoutEffect(() => () => {
    setMode("login");
    setUsername("");
    setPassword("");
    setError(null);
    setSubmitting(false);
  }, []);

  // Already logged in: go straight to the app.
  useEffect(() => {
    if (getToken()) router.replace("/");
  }, [router]);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const { access_token } = mode === "login"
        ? await api.login(username, password)
        : await api.register(username, password);
      setToken(access_token);
      router.replace("/");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      setSubmitting(false);
    }
  }

  const isLogin = mode === "login";

  return (
    <main className="flex flex-1 items-center justify-center px-4 py-12">
      <div className="w-full max-w-sm">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-11 w-11 items-center justify-center rounded-xl bg-accent text-lg font-semibold text-white">
            R
          </div>
          <h1 className="text-2xl font-semibold tracking-tight">Agentic RAG</h1>
          <p className="mt-1 text-sm text-muted">
            Ask questions about your documents, with the web as backup.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4 rounded-2xl border border-line bg-surface p-6 shadow-sm">
          <h2 className="text-base font-medium">{isLogin ? "Log in" : "Create an account"}</h2>

          <label className="block">
            <span className="mb-1 block text-sm text-muted">Username</span>
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              required
              className="w-full rounded-lg border border-line bg-background px-3 py-2 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
            />
          </label>

          <label className="block">
            <span className="mb-1 block text-sm text-muted">Password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={isLogin ? "current-password" : "new-password"}
              required
              className="w-full rounded-lg border border-line bg-background px-3 py-2 text-sm outline-none focus:border-accent focus:ring-2 focus:ring-accent/20"
            />
            {!isLogin && <span className="mt-1 block text-xs text-muted">At least 8 characters</span>}
          </label>

          {error && (
            <p role="alert" className="rounded-lg bg-danger/10 px-3 py-2 text-sm text-danger">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition hover:bg-accent/90 disabled:opacity-60"
          >
            {submitting ? "Please wait…" : isLogin ? "Log in" : "Create account"}
          </button>
        </form>

        <p className="mt-4 text-center text-sm text-muted">
          {isLogin ? "No account yet?" : "Already have an account?"}{" "}
          <button
            type="button"
            onClick={() => {
              setMode(isLogin ? "register" : "login");
              setUsername("");
              setPassword("");
              setError(null);
            }}
            className="font-medium text-accent hover:underline"
          >
            {isLogin ? "Create one" : "Log in"}
          </button>
        </p>
      </div>
    </main>
  );
}
