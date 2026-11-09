import type { NextConfig } from "next";

// Set by the Dockerfile: build plain static files for FastAPI to serve, so the
// image needs no Node server. Local `npm run dev` is unaffected.
const staticExport = process.env.STATIC_EXPORT === "1";

const nextConfig: NextConfig = {
  // Next.js rejects cacheComponents and partialPrefetching in export mode.
  ...(staticExport
    ? { output: "export", trailingSlash: true }
    : { cacheComponents: true, partialPrefetching: true }),
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
