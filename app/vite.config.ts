import { defineConfig } from "vite";
import pkg from "./package.json" with { type: "json" };
import { precache } from "./vite-plugins/precache.ts";

export default defineConfig({
  base: "./",
  define: { __APP_VERSION__: JSON.stringify(pkg.version) },
  plugins: [precache()],
  build: {
    target: "es2022",
    rollupOptions: {
      input: { main: "index.html", sw: "src/sw.ts" },
      output: {
        entryFileNames: (chunk) => (chunk.name === "sw" ? "sw.js" : "assets/[name]-[hash].js"),
      },
    },
  },
});
