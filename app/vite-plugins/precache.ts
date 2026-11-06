import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import type { Plugin } from "vite";

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    return statSync(path).isDirectory() ? walk(path) : [path];
  });
}

/**
 * After the bundle is written, list every emitted file and bake the list and a content
 * hash into dist/sw.js. A new build changes the hash, so the old cache is replaced.
 */
export function precache(): Plugin {
  let outDir = "dist";
  return {
    name: "precache",
    apply: "build",
    configResolved(config) {
      outDir = config.build.outDir;
    },
    closeBundle() {
      const files = walk(outDir)
        .map((path) => relative(outDir, path).split("\\").join("/"))
        .filter((file) => file !== "sw.js" && !file.endsWith(".map"))
        .sort();
      const hash = createHash("sha256");
      for (const file of files) hash.update(file).update(readFileSync(join(outDir, file)));
      const swPath = join(outDir, "sw.js");
      const source = readFileSync(swPath, "utf8")
        .replace("__PRECACHE__", ["", ...files].join("|"))
        .replace("__BUILD_ID__", hash.digest("hex").slice(0, 12));
      writeFileSync(swPath, source);
    },
  };
}
