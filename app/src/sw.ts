/// <reference lib="webworker" />
export {};
declare const self: ServiceWorkerGlobalScope;

// Both tokens are replaced after the build by vite-plugins/precache.ts.
const BUILD_ID = "__BUILD_ID__";
const PRECACHE = "__PRECACHE__".split("|");
const CACHE = `plant-doctor-${BUILD_ID}`;

const scoped = (path: string) => new URL(path, self.registration.scope).href;

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) => cache.addAll(PRECACHE.map((file) => new Request(scoped(file), { cache: "reload" }))))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET" || !request.url.startsWith(self.registration.scope)) return;
  const key = request.mode === "navigate" ? scoped("index.html") : request;
  event.respondWith(
    caches.match(key, { ignoreSearch: true }).then((hit) => hit ?? fetch(request)),
  );
});
