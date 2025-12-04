const BASE = 'https://storage.googleapis.com/beatdetect-app-artifacts/artifacts';
const METADATA_URL = `${BASE}/metadata.json`;

const METADATA_CACHE_NAME = 'beat-detect-metadata';
const MODEL_CACHE_NAME = 'beat-detect-models';

export type ArtifactKey = 'beat_model' | 'preprocess' | 'init_dist' | 'transitions';
export interface ModelMetadata {
  latest: string;
  files: Record<ArtifactKey, string>;
}
let metadataMemoryCache: ModelMetadata | null = null;

export async function fetchMetadata(): Promise<ModelMetadata> {
  if (metadataMemoryCache) return metadataMemoryCache;

  const cache = await caches.open(METADATA_CACHE_NAME);

  const cached = await cache.match(METADATA_URL);
  if (cached) {
    try {
      const json = await cached.json();
      metadataMemoryCache = json;
      return json;
    } catch (err) {
      console.warn('Corrupted cached metadata.json. Deleting + refetching.', err);
      await cache.delete(METADATA_URL);
    }
  }

  const res = await fetch(METADATA_URL, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch metadata.json: ${res.status} ${res.statusText}`);
  }

  const data = await res.clone().json();
  await cache.put(METADATA_URL, res);
  metadataMemoryCache = data;
  return data;
}

export async function getModelURL(key: ArtifactKey): Promise<string> {
  const metadata = await fetchMetadata();

  const filename = metadata.files[key];

  if (!filename) {
    throw new Error(`Model key "${key}" does not exist in metadata.files`);
  }

  return `${BASE}/${filename}`;
}

export async function fetchBinaryCached(url: string): Promise<ArrayBuffer> {
  const cache = await caches.open(MODEL_CACHE_NAME);

  const cached = await cache.match(url);
  if (cached) return cached.arrayBuffer();

  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }

  await cache.put(url, res.clone());
  return res.arrayBuffer();
}
