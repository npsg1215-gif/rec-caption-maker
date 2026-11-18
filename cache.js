// AI 모델 파일 저장소: 브라우저 기본 저장소(Cache)는 300MB가 넘는 파일을 못 넣는 경우가 있어서,
// 모델 파일을 브라우저의 파일 저장 공간(OPFS)에 직접 저장합니다. 한 번 받으면 다시 받지 않아요.
// 자막 AI(worker.js)와 요약 AI(summarize.js)가 함께 써요.
export const opfsCache = {
  dir: null,
  async root() {
    if (!this.dir) {
      const r = await navigator.storage.getDirectory();
      this.dir = await r.getDirectoryHandle('models', { create: true });
    }
    return this.dir;
  },
  name(key) { return String(key).replace(/^https?:\/\//, '').replace(/[^a-zA-Z0-9._-]/g, '_'); },
  async match(key) {
    try {
      const d = await this.root();
      const n = this.name(key);
      await d.getFileHandle(n + '.ok'); // 다 받은 파일에만 표시가 있음
      const f = await (await d.getFileHandle(n)).getFile();
      return new Response(f, { headers: { 'content-length': String(f.size) } });
    } catch {
      return undefined;
    }
  },
  async put(key, response) {
    const d = await this.root();
    const n = this.name(key);
    try { await d.removeEntry(n + '.ok'); } catch {}
    const fh = await d.getFileHandle(n, { create: true });
    const reader = response.body.getReader();
    // 받는 도중 끊기면(불러오기가 실패해서 내려받기가 취소되는 등) '완료' 표시를 하지 않아요
    const expected = Number(response.headers.get('content-length') || response.headers.get('x-linked-size') || 0);
    let written = 0;
    if ('createSyncAccessHandle' in fh && typeof WorkerGlobalScope !== 'undefined') {
      const h = await fh.createSyncAccessHandle(); // 워커에서는 더 빠른 방법으로 써요
      try {
        h.truncate(0);
        let at = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          h.write(value, { at });
          at += value.byteLength;
        }
        h.flush();
        written = at;
      } finally {
        h.close();
      }
    } else {
      const w = await fh.createWritable();
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          await w.write(value);
          written += value.byteLength;
        }
      } finally {
        await w.close();
      }
    }
    if (expected && written !== expected) return;
    await d.getFileHandle(n + '.ok', { create: true });
  },
};

// 큰 모델은 먼저 직접 내려받아 저장해 둬요 (크기를 확인해서, 끝까지 받은 파일만 '완료' 표시).
// 이렇게 하면 불러올 때 파일 크기를 정확히 알아서 메모리를 아끼고, 중간에 끊긴 파일이 남지 않아요.
// files: [{ url, size }], onProgress(받은 바이트, 전체 바이트)
export async function prefetchFiles(files, onProgress) {
  const total = files.reduce((a, f) => a + (f.size || 0), 0);
  let done = 0;
  for (const f of files) {
    if (await opfsCache.match(f.url)) { done += f.size || 0; onProgress && onProgress(done, total); continue; }
    const res = await fetch(f.url);
    if (!res.ok) throw new Error(`모델 파일을 받지 못했어요 (${res.status})`);
    const d = await opfsCache.root();
    const n = opfsCache.name(f.url);
    try { await d.removeEntry(n + '.ok'); } catch {}
    const fh = await d.getFileHandle(n, { create: true });
    const reader = res.body.getReader();
    let at = 0;
    const useSync = 'createSyncAccessHandle' in fh && typeof WorkerGlobalScope !== 'undefined';
    const h = useSync ? await fh.createSyncAccessHandle() : null;
    const w = useSync ? null : await fh.createWritable();
    try {
      if (h) h.truncate(0);
      for (;;) {
        const { done: end, value } = await reader.read();
        if (end) break;
        if (h) h.write(value, { at }); else await w.write(value);
        at += value.byteLength;
        onProgress && onProgress(done + at, total);
      }
      if (h) h.flush();
    } finally {
      if (h) h.close(); else await w.close();
    }
    if (f.size && at !== f.size) throw new Error('모델 파일을 끝까지 받지 못했어요. 다시 시도해 주세요.');
    await d.getFileHandle(n + '.ok', { create: true });
    done += at;
  }
}

export function useOpfsCache(env) {
  if (navigator.storage && navigator.storage.getDirectory && self.FileSystemFileHandle) {
    env.useBrowserCache = false;
    env.useCustomCache = true;
    env.customCache = opfsCache;
  }
}
