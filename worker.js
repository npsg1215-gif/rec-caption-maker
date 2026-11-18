// 음성 인식 워커: Whisper 모델을 브라우저 안에서 실행해 자막을 만듭니다.
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

env.allowLocalModels = false;

// 브라우저 기본 저장소(Cache)는 300MB가 넘는 파일을 못 넣는 경우가 있어서,
// 모델 파일을 브라우저의 파일 저장 공간(OPFS)에 직접 저장합니다. 한 번 받으면 다시 받지 않아요.
const opfsCache = {
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
    const h = await fh.createSyncAccessHandle();
    try {
      h.truncate(0);
      const reader = response.body.getReader();
      let at = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        h.write(value, { at });
        at += value.byteLength;
      }
      h.flush();
    } finally {
      h.close();
    }
    await d.getFileHandle(n + '.ok', { create: true });
  },
};
if (navigator.storage && navigator.storage.getDirectory && self.FileSystemFileHandle &&
    'createSyncAccessHandle' in FileSystemFileHandle.prototype) {
  env.useBrowserCache = false;
  env.useCustomCache = true;
  env.customCache = opfsCache;
}

const SR = 16000;
let asr = null;
let loadedKey = '';

async function detectDevice() {
  try {
    if (!navigator.gpu) return { device: 'wasm', f16: false };
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { device: 'wasm', f16: false };
    return { device: 'webgpu', f16: adapter.features.has('shader-f16') };
  } catch {
    return { device: 'wasm', f16: false };
  }
}

function dtypeFor(model, device, f16) {
  if (device !== 'webgpu') return 'q8';
  // fp16 인코더는 일부 내장 그래픽(Intel Iris Xe 등)에서 엉뚱한 글자를 내므로 쓰지 않습니다.
  const big = model.includes('turbo');
  return { encoder_model: big ? 'q4' : 'fp32', decoder_model_merged: 'q4' };
}

async function load(model, force) {
  const gpu = await detectDevice();
  if (force && force.device) gpu.device = force.device;
  if (force && force.f16 != null) gpu.f16 = force.f16;
  const tryLoad = async (device) => {
    const key = model + '|' + device + '|' + JSON.stringify((force && force.dtype) || '');
    if (asr && loadedKey === key) return device;
    if (asr) { try { await asr.dispose(); } catch {} asr = null; }
    self.postMessage({ type: 'loading', device });
    asr = await pipeline('automatic-speech-recognition', model, {
      device,
      dtype: (force && force.dtype) || dtypeFor(model, device, gpu.f16),
      progress_callback: (p) => self.postMessage({ type: 'progress', data: p }),
    });
    loadedKey = key;
    return device;
  };
  if (gpu.device === 'webgpu') {
    try { return await tryLoad('webgpu'); }
    catch (err) { console.warn('WebGPU 실패, CPU로 전환', err); asr = null; loadedKey = ''; }
  }
  return await tryLoad('wasm');
}

// 30초 이하 조각으로 자르되, 가장 조용한 지점에서 자릅니다 (말이 중간에 잘리지 않게).
// audio: Float32Array(-1~1) 또는 Int16Array(녹화 중에 따로 받아 둔 소리). scale로 -1~1 범위로 맞춥니다.
function split(audio, scale) {
  const frame = Math.round(SR * 0.1);
  const n = Math.floor(audio.length / frame);
  const rms = new Float32Array(n + 1);
  for (let f = 0; f < n; f++) {
    let s = 0;
    for (let i = f * frame; i < (f + 1) * frame; i++) { const v = audio[i] * scale; s += v * v; }
    rms[f] = Math.sqrt(s / frame);
  }
  const minLen = 18 * SR, maxLen = 28 * SR;
  const segs = [];
  let start = 0;
  while (audio.length - start > maxLen) {
    let bestF = -1, best = Infinity;
    const f0 = Math.floor((start + minLen) / frame), f1 = Math.floor((start + maxLen) / frame);
    for (let f = f0; f < f1; f++) if (rms[f] < best) { best = rms[f]; bestF = f; }
    const cut = bestF * frame + (frame >> 1);
    segs.push([start, cut]);
    start = cut;
  }
  segs.push([start, audio.length]);
  return segs.map(([a, b]) => {
    let peak = 0;
    const fa = Math.floor(a / frame), fb = Math.floor(b / frame);
    for (let f = fa; f < fb; f++) peak = Math.max(peak, rms[f]);
    // 조각 앞부분의 무음은 잘라내요. Whisper는 앞이 조용하면 첫 자막을 0초로 잡아 버려서 시간이 어긋나요.
    const th = Math.max(0.004, peak * 0.06);
    let f = fa;
    while (f < fb && rms[f] < th) f++;
    const start = Math.max(a, (f - 3) * frame); // 말 시작 0.3초 전부터
    return { a: start < b ? start : a, b, peak };
  });
}

// Whisper가 자주 덧붙이는 군더더기 정리: 앞의 '-' 같은 기호, 말소리가 없을 때 지어내는 단골 문장
const PHANTOM = /^(시청\s*해\s*주셔서\s*감사합니다|구독과\s*좋아요(\s*부탁드립니다)?|구독\s*좋아요\s*알림\s*설정(\s*부탁드립니다)?|MBC\s*뉴스\s*\S+입니다|자막\s*(제공|by)\s*.*|한글\s*자막\s*.*)[.!]?$/;
function cleanText(t) {
  let s = (t || '').replace(/\s+/g, ' ').trim();
  s = s.replace(/^[-–—~"'“”‘’·•\s]+/, '').trim();
  if (!s || /^[\p{P}\p{S}\s]+$/u.test(s)) return '';
  if (PHANTOM.test(s)) return '';
  return s;
}

// 모델 미리 준비(녹화하는 동안 등)와 자막 만들기를 차례대로 처리합니다.
let queue = Promise.resolve();
self.onmessage = (e) => { queue = queue.then(() => handle(e.data)); };

async function handle(msg) {
  if (msg.type === 'preload') {
    try {
      const device = await load(msg.model, msg.force);
      self.postMessage({ type: 'preloaded', device, model: msg.model });
    } catch (err) {
      self.postMessage({ type: 'preload-error', message: String((err && err.message) || err) });
    }
    return;
  }
  // 실시간 자막: 짧은 조각(30초 미만) 하나를 바로 받아 적어 돌려줘요. 시간 정보 없이 글자만 (더 빨라요).
  if (msg.type === 'live') {
    try {
      const device = await load(msg.model, msg.force);
      const t0 = performance.now();
      const opts = { task: 'transcribe' };
      if (msg.language) opts.language = msg.language;
      const out = await asr(msg.audio, opts);
      self.postMessage({ type: 'live-result', id: msg.id, kind: msg.kind, text: cleanText(out.text), ms: performance.now() - t0, device });
    } catch (err) {
      self.postMessage({ type: 'live-error', id: msg.id, message: String((err && err.message) || err) });
    }
    return;
  }
  const { model, language, force } = msg;
  const audio = msg.pcm16 || msg.audio;
  const scale = msg.pcm16 ? 1 / 32768 : 1;
  try {
    const device = await load(model, force);
    self.postMessage({ type: 'ready', device });

    const segs = split(audio, scale);
    const total = audio.length / SR;
    for (let i = 0; i < segs.length; i++) {
      const { a, b, peak } = segs[i];
      const off = a / SR, len = (b - a) / SR;
      const cues = [];
      // 거의 무음인 구간은 건너뜁니다 (Whisper가 없는 말을 지어내는 것을 막음).
      if (len >= 0.5 && peak > 0.004) {
        const opts = { task: 'transcribe', return_timestamps: true };
        if (language) opts.language = language;
        const seg = new Float32Array(b - a);
        for (let k = 0; k < seg.length; k++) seg[k] = audio[a + k] * scale;
        const out = await asr(seg, opts);
        const chunks = out.chunks && out.chunks.length ? out.chunks : [{ timestamp: [0, len], text: out.text || '' }];
        for (const c of chunks) {
          const text = cleanText(c.text);
          if (!text) continue;
          let [s, t] = c.timestamp || [0, len];
          s = Math.min(Math.max(s ?? 0, 0), len);
          t = Math.min(t ?? len, len);
          if (t <= s) t = Math.min(s + 2, len);
          cues.push({ start: off + s, end: off + t, text });
        }
      }
      self.postMessage({ type: 'segment', cues, done: b / SR, total, index: i + 1, count: segs.length });
    }
    self.postMessage({ type: 'done' });
  } catch (err) {
    self.postMessage({ type: 'error', message: String((err && err.message) || err) });
  }
}
