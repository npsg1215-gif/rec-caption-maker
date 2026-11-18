// AI 요약 워커: 작은 공개 언어 모델을 브라우저 안에서 돌려 대본을 요약해요.
// 대본은 인터넷으로 보내지 않고 내 컴퓨터 안에서만 처리해요.
import { AutoTokenizer, AutoModelForCausalLM, TextStreamer, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
import { useOpfsCache, prefetchFiles } from './cache.js';

env.allowLocalModels = false;
useOpfsCache(env); // 한 번 받은 모델은 다시 받지 않아요

let tok = null, model = null, loadedKey = '';

// 모델 파일을 먼저 끝까지 받아 저장해요 (크기 확인). 큰 파일을 크기 모른 채 불러오면 메모리가 모자라 멈출 수 있어서예요.
async function prefetch(id, dtype) {
  let siblings;
  try { siblings = (await (await fetch(`https://huggingface.co/api/models/${id}?blobs=true`)).json()).siblings || []; }
  catch { return; } // 인터넷이 안 되면 저장된 파일로 바로 시도해요
  const want = new Set(['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', `onnx/model_${dtype}.onnx`, `onnx/model_${dtype}.onnx_data`]);
  const files = siblings.filter((s) => want.has(s.rfilename)).map((s) => ({ url: `https://huggingface.co/${id}/resolve/main/${s.rfilename}`, size: s.size }));
  let last = 0;
  await prefetchFiles(files, (loaded, total) => {
    const now = performance.now();
    if (now - last > 300 || loaded === total) { last = now; self.postMessage({ type: 'download', loaded, total }); }
  });
}

async function load(id, dtype) {
  const key = id + '|' + dtype;
  if (model && loadedKey === key) return;
  if (model) { try { await model.dispose(); } catch {} model = null; }
  await prefetch(id, dtype);
  self.postMessage({ type: 'loading' });
  const progress_callback = (p) => self.postMessage({ type: 'progress', data: p });
  tok = await AutoTokenizer.from_pretrained(id, { progress_callback });
  model = await AutoModelForCausalLM.from_pretrained(id, { device: 'webgpu', dtype, progress_callback });
  loadedKey = key;
}

const SYSTEM = '너는 강의·회의 대본을 정리하는 한국어 비서야. 대본에 있는 내용만 쓰고, 대본에 없는 내용은 절대 지어내지 마. 답은 한국어로만, 짧고 정확하게 써.';

// 한 번 생성 (글자가 나오는 대로 화면에 보내요)
async function generate(user, maxNew, stream) {
  const messages = [{ role: 'system', content: SYSTEM }, { role: 'user', content: user }];
  const inputs = tok.apply_chat_template(messages, { add_generation_prompt: true, return_dict: true, enable_thinking: false });
  const nIn = inputs.input_ids.dims[1];
  const streamer = stream ? new TextStreamer(tok, { skip_prompt: true, skip_special_tokens: true, callback_function: (t) => self.postMessage({ type: 'token', text: t }) }) : undefined;
  const out = await model.generate({ ...inputs, max_new_tokens: maxNew, do_sample: false, repetition_penalty: 1.1, streamer });
  return tok.decode(out.slice(null, [nIn, null]), { skip_special_tokens: true }).trim();
}

// 작은 AI는 긴 글을 정리하다 보면 내용을 지어내거나 예시를 베끼기 쉬워서,
// 핵심 문장과 할 일은 화면 쪽에서 대본 문장을 그대로 골라 보여 주고,
// AI에게는 그 핵심 문장들을 읽고 '한 줄 요약' 한 문장만 쓰게 해요. (입력이 짧아서 긴 대본도 빨라요)
async function summarize(lines) {
  self.postMessage({ type: 'stage', step: 1, steps: 1, label: '한 줄 요약을 쓰는 중' });
  // 작은 AI는 형식·빈칸·예시를 그대로 베끼는 경우가 많아서, 평범한 질문 하나로 물어봐요
  const prompt = lines.join('\n') + '\n\n질문: 위 내용은 무엇을 설명하고 있나요? 한국어 한 문장으로 짧게 답해 주세요.';
  return generate(prompt, 60, true);
}

let queue = Promise.resolve();
self.onmessage = (e) => { queue = queue.then(() => handle(e.data)); };
async function handle(msg) {
  if (msg.type !== 'summarize') return;
  try {
    await load(msg.model, msg.dtype);
    self.postMessage({ type: 'ready' });
    const t0 = performance.now();
    const text = await summarize(msg.lines);
    self.postMessage({ type: 'result', id: msg.id, text, ms: performance.now() - t0 });
  } catch (err) {
    self.postMessage({ type: 'error', id: msg.id, message: String((err && err.message) || err) });
  }
}
