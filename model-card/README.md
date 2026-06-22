---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen2.5-0.5B-Instruct
pipeline_tag: text-generation
library_name: mlc-llm
tags:
- webllm
- webgpu
- mlc-llm
- browser
- recursive-multi-agent
- qwen2
- latent
---

# RecursiveMAS-0.5B-MLC

A **WebGPU / WebLLM** build of [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct),
compiled from source with [MLC-LLM](https://github.com/mlc-ai/mlc-llm) (quantization `q4f16_1`) and
**patched to expose its last-layer hidden states**. It runs entirely in the browser — no server, no API key —
and is built for latent multi-agent experiments inspired by the paper
[*Recursive Multi-Agent Systems*](https://recursivemas.github.io).

> Build pipeline & sources: **https://github.com/vishalmysore/recursiveMASWebLLM**

## What's special

Standard WebLLM chat models only expose `input_ids → logits`. This build adds two functions to the
compiled module so latent state can be read/looped between agents (the RecursiveMAS "RecursiveLink" idea):

- `get_last_hidden(input_embed, kv_cache) → (hidden_states, kv_cache)`
- `decode_last_hidden(input_embed, kv_cache) → (hidden_states, kv_cache)`

It also works as an ordinary chat backbone.

## Files

| File | What |
|------|------|
| `params_shard_*.bin`, `ndarray-cache.json` | `q4f16_1` quantized weights |
| `mlc-chat-config.json` | MLC chat config |
| `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json` | tokenizer |
| `libs/RecursiveMAS-0.5B-q4f16_1-webgpu.wasm` | the WebGPU model library |

## Usage (WebLLM, in the browser)

```js
import * as webllm from "@mlc-ai/web-llm";

const appConfig = {
  model_list: [{
    model:     "https://huggingface.co/VishalMysore/RecursiveMAS-0.5B-MLC",
    model_id:  "recursivemas-0.5b",
    model_lib: "https://huggingface.co/VishalMysore/RecursiveMAS-0.5B-MLC/resolve/main/libs/RecursiveMAS-0.5B-q4f16_1-webgpu.wasm",
  }],
};
const engine = await webllm.CreateMLCEngine("recursivemas-0.5b", { appConfig });
const r = await engine.chat.completions.create({ messages: [{ role: "user", content: "Hello!" }] });
console.log(r.choices[0].message.content);
```

## How it was built

Built **from source** against `mlc-llm` **v0.19.0** (the last release before the `apache-tvm-ffi`
migration), TVM compiled with LLVM, model definition patched via
[`expose_hidden.py`](https://github.com/vishalmysore/recursiveMASWebLLM/blob/main/expose_hidden.py),
then `mlc_llm compile --device webgpu`. The full, reproducible pipeline (and a Colab notebook) is in the
[recursiveMASWebLLM](https://github.com/vishalmysore/recursiveMASWebLLM) repo.

## ⚠️ Version compatibility

The `.wasm` model library was compiled with **mlc-llm v0.19.0**. WebGPU model libraries are tied to the
runtime version, so load it with a **compatible `@mlc-ai/web-llm`** build — if you hit a "model lib
version" error, pin `@mlc-ai/web-llm` to the version matching mlc-llm v0.19.0 (or recompile the `.wasm`
against your runtime's version).

## License

Apache-2.0, inheriting the base model [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
This is a research/educational artifact.
