# recursiveMASWebLLM — the model builder

Compiles a **latent-transfer-capable WebLLM model** for the
[RecursiveMAS Playground](https://github.com/vishalmysore/recursiveMAS) app — one whose
compiled WebGPU graph exposes its **last-layer hidden states**, which is what a faithful
[RecursiveMAS](https://recursivemas.github.io) needs (and what stock WebLLM/ONNX models
don't give you).

This repo is the **build pipeline only**. Its outputs (the `.wasm` model library, the
quantized weights, and the trained RecursiveLink) are published to a GitHub **Release**
and **Hugging Face**, then consumed by the app via its `CUSTOM_MODELS` config.

```
recursiveMASWebLLM  (this repo)  ──build──▶  *.wasm + weights + recursivelink.json
                                                   │ host on Release + Hugging Face
                                                   ▼
recursiveMAS  (the app)  ──CUSTOM_MODELS─▶  loads the model by URL in the browser
```

## Why this is possible on the WebLLM stack

Unlike a sealed `.onnx`, a WebLLM model is **compiled by you** from an editable MLC-LLM
(TVM Relax) definition. Two facts make latent transfer feasible:

1. **MLC already feeds embeddings, not token ids, into `prefill`/`decode`** (`get_default_spec`
   shows `input_embed: Tensor([1, seq_len, hidden])`) — so the "inject a latent vector" half exists.
2. **Exposing hidden states is a one-function edit** — `prefill` computes `hidden` then
   `lm_head`; we add a sibling that returns `hidden` directly. See [`expose_hidden.md`](expose_hidden.md)
   (human diff) / [`expose_hidden.py`](expose_hidden.py) (automated patcher).

## 🤖 Build it on GitHub Actions (no local Linux/GPU needed)

Key fact: **`mlc_llm compile` is code generation, not GPU execution** — it emits the WebGPU
wasm on a plain CPU runner. `convert_weight`/`gen_config` also run on CPU for small models.

Run [`.github/workflows/build-model.yml`](.github/workflows/build-model.yml) from the
**Actions** tab → *Run workflow* (inputs: base model, arch, quant, `train_link`, `upload_hf`).
It: installs MLC nightly (CPU) + emscripten → patches the model def (`expose_hidden.py`) →
`convert_weight` + `gen_config` + `compile --device webgpu` → uploads the `.wasm` + weights
as an artifact **and** attaches the `.wasm` to a Release.

**Host the outputs so the app can load them by URL:**
- **weights** (many shard files) → **Hugging Face**: set repo **secret** `HF_TOKEN` +
  **variable** `HF_REPO` (e.g. `vishalmysore/RecursiveMAS-0.5B-MLC`), tick `upload_hf`.
- **wasm** (single file) → the Release asset URL.

Then in the app's `main.js`, fill `CUSTOM_MODELS`:
```js
{
  model:    'https://huggingface.co/vishalmysore/RecursiveMAS-0.5B-MLC',
  model_id: 'recursivemas-0.5b',
  model_lib:'https://github.com/vishalmysore/recursiveMASWebLLM/releases/download/model-RecursiveMAS-0.5B/RecursiveMAS-0.5B-q4f16_1-webgpu.wasm',
  recursiveLink: 'https://github.com/vishalmysore/recursiveMASWebLLM/releases/download/model-RecursiveMAS-0.5B/recursivelink.json',
  vram_required_MB: 900, label: 'RecursiveMAS 0.5B', size: '~0.5 GB · custom', exposesLatent: true,
}
```

## Build locally instead (Linux / WSL2)

```bash
pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu
# (use the -cu123 wheels if you have CUDA)
source path/to/emsdk/emsdk_env.sh        # emscripten for the WebGPU target
python expose_hidden.py --arch qwen2     # patch the installed mlc_llm model def
./build.sh                               # convert_weight + gen_config + compile
python train_recursivelink.py --model Qwen/Qwen2.5-0.5B-Instruct   # optional, needs a GPU to be quick
```

## Files

| file | what |
|---|---|
| `.github/workflows/build-model.yml` | the CI pipeline (CPU Ubuntu) |
| `expose_hidden.md` | human-readable diff: add `get_last_hidden` to the MLC model def |
| `expose_hidden.py` | automated, idempotent patcher (used by CI) |
| `build.sh` | `convert_weight` → `gen_config` → `compile --device webgpu` |
| `train_recursivelink.py` | offline PyTorch training of W₁/W₂/W₃ → `recursivelink.json` |

## Honest limits

- ⚠️ **Scaffolds, not artifacts compiled here.** Written against the real `mlc_llm` source,
  but the compile/training must run on a real toolchain. Re-check the patch anchors against
  the `mlc_llm` nightly you install (the model-definition API drifts).
- **No GPU on free runners** → `train_link` is CPU-slow (opt-in); the compiled model is usable
  without a trained link (the link only affects latent-loop quality).
- **Small models only** (14 GB disk / 6 h per job). 0.5–1.5B fine; 7B no.
- **Version pin:** the `.wasm` must match the app's `@mlc-ai/web-llm` version (currently `0.2.79`).
- The remaining research piece (calling `get_last_hidden` from JS and looping hidden states
  through the link) lives in the app's `recursive-link.js` — not this repo.

## License

MIT
