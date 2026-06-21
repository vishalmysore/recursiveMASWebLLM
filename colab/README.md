# Build the model in Google Colab

Interactive alternative to the GitHub Actions build — useful for fast iteration and for the
optional GPU step (training the RecursiveLink).

**Open it directly in Colab:**

https://colab.research.google.com/github/vishalmysore/recursiveMASWebLLM/blob/main/colab/build_recursivemas_colab.ipynb

(or in Colab: File → Open notebook → GitHub → `vishalmysore/recursiveMASWebLLM`)

## Steps
1. Runtime → Change runtime type → **T4 GPU** (free).
2. Run the cells top to bottom. The TVM build (~25–35 min) runs once and stays built for the
   session, so you can re-run the **compile** cell freely while iterating.
3. Paste a Hugging Face **Write** token at the login cell to upload the weights.

## Why this works where the wheels don't
The current MLC nightly wheels are ABI-broken (mid `apache-tvm-ffi` migration). This notebook
builds TVM + mlc-llm **from source** pinned to **`v0.19.0`** (the last pre-migration release),
with the LLVM/Polly and cargo-sparse-index fixes baked in — the same recipe as
[`.github/workflows/build-source.yml`](../.github/workflows/build-source.yml), just interactive.

## Output → app
- Weights → Hugging Face (`vishalmysore/RecursiveMAS-0.5B-MLC`)
- `.wasm` → download from `dist-model/libs/` and attach to a GitHub Release (or push to HF)

Then point the app at them via the `CUSTOM_MODELS` block in the recursiveMAS app's `main.js`.
You can't test WebGPU in Colab — testing happens in the browser app.
