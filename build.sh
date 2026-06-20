#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Build a latent-transfer-capable WebLLM model (Qwen2.5-0.5B-Instruct, q4f16_1).
#
# RUN ON LINUX OR WSL2 (the WebGPU target needs emscripten; not Windows-native).
# This is a scaffold — re-check flags against your installed mlc_llm nightly.
#
# Prereqs (see README.md): python 3.11, git, an emsdk install, and:
#   pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-cpu mlc-ai-nightly-cpu
#   (use the -cu123 wheels instead of -cpu if you have CUDA and want GPU convert)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_HF="Qwen/Qwen2.5-0.5B-Instruct"     # base model on HF
ARCH="qwen2"                               # mlc_llm model arch (patched in expose_hidden.md)
QUANT="q4f16_1"                            # quantization
NAME="RecursiveMAS-0.5B"                   # output name
OUT="./dist-model"                         # output root
WEIGHTS="$OUT/$NAME-MLC"                   # converted weights dir
LIBS="$OUT/libs"                           # compiled wasm dir
PREFILL_CHUNK=1024

mkdir -p "$WEIGHTS" "$LIBS"

echo "==> 0. Sanity: emscripten + mlc_llm present"
command -v emcc >/dev/null || { echo "emcc not found — 'source path/to/emsdk_env.sh' first"; exit 1; }
python -c "import mlc_llm; print('mlc_llm at', mlc_llm.__file__)"
echo "    >>> Make sure you applied expose_hidden.md to this mlc_llm's $ARCH model file. <<<"

echo "==> 1. Download base weights"
HF_DIR="./hf/$NAME"
hf download "$MODEL_HF" --local-dir "$HF_DIR"   # huggingface_hub 1.x CLI

echo "==> 2. convert_weight  (HF -> MLC params)"
mlc_llm convert_weight "$HF_DIR" \
  --quantization "$QUANT" \
  --model-type "$ARCH" \
  -o "$WEIGHTS"

echo "==> 3. gen_config  (chat template, tokenizer, model metadata)"
mlc_llm gen_config "$HF_DIR" \
  --quantization "$QUANT" \
  --model-type "$ARCH" \
  --conv-template qwen2 \
  --prefill-chunk-size "$PREFILL_CHUNK" \
  -o "$WEIGHTS"

echo "==> 4. compile  ->  WebGPU wasm (includes your get_last_hidden function)"
mlc_llm compile "$WEIGHTS/mlc-chat-config.json" \
  --device webgpu \
  -o "$LIBS/$NAME-$QUANT-webgpu.wasm"

cat <<EOF

==> DONE
   Weights : $WEIGHTS
   Wasm    : $LIBS/$NAME-$QUANT-webgpu.wasm

Next:
  • Upload $WEIGHTS to a Hugging Face model repo (e.g. you/$NAME-MLC).
  • Host $LIBS/$NAME-$QUANT-webgpu.wasm on any static CDN (GitHub Pages / HF).
  • Put both URLs into CUSTOM_MODELS in ../main.js (exposesLatent: true).
  • Train the link:  python train_recursivelink.py --model $MODEL_HF
  • Verify the new function is in the wasm:
      the compile log lists exported funcs; expect 'get_last_hidden' + 'decode_last_hidden'.
EOF
