#!/usr/bin/env python
"""
Programmatically apply the "expose last-layer hidden states" edit to the INSTALLED
mlc_llm model definition (so CI can do it without manual editing). Idempotent.

This is the automated form of expose_hidden.md. It inserts two methods
(get_last_hidden / decode_last_hidden) and their get_default_spec entries into the
target architecture's model file.

Usage:  python expose_hidden.py --arch qwen2
Run it AFTER `pip install mlc-llm-nightly...` and BEFORE `mlc_llm compile`.

The compiled WASM will export these functions ONLY if the export list is manually
edited in mlc-llm source BEFORE building (patch_source_before_compile.py does this).

NOTE: string-anchor based — fragile across mlc_llm versions. If anchors don't match,
open the file (path is printed) and apply expose_hidden.md by hand.
"""
import argparse, importlib, os, sys

METHODS = '''
    def get_last_hidden(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        """RecursiveMAS: full last-layer hidden states (latent thoughts), no LM head."""
        op_ext.configure()
        hidden_states = self.model(input_embed, paged_kv_cache)
        if hidden_states.dtype != self.dtype:
            hidden_states = hidden_states.astype(self.dtype)
        return hidden_states, paged_kv_cache

    def decode_last_hidden(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        """RecursiveMAS: single-step latent recurrence variant."""
        op_ext.configure()
        hidden_states = self.model(input_embed, paged_kv_cache)
        if hidden_states.dtype != self.dtype:
            hidden_states = hidden_states.astype(self.dtype)
        return hidden_states, paged_kv_cache

'''

SPEC = '''        "get_last_hidden": {
            "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
            "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
            "$": {"param_mode": "packed", "effect_mode": "none"},
        },
        "decode_last_hidden": {
            "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
            "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
            "$": {"param_mode": "packed", "effect_mode": "none"},
        },
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="qwen2", help="mlc_llm model arch (qwen2, llama, gemma2, ...)")
    args = ap.parse_args()

    mod_name = f"mlc_llm.model.{args.arch}.{args.arch}_model"
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        sys.exit(f"Could not import {mod_name}: {e}")
    path = mod.__file__
    src = open(path, encoding="utf-8").read()
    print(f"Patching {path}")

    if "def get_last_hidden" in src:
        print("Already patched — skipping methods.");
    else:
        anchor = "    def prefill(self, input_embed"
        if anchor not in src:
            sys.exit(f"Anchor '{anchor}' not found — apply expose_hidden.md by hand to {path}")
        src = src.replace(anchor, METHODS + anchor, 1)

    if '"get_last_hidden":' in src:
        print("Spec already has get_last_hidden — skipping spec.")
    else:
        spec_anchor = '        "create_paged_kv_cache": {'
        if spec_anchor not in src:
            sys.exit(f"Spec anchor not found — apply expose_hidden.md by hand to {path}")
        src = src.replace(spec_anchor, SPEC + spec_anchor, 1)

    open(path, "w", encoding="utf-8").write(src)
    # sanity: re-import to ensure it still parses
    importlib.reload(mod)
    print("✓ Patched and re-imports cleanly. Exposed: get_last_hidden, decode_last_hidden")


if __name__ == "__main__":
    main()
