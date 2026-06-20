# Patch: expose last-layer hidden states in the MLC model definition

This adds a `get_last_hidden` function to the model head so the compiled WebGPU library
returns the **last-layer hidden states** (the "latent thoughts") instead of only logits.
It mirrors the real `prefill` in `mlc_llm/model/qwen2/qwen2_model.py` — minus the LM head
and the `index_last_token` reduction, so you get the full `[1, seq_len, hidden]` tensor.

> Verify against your installed source:
> `python -c "import mlc_llm, os; print(os.path.dirname(mlc_llm.__file__))"`
> then edit `…/model/qwen2/qwen2_model.py`. The API drifts between nightlies — match the
> surrounding code (decorators, `op_ext.configure()`, the cache object name).

## 1) Add the method to `QWen2LMHeadModel`

Insert next to `prefill` / `decode`:

```python
def get_last_hidden(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
    """Return the full last-layer hidden states (latent thoughts), no LM head.
    Shape: [1, seq_len, hidden_size]. Used for RecursiveMAS latent transfer."""
    op_ext.configure()
    hidden_states = self.model(input_embed, paged_kv_cache)   # same body prefill uses
    if hidden_states.dtype != self.dtype:
        hidden_states = hidden_states.astype(self.dtype)
    return hidden_states, paged_kv_cache

def decode_last_hidden(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
    """Single-step variant for latent recurrence. Shape: [1, 1, hidden_size]."""
    op_ext.configure()
    hidden_states = self.model(input_embed, paged_kv_cache)
    if hidden_states.dtype != self.dtype:
        hidden_states = hidden_states.astype(self.dtype)
    return hidden_states, paged_kv_cache
```

## 2) Register them in `get_default_spec`

Add these two entries to the `mod_spec` dict (same shapes as `prefill`/`decode`):

```python
        "get_last_hidden": {
            "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
            "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
            "$": {"param_mode": "packed", "effect_mode": "none"},
        },
        "decode_last_hidden": {
            "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
            "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
            "$": {"param_mode": "packed", "effect_mode": "none"},
        },
```

## 3) (optional) Expose the input-embedding matrix

For the **outer link** you may want to project into the next agent's *input-embedding*
space. `embed(input_ids)` already exists; that's enough to embed real tokens. The latent
you feed back into `prefill`/`get_last_hidden` is just a `[1, seq_len, hidden]` tensor, so
no extra export is required to inject a vector — `prefill` already accepts `input_embed`.

## What you get after compiling

The compiled module exposes callable functions: `embed`, `prefill`, `decode`,
`get_last_hidden`, `decode_last_hidden`, `create_paged_kv_cache`, … You call them from JS
via the TVM runtime (see `../recursive-link.js` → `LOW_LEVEL_NOTES`). The high-level
`engine.chat.completions` path ignores the new functions — that's expected.

## For other architectures

The same edit applies to `llama_model.py`, `gemma_model.py`, etc. — find the head class's
`prefill` + `get_default_spec` and add the analogous `get_last_hidden`. The body call is
always `self.model(input_embed, paged_kv_cache)`; only names differ.
