# WASM Export Issue: Why get_last_hidden Isn't in the Compiled Binary

## The Problem

When `mlc_llm compile --device webgpu` runs, it generates C++ code and compiles it to WebAssembly using emscripten. However, **only certain functions are exported as WASM entry points**. Currently, only `prefill` and `decode` are exported.

Even though `expose_hidden.py` successfully adds `get_last_hidden` and `decode_last_hidden` as Python methods, they don't appear in the compiled WASM binary because:

1. The Python specs define the method signature for internal use ✓
2. mlc-llm generates C++ from that Python definition ✓  
3. **The C++ compiler emscripten) is NOT told to export these functions to WASM** ✗

## Why patch_config_exports.py Doesn't Work

The `mlc-chat-config.json` file is NOT read by the C++ compiler. It's used by the JavaScript runtime to configure parameters. Adding `exported_functions` to that JSON does nothing because emscripten's export list is determined by:

1. **EXPORTED_FUNCTIONS in emscripten link flags** (static C++ symbols)
2. **TVM's packed function registry** (dynamic runtime functions)

Neither of these is controlled by the JSON config.

## The Real Fix Required

To properly export `get_last_hidden` and `decode_last_hidden`, one of these must be done:

### Option A: Patch mlc-llm Source Before Compile (Most Correct)

Modify mlc-llm's C++ model codegen or build system to include the latent functions in the export list.

**Where to patch:**
- `mlc-llm/cpp/relax_vm.cc` or the model generation code that creates exported function registry
- `mlc-llm/web/module.ts` (TypeScript wrapper that defines WASM exports)
- The emscripten build flags for `-sEXPORTED_FUNCTIONS`

**Implementation:**
```bash
# After: git clone mlc-llm v0.19.0
# Before: mlc_llm compile

# Find where functions are exported and add:
sed -i 's/"prefill"/"prefill", "get_last_hidden", "decode_last_hidden"/' mlc-llm/web/module.ts
# Or equivalent for C++ code
```

This is version-specific and requires understanding mlc-llm v0.19.0's export mechanism.

### Option B: Post-Process WASM Binary

After compilation, use a tool like `wasm-opt` or `wasm-cli` to:
1. Copy the `get_last_hidden` function bytecode in the binary
2. Add it to the WASM export section
3. Re-pack and re-sign if needed

Less reliable and may break the module.

### Option C: JavaScript Wrapper Workaround

Modify [recursiveMASDemo/latent-core.js](latent-core.js#L1) to:
1. Detect if the WASM exports the latent functions
2. If not, call `prefill` instead and extract hidden states from intermediate buffers
3. Or use a fallback that calls Python backend

Not ideal for browser-only usage.

## Current Status

- ✓ `expose_hidden.py` successfully patches the Python model class
- ✓ mlc-llm successfully compiles the patched model to C++
- ✗ **The C++ compiler does not export latent functions to WASM**
- ✓ `mlc-chat-config.json` correctly lists them (but compiler ignores this)

## Next Steps

1. **Investigate mlc-llm v0.19.0 source** to find where WASM exports are defined
2. **Add a patch script** that modifies mlc-llm source before `mlc_llm compile` runs
3. **Test locally** to verify the compiled WASM has the exported symbols
4. **Update workflow** to call the patch script

For now, the workaround is to use the text-based latent fall

back in the demo (compress/decompress).
