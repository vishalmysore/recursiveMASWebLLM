#!/usr/bin/env python
"""
Patch mlc-llm source BEFORE compilation to export latent functions.

The mlc_llm compile step uses TVM codegen which decides which functions to emit
as WASM exports. By default, only standard functions (prefill, decode) are exported.

This script patches the mlc-llm source to add get_last_hidden and decode_last_hidden
to the export list so they appear in the compiled WASM binary.

Run AFTER cloning mlc-llm but BEFORE mlc_llm compile.
"""
import os
import sys
from pathlib import Path


def patch_export_list(mlc_llm_path: str):
    """Patch mlc-llm's web runtime to export latent functions."""
    
    # Path to the web runtime that defines exported functions
    runtime_h = Path(mlc_llm_path) / "web" / "mlc_web_runtime.h"
    
    if not runtime_h.exists():
        print(f"Warning: {runtime_h} not found, skipping export list patch.")
        return
    
    src = runtime_h.read_text(encoding="utf-8")
    print(f"Patching {runtime_h}")
    
    # Look for the function registry or export list and add latent functions
    # The export list is typically in module.ts or in the TVM runtime config
    # For emscripten builds, it's in the exported_functions list
    
    # For mlc-llm, the actual export happens via the packed func mechanism
    # We need to ensure get_last_hidden and decode_last_hidden are registered
    # in the RuntimeModule or ModelRuntime
    
    if "get_last_hidden" in src:
        print("get_last_hidden already in export list")
        return
    
    # The actual mechanism depends on mlc-llm version, but typically:
    # Look for where functions are registered and add our latent ones
    # This is a placeholder - the real fix is version-specific
    print("Export list patch: check mlc-llm version and apply manually if needed")
    print("The WASM export list is typically defined in:")
    print("  - mlc-llm/web/module.ts (TypeScript)")
    print("  - mlc-llm/web/mlc_web_runtime.cc (C++)")
    print("Look for 'prefill' or 'decode' in those files and add 'get_last_hidden' nearby.")


def patch_tvmjs_export(mlc_llm_path: str):
    """Patch the TVM.js module exports to include latent functions."""
    
    # The actual WASM exports are controlled by emscripten's EXPORTED_FUNCTIONS
    # Or by TVM's packed func registry
    # This needs to be set at link time or in the runtime initialization
    
    # For a proper fix, we'd need to:
    # 1. Add the functions to the TVM packed func registry in mlc_llm_api.cc
    # 2. Or patch the emscripten link flags to include these in -sEXPORTED_FUNCTIONS
    
    print("TVMjs export patch: version-specific, apply in build system")
    print("For emscripten: add to -sEXPORTED_FUNCTIONS in CMakeLists.txt")
    print("For TVM runtime: register in TVM_REGISTER_PACKED_FUNC")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Note: This script needs mlc-llm source path")
        print("Usage: python patch_mlc_exports.py /path/to/mlc-llm")
        print("")
        print("The actual fix for WASM exports requires:")
        print("1. Patching expose_hidden.py to use TVM export decorators")
        print("2. OR manually editing the mlc-llm build to include get_last_hidden")
        print("")
        print("For now, the workaround:")
        print("- Modify .github/workflows/build-source.yml to patch mlc-llm/web before compile")
        print("- OR compile locally and verify WASM exports with:")
        print("  strings dist-model/libs/*.wasm | grep -E 'get_last_hidden|decode_last_hidden'")
        sys.exit(1)
    
    mlc_path = sys.argv[1]
    patch_export_list(mlc_path)
    patch_tvmjs_export(mlc_path)
