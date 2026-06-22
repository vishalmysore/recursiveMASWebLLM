#!/usr/bin/env python
"""
Patch mlc-chat-config.json to declare get_last_hidden and decode_last_hidden
as exported functions so mlc_llm compile emits them as WASM entry points.

Run AFTER mlc_llm gen_config and BEFORE mlc_llm compile.
"""
import json
import sys
from pathlib import Path


def patch_config(config_path: str):
    """Add latent functions to exported_functions list in config."""
    path = Path(config_path)
    if not path.exists():
        sys.exit(f"Config not found: {config_path}")
    
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    
    print(f"Patching {config_path}")
    
    # Ensure exported_functions list exists
    if "exported_functions" not in config:
        config["exported_functions"] = []
    
    exported = config["exported_functions"]
    
    # Add latent functions if not already there
    new_funcs = ["get_last_hidden", "decode_last_hidden"]
    for func in new_funcs:
        if func not in exported:
            exported.append(func)
            print(f"  Added: {func}")
        else:
            print(f"  Already present: {func}")
    
    # Write back
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Config patched. Exported functions: {exported}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python patch_config_exports.py <path/to/mlc-chat-config.json>")
    patch_config(sys.argv[1])
