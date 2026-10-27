#!/usr/bin/env python3
"""Download the clinical-note starting checkpoint (LoRA adapter + pointer head, Apache-2.0)
into init/clinical-0.8b, for `train --init_from init/clinical-0.8b`.

    python scripts/fetch_init.py
"""
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "junma/MedJev-Qwen3.5-0.8B"
DEST = Path(__file__).resolve().parent.parent / "init" / "clinical-0.8b"

path = snapshot_download(REPO, local_dir=str(DEST),
                         allow_patterns=["*.json", "*.safetensors", "*.pt", "*.jinja", "*.txt"])
print(f"ready: {path}")
