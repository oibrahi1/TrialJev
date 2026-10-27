#!/usr/bin/env python3
"""Check that the current Python environment can train the trialmatch model.

    python check_env.py                 # full check, incl. loading Qwen3.5-0.8B on the freest GPU
    python check_env.py --no-model      # packages + CUDA only
    python check_env.py --gpu 2         # run the model test on a specific GPU

Prints PASS / WARN / FAIL per item and a summary line; paste the whole output back.
"""
import argparse
import importlib
import logging
import platform
import sys
import time
from importlib import metadata

RESULTS = []


def report(status, item, detail=""):
    RESULTS.append(status)
    print(f"[{status}] {item}" + (f" -- {detail}" if detail else ""))


def vtuple(v):
    out = []
    for part in v.split("+")[0].split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        if not num:
            break
        out.append(int(num))
    return tuple(out)


# (distribution name, import name, minimum, exclusive maximum, required?)
PACKAGES = [
    ("torch", "torch", "2.6", None, True),
    ("transformers", "transformers", "5.17", "6", True),
    ("peft", "peft", "0.21", None, True),
    ("pydantic", "pydantic", "2.9", None, True),
    ("accelerate", "accelerate", "1.0", None, True),
    ("huggingface-hub", "huggingface_hub", "0.26", None, True),
    ("numpy", "numpy", "2.0", None, True),
    ("scikit-learn", "sklearn", "1.5", None, True),
    ("flash-linear-attention", "fla", "0.5", None, True),
    ("triton", "triton", None, None, True),
    ("wandb", "wandb", None, None, False),
    ("pytest", "pytest", None, None, False),
]


def check_packages():
    print("\n== Python packages ==")
    ok = sys.version_info >= (3, 12)
    report("PASS" if ok else "FAIL", f"python {platform.python_version()}", "" if ok else "requires >=3.12")
    print(f"       executable: {sys.executable}")
    for dist, mod, lo, hi, required in PACKAGES:
        try:
            ver = metadata.version(dist)
        except metadata.PackageNotFoundError:
            ver = None
        try:
            importlib.import_module(mod)
            imported, err = True, ""
        except Exception as e:
            imported, err = False, f"{type(e).__name__}: {e}"
        miss = "FAIL" if required else "WARN"
        if ver is None and not imported:
            report(miss, dist, "not installed" + ("" if required else " (optional)"))
            continue
        if not imported:
            report("FAIL", f"{dist} {ver}", f"installed but import fails: {err}")
            continue
        problems = []
        if ver and lo and vtuple(ver) < vtuple(lo):
            problems.append(f"need >={lo}")
        if ver and hi and vtuple(ver) >= vtuple(hi):
            problems.append(f"need <{hi}")
        report("FAIL" if problems else "PASS", f"{dist} {ver}", ", ".join(problems))
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        report("PASS", "fla gated_delta_rule kernel importable")
    except Exception as e:
        report("FAIL", "fla gated_delta_rule kernel", f"{type(e).__name__}: {e}")
    try:
        import trialmatch  # noqa: F401
        report("PASS", "trialmatch package importable", trialmatch.__file__)
    except Exception:
        report("WARN", "trialmatch package not installed yet", "fine before `pip install -e . --no-deps`")


def check_cuda():
    print("\n== CUDA / GPUs ==")
    try:
        import torch
    except Exception:
        report("FAIL", "torch import", "cannot check CUDA")
        return None
    report("PASS" if torch.cuda.is_available() else "FAIL", "torch.cuda.is_available()", str(torch.cuda.is_available()))
    print(f"       torch built for CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        return None
    best, best_free = None, -1
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f"       GPU {i}: {name}  sm_{cap[0]}{cap[1]}  free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
        if free > best_free:
            best, best_free = i, free
    report("PASS" if torch.cuda.is_bf16_supported() else "FAIL", "bf16 supported")
    report("PASS" if best_free >= 16 * 2**30 else "WARN", f"freest GPU is {best}",
           f"{best_free / 2**30:.1f} GiB free (needs >=16 GiB)")
    return best


def check_hub(model):
    print("\n== Hugging Face Hub ==")
    try:
        from huggingface_hub import HfApi
        for repo in (model,):
            HfApi().model_info(repo)
            report("PASS", f"reachable: {repo}")
    except Exception as e:
        report("WARN", "Hub not reachable", f"{type(e).__name__}: {e} -- pre-download models and set HF_HUB_OFFLINE=1")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def check_model(model, gpu):
    print(f"\n== Model test: {model} on GPU {gpu} ==")
    import warnings
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cap = _Capture()
    logging.getLogger("transformers").addHandler(cap)
    dev = torch.device(f"cuda:{gpu}")
    try:
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            tok = AutoTokenizer.from_pretrained(model)
            lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).model.to(dev)
            types = set(getattr(lm.config, "layer_types", None) or [])
            report("PASS" if "linear_attention" in types else "WARN", "hybrid Gated DeltaNet backbone",
                   f"layer_types={sorted(types)}")
            ids = tok("A 45-year-old man with anaplastic astrocytoma. " * 60, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                lm(input_ids=ids)
                torch.cuda.synchronize(dev)
                t0 = time.time()
                lm(input_ids=ids)
                torch.cuda.synchronize(dev)
            ms = (time.time() - t0) * 1000
            report("PASS", "forward pass", f"{ids.shape[1]} tokens in {ms:.0f} ms, peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB")
        msgs = cap.lines + [str(w.message) for w in warns]
        slow = [m for m in msgs if "fast path" in m.lower() or "flash-linear-attention" in m.lower() or "fla" in m.lower().split()]
        if slow:
            for m in slow:
                report("FAIL", "slow fallback path in use", m[:200])
        else:
            report("PASS", "no slow-path warnings (fla kernels used)")
        for m in msgs:
            if "falling back" in m.lower() and m not in slow:
                report("WARN", "optional kernel missing", m[:160])
    except Exception as e:
        report("FAIL", "model load/forward", f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    a = ap.parse_args()
    check_packages()
    best = check_cuda()
    check_hub(a.model)
    if not a.no_model and best is not None:
        check_model(a.model, a.gpu if a.gpu is not None else best)
    n_fail, n_warn = RESULTS.count("FAIL"), RESULTS.count("WARN")
    print(f"\n== SUMMARY: {RESULTS.count('PASS')} pass, {n_warn} warn, {n_fail} fail ==")
    print("READY for training" if n_fail == 0 else "NOT READY -- fix the FAIL lines above")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
