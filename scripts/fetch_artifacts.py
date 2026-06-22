#!/usr/bin/env python3
"""
fetch_artifacts.py — pull the third-party material OpenFugu does NOT redistribute.

Run once after cloning:
    python scripts/fetch_artifacts.py

Fetches (each from its original, licensed source):
  1. model_iter_60.npy           — TRINITY router checkpoint (HF dataset, MIT)
  2. qwen_router_prompt_eval_cases.json — 37-case routing fixture (trinity_coordinator, MIT)
  3. Qwen/Qwen3-0.6B             — backbone (Apache-2.0), via huggingface_hub

Nothing here is committed to the repo; see NOTICE.
"""
from __future__ import annotations
import argparse, os, sys, urllib.request

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")

# router checkpoint + fixture live in the third-party HF dataset / repo
HF_DATASET = "nshkrdotcom/trinity-coordinator-adapted-qwen3-0.6b"
FIXTURE_URL = ("https://raw.githubusercontent.com/nshkrdotcom/trinity_coordinator/"
               "main/examples/fixtures/qwen_router_prompt_eval_cases.json")
NPY_CANDIDATES = [  # the released ES vector; try repo paths in order
    "logs/ckpt/models/model_iter_60.npy",
    "model_iter_60.npy",
]


def _get(url: str, dst: str):
    print(f"  -> {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"     saved {dst} ({os.path.getsize(dst)} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-backbone", action="store_true",
                    help="don't download Qwen3-0.6B (if you already have it)")
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)

    print("[1/3] routing eval fixture (trinity_coordinator, MIT)")
    _get(FIXTURE_URL, os.path.join(ART, "qwen_router_prompt_eval_cases.json"))

    print("[2/3] TRINITY router checkpoint model_iter_60.npy (HF dataset, MIT)")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("pip install huggingface_hub  (needed to fetch the checkpoint)")
    got = False
    for path in NPY_CANDIDATES:
        try:
            p = hf_hub_download(repo_id=HF_DATASET, filename=path, repo_type="dataset")
            dst = os.path.join(ART, "model_iter_60.npy")
            if os.path.abspath(p) != os.path.abspath(dst):
                import shutil; shutil.copy(p, dst)
            print(f"     saved {dst}")
            got = True
            break
        except Exception as e:
            print(f"     (not at {path}: {e})")
    if not got:
        print("     !! could not locate model_iter_60.npy automatically; see NOTICE for the source repo")

    if not args.skip_backbone:
        print("[3/3] backbone Qwen/Qwen3-0.6B (Apache-2.0)")
        from huggingface_hub import snapshot_download
        d = snapshot_download("Qwen/Qwen3-0.6B")
        print(f"     backbone at {d}")
        print(f"     export FUGU_MODEL={d}")
    else:
        print("[3/3] skipped backbone")

    print(f"\nDone. Artifacts in {ART}/")
    print("Set:  export FUGU_VECTOR=$PWD/artifacts/model_iter_60.npy")
    print("      export FUGU_FIXTURE=$PWD/artifacts/qwen_router_prompt_eval_cases.json")


if __name__ == "__main__":
    main()
