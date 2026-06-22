#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: the real end-to-end loop in one command — train a per-step TRINITY
# head, serve THAT head over a real local pool, verify a live request. Original.
"""
e2e_train_serve.py — one command: train -> serve -> verify, on a fresh head.

Composes the already-verified entrypoints (no refactor of their internals):
  stage 1  train/train_trinity_perstep.py  --out <fresh head .npy>
  stage 2  eval/serve_e2e.py  --head <that same head>  (boots serve.py over the
           real local pool, POSTs a real GSM8K question, asserts a real answer)

The head served is provably the head just trained (same generated path). Exit
code is non-zero if any stage fails; zero only when the live request returns the
expected answer through a real local worker.

  python pipeline/e2e_train_serve.py \
    --model <qwen3-0.6b dir> --vector model_iter_60.npy \
    --local-models "<llama dir>,<gemma dir>" --port 8099

  # serve+verify only, against an existing head:
  python pipeline/e2e_train_serve.py --skip-train --head trinity_perstep.npy ...
"""
from __future__ import annotations
import argparse, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
# default to the release tree layout; override via env for other layouts (e.g. a
# flat server dir where all scripts sit side by side).
TRAIN = os.environ.get("FUGU_TRAIN_SCRIPT", os.path.join(ROOT, "train", "train_trinity_perstep.py"))
SERVE_E2E = os.environ.get("FUGU_SERVE_E2E_SCRIPT", os.path.join(ROOT, "eval", "serve_e2e.py"))


def run_stage(name, cmd):
    print(f"\n========== STAGE: {name} ==========\n  {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd)
    print(f"[pipeline] stage {name} exit={p.returncode}", flush=True)
    return p.returncode
def main():
    ap = argparse.ArgumentParser(description="One-command train->serve->verify pipeline.")
    ap.add_argument("--model", required=True, help="Qwen3-0.6B router dir")
    ap.add_argument("--vector", default="model_iter_60.npy", help="base SVF+head vector (19456)")
    ap.add_argument("--local-models", required=True, metavar="CSV",
                    help="local HF worker paths (path or path@device), comma-separated")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--max-turns", type=int, default=4)
    # training knobs (forwarded to train_trinity_perstep.py)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    # escape hatch: reuse an existing head, skip training
    ap.add_argument("--skip-train", action="store_true",
                    help="skip stage 1; serve+verify the --head provided")
    ap.add_argument("--head", default=None,
                    help="with --skip-train: existing head .npy to serve")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    # decide the head path: fresh (generated) for a full run, or the provided one
    tmp = None
    if args.skip_train:
        if not args.head or not os.path.exists(args.head):
            print("[pipeline] --skip-train requires an existing --head .npy", flush=True)
            return 2
        head_path = args.head
        print(f"[pipeline] skip-train: serving existing head {head_path}", flush=True)
    else:
        tmp = tempfile.NamedTemporaryFile(prefix="fugu_head_", suffix=".npy", delete=False)
        tmp.close()
        head_path = tmp.name
        rc = run_stage("train", [
            args.python, TRAIN, "--router-model", args.model, "--vector", args.vector,
            "--n-train", str(args.n_train), "--iters", str(args.iters),
            "--max-turns", str(args.max_turns), "--sigma0", str(args.sigma0),
            "--seed", str(args.seed), "--out", head_path])
        if rc != 0:
            print("[pipeline] FAIL — training stage errored", flush=True)
            return rc
        if not (os.path.exists(head_path) and os.path.getsize(head_path) > 0):
            print(f"[pipeline] FAIL — training produced no head at {head_path}", flush=True)
            return 1
        print(f"[pipeline] trained head ready: {head_path} "
              f"({os.path.getsize(head_path)} bytes) — serving THIS head", flush=True)

    # stage 2: serve the (fresh) head over the local pool + verify a live request
    serve_cmd = [
        args.python, SERVE_E2E, "--model", args.model, "--vector", args.vector,
        "--head", head_path, "--local-models", args.local_models,
        "--port", str(args.port), "--max-turns", str(args.max_turns)]
    serve_script = os.environ.get("FUGU_SERVE_SCRIPT")
    if serve_script:                       # flat-layout override for serve.py path
        serve_cmd += ["--serve-script", serve_script]
    rc = run_stage("serve+verify", serve_cmd)
    if rc == 0:
        print("\n[pipeline] PASS — trained a head, served THAT head over the real "
              "local pool, and a live request returned the correct answer.", flush=True)
    else:
        print("\n[pipeline] FAIL — serve+verify stage did not pass.", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
