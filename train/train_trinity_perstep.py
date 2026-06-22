#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: TRINITY (arXiv:2512.04695). The REAL per-STEP training: sep-CMA-ES
# optimizes the router head where fitness = terminal reward of the FULL
# multi-turn step_trinity rollout (openfugu/mini.py Coordinator). Original code.
"""
train_trinity_perstep.py — per-STEP TRINITY router training.

Unlike train_trinity_local.py (per-QUESTION: route once, one worker answers the
whole question), this trains at Fugu's real granularity: each sep-CMA-ES
candidate is a router head; its fitness is the terminal reward of running the
full per-step Coordinator loop over GSM8K — per turn the router re-reads the
evolving obs (question + accumulated <reference_thought_N>), re-routes a
(worker, role) action, a local worker advances one step, a verifier terminates.

  router  : Qwen3-0.6B hidden state -> candidate head -> (agent_id, role_id)
  workers : local multi-vendor <=8B models (one per GPU), via the Coordinator
  reward  : terminal — did the final answer match the GSM8K number
  train   : sep-CMA-ES over the 10240-dim head (SVF frozen)

This is the piece that makes the routing per-step. Cost is real: each fitness
eval is a multi-turn rollout with several worker generations, so scale is small.
"""
from __future__ import annotations
import argparse, os, re, sys, glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "openfugu"))
sys.path.insert(0, "/root")                      # server fallback for mini.py
import mini
from mini import FuguRouter, Coordinator, HIDDEN, HEAD_ROWS, N_AGENTS


def numeric_answer(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", (text or "").replace(",", ""))
    return nums[-1] if nums else None


class LocalPoolWorker:
    """Worker pool of local multi-vendor models. The Coordinator calls
    (role_name, messages, agent_id) -> reply; we dispatch to model[agent_id].
    Solver replies are nudged to include <think>…</think> so the Coordinator can
    extract a thought into the router obs (matching _get_obs)."""
    def __init__(self, specs, max_new=384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.max_new = torch, max_new
        self.names, self.toks, self.models, self.devs = [], [], [], []
        for name, path, dev in specs:
            tk = AutoTokenizer.from_pretrained(path)
            if tk.pad_token is None:
                tk.pad_token = tk.eos_token
            try:
                m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
            except TypeError:
                m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(dev).eval()
            self.names.append(name); self.toks.append(tk); self.models.append(m); self.devs.append(dev)
        self.cache = {}

    def __call__(self, role_name, messages, agent_id):
        wid = agent_id % len(self.models)
        key = (wid, role_name, messages[-1]["content"][:200])
        if key in self.cache:
            return self.cache[key]
        torch = self.torch
        tk, model, dev = self.toks[wid], self.models[wid], self.devs[wid]
        try:
            text = tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = "\n".join(m["content"] for m in messages)
        ids = tk(text, return_tensors="pt", truncation=True, max_length=2048).to(dev)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=self.max_new, do_sample=False,
                                 pad_token_id=tk.pad_token_id)
        reply = tk.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        self.cache[key] = reply
        return reply


def main():
    ap = argparse.ArgumentParser(description="Per-step TRINITY router training (real rollout fitness).")
    ap.add_argument("--router-model", default=os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument("--vector", default="/root/model_iter_60.npy",
                    help="base vector; SVF part applied & frozen, head is what we train")
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--sigma0", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="trinity_perstep.npy")
    args = ap.parse_args()

    import cma
    from datasets import load_dataset

    HUB = "/vePFS-Mindverse/share/huggingface/hub"
    def snap(repo):
        g = glob.glob(f"{HUB}/{repo}/snapshots/*/")
        return g[0] if g else None
    POOL = [(n, snap(r), d) for n, r, d in [
        ("deepseek-distill-7b", "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B", "cuda:1"),
        ("llama-3.2-3b",        "models--meta-llama--Llama-3.2-3B-Instruct",        "cuda:2"),
        ("gemma-3-4b",          "models--google--gemma-3-4b-it",                    "cuda:3"),
    ] if snap(r)]
    print(f"[perstep] worker pool: {[n for n,_,_ in POOL]}", flush=True)

    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{args.n_train}]")
    tasks = [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in ds]

    # router on cuda:0; load base vector, keep SVF applied (frozen), train the head
    router = FuguRouter(args.router_model, args.vector, device="cuda:0", seed=args.seed)
    base_head = router.head.clone()
    pool = LocalPoolWorker(POOL)
    import torch
    HEAD_DIM = HEAD_ROWS * HIDDEN

    def rollout_solved(head_vec):
        router.head = torch.from_numpy(head_vec.copy()).float().reshape(HEAD_ROWS, HIDDEN).to(router.device)
        coord = Coordinator(router, pool, max_turns=args.max_turns, sample=False)
        solved = 0
        for q, gold in tasks:
            res = coord.run(q)
            if numeric_answer(res.final) == gold:
                solved += 1
        return solved / len(tasks)

    base_fit = rollout_solved(base_head.cpu().numpy().ravel())
    print(f"[perstep] base head rollout solved={base_fit:.3f}  (n={len(tasks)}, max_turns={args.max_turns})", flush=True)

    es = cma.CMAEvolutionStrategy(base_head.cpu().numpy().ravel(), args.sigma0,
                                  {"seed": args.seed, "verbose": -9, "CMA_diagonal": True})
    best_vec, best_fit = base_head.cpu().numpy().ravel(), base_fit
    for it in range(args.iters):
        cands = es.ask()
        fits = [rollout_solved(c) for c in cands]
        es.tell(cands, [-f for f in fits])
        i = int(np.argmax(fits))
        if fits[i] > best_fit:
            best_fit, best_vec = fits[i], cands[i].copy()
        print(f"[iter {it}] best_solved={best_fit:.3f} (base {base_fit:.3f})", flush=True)

    np.save(args.out, best_vec)
    print(f"\n[result] per-step trained head solved={best_fit:.3f} vs base {base_fit:.3f}")
    print(f"[result] saved {args.out}")
    if best_fit > base_fit + 0.01:
        print("PASS — per-step sep-CMA improved the router over the multi-turn rollout")
    else:
        print(f"NOTE — no improvement over base in {args.iters} iters (small scale / saturated)")


if __name__ == "__main__":
    main()
