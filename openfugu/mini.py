#!/usr/bin/env python3
"""
fugu_mini.py — a faithful, minimal, runnable reconstruction of Sakana Fugu's
inference path (the TRINITY line), built entirely from reverse-engineered
constants that were verified end-to-end against the released `model_iter_60.npy`
checkpoint (95% agent / 100% role on a 37-case fixture).

This is the *implementation*, not a verification script: it exposes a clean
`FuguRouter` (hidden-state -> worker/role logits) and a `Coordinator` (the full
multi-turn step_trinity loop with role injection, worker dispatch, and
verifier/max-turn termination). Worker LLMs are pluggable; a MockWorker lets the
whole loop run offline with no API keys.

Every non-obvious constant is annotated with how it was established:
  [EXEC]  reproduced by running real weights
  [CODE]  read from the TRINITY authors' code submission
  [DATA]  from the released training log / checkpoint

Usage:
  python fugu_mini.py --self-test            # re-run the 37-case fixture, assert faithfulness
  python fugu_mini.py --demo                 # run the coordination loop with a mock worker pool
  python fugu_mini.py --route "your prompt"  # one routing decision

Requires: torch, transformers, numpy, and a local Qwen3-0.6B + model_iter_60.npy.
Paths are overridable via --model / --vector / --fixture or the env vars
FUGU_MODEL / FUGU_VECTOR / FUGU_FIXTURE.
"""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---- verified structural constants ------------------------------------------
HIDDEN = 1024              # [EXEC] Qwen3-0.6B hidden size
N_AGENTS = 7               # [DATA] es_log.json: 7-model worker pool
N_ROLES = 3                # [CODE] solver/thinker/verifier
HEAD_ROWS = N_AGENTS + N_ROLES        # 10
SVF_LEN = 9 * HIDDEN                   # 9216  [EXEC] 9 matrices x 1024 singular values
HEAD_LEN = HEAD_ROWS * HIDDEN          # 10240
VEC_LEN = SVF_LEN + HEAD_LEN           # 19456 [EXEC] exact length of model_iter_60.npy
HIDDEN_POS = -2            # [EXEC] penultimate-token hidden state
OPT_LAYER = 26            # [DATA] es_log.json opt_layer_indices=[26]
MAX_TURNS = 5             # [DATA] es_log.json max_turns=5
ROLE_NAMES = ["Worker", "Thinker", "Verifier"]   # [CODE] index order (Python: solver/thinker/verifier)

# role system prompts — paraphrase of the TRINITY role contracts [CODE]
ROLE_PROMPTS = {
    "Worker":  ("Execute the next concrete step of the solution. Produce code, "
                "math, derivations, or concrete answer content that advances the task."),
    "Thinker": ("Analyze the current state and give high-level guidance: plans, "
                "decompositions, or critiques. You may end with a line "
                "'<suggested_role>solver|verifier</suggested_role>' to steer the next turn."),
    "Verifier":("Check the current solution for correctness and completeness. "
                "Begin your reply with exactly ACCEPT or REVISE, then a brief reason."),
}


def _resolve(path_arg, env, default):
    return path_arg or os.environ.get(env) or default


# ---- router core ------------------------------------------------------------
class FuguRouter:
    """Qwen3-0.6B backbone + SVF adaptation + bias-free linear head.

    route(messages) -> dict(agent_id, role_id, role_name, agent_logits, role_logits)
    The backbone's own text output is never used; only the head logits matter,
    which is what makes a routing decision ~one forward pass. [EXEC]
    """

    def __init__(self, model_dir: str, vector_path: str, dtype="float32",
                 device: str | None = None, seed: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.rng = np.random.default_rng(seed)

        vec = np.load(vector_path).astype(np.float64)
        if vec.shape != (VEC_LEN,):
            raise ValueError(f"router vector must be {VEC_LEN} floats, got {vec.shape}")

        self.tok = AutoTokenizer.from_pretrained(model_dir)
        # transformers >=5 uses dtype=, <5 uses torch_dtype= — support both
        td = getattr(torch, dtype)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=td).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=td).eval()
        if device:
            self.model.to(device)
        self.device = next(self.model.parameters()).device

        self._apply_svf(vec[:SVF_LEN])
        # head: last 10240 -> (10, 1024) [EXEC]
        self.head = torch.from_numpy(vec[SVF_LEN:].copy()).float().reshape(HEAD_ROWS, HIDDEN).to(self.device)

    # SVF: scale only singular values, freeze U/V, energy-preserving renorm. [CODE]
    # Matrices consumed in state_dict order: embed_tokens, layer-26 {q,k,v,o,
    # gate,up,down}, lm_head — exactly 9 x 1024 = 9216 offsets. [EXEC]
    def _apply_svf(self, offsets: np.ndarray):
        torch = self.torch
        sd = self.model.state_dict()
        keys = [k for k in sd
                if sd[k].ndim == 2 and min(sd[k].shape) > 1
                and ("model.layers." not in k or f"model.layers.{OPT_LAYER}." in k)]
        off = 0
        with torch.no_grad():
            for k in keys:
                W = sd[k].float()
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                n = S.numel()
                scale = torch.from_numpy(offsets[off:off + n].copy()).float().to(W.device) + 1.0
                off += n
                sS = S * scale
                newW = (U @ torch.diag(sS) @ Vh) * (S.sum() / sS.sum())
                sd[k].copy_(newW.to(sd[k].dtype))
        if off != SVF_LEN:
            raise RuntimeError(f"consumed {off} SVF offsets, expected {SVF_LEN} "
                               f"(model may not be Qwen3-0.6B)")
        self.svf_keys = keys

    @staticmethod
    def format_transcript(messages: list[dict]) -> str:
        # raw 'role: content', NOT a chat template — proven decisive (95% vs 11%). [EXEC]
        return "\n".join(f'{m["role"]}: {m["content"]}' for m in messages)

    def _hidden(self, messages):
        torch = self.torch
        text = self.format_transcript(messages)
        ids = self.tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.model(**ids)          # backbone only; LM head unused
        return out.last_hidden_state[0, HIDDEN_POS, :]

    def _pick(self, logits, sample: bool):
        torch = self.torch
        if sample:                                  # softmax sampling = training behavior [CODE]
            p = torch.softmax(logits, 0).cpu().numpy()
            return int(self.rng.choice(len(p), p=p))
        return int(torch.argmax(logits))            # argmax = eval behavior

    def route(self, messages: list[dict], sample: bool = False) -> dict:
        h = self._hidden(messages)
        logits = self.head @ h                      # (10,)
        agent_logits, role_logits = logits[:N_AGENTS], logits[N_AGENTS:]
        agent_id = self._pick(agent_logits, sample)
        role_id = self._pick(role_logits, sample)
        return {
            "agent_id": agent_id,
            "role_id": role_id,
            "role_name": ROLE_NAMES[role_id],
            "agent_logits": agent_logits.detach().cpu().numpy(),
            "role_logits": role_logits.detach().cpu().numpy(),
        }

# COORDINATOR_MARKER

# ---- worker pool ------------------------------------------------------------
# A worker is any callable: (role_name, messages, agent_id) -> reply text.
# Real deployment binds each of the 7 agent slots to a provider; slot labels in
# the checkpoint ("gpt-5", "gemini-2.5-pro", ...) are training metadata, freely
# remappable — which is the product's "swap providers / dodge export controls". [CODE]
WorkerFn = Callable[[str, list, int], str]

DEFAULT_SLOT_LABELS = [          # [DATA] es_log.json llm_names order
    "gpt-5", "claude-sonnet-4", "gemini-2.5-pro",
    "deepseek-r1-distill-qwen-32b", "gemma-3-27b-it",
    "qwen3-32b-reasoning", "qwen3-32b-direct",
]


class MockWorker:
    """Offline stand-in: deterministic, lets the full loop run with no API keys.
    The verifier accepts on the 2nd verification (so the loop demonstrably
    terminates via ACCEPT rather than only by max-turns)."""
    def __init__(self):
        self._verifications = 0

    def __call__(self, role_name: str, messages: list, agent_id: int) -> str:
        slot = DEFAULT_SLOT_LABELS[agent_id] if agent_id < len(DEFAULT_SLOT_LABELS) else f"agent{agent_id}"
        if role_name == "Thinker":
            return ("Plan: decompose, solve, then verify. "
                    "<suggested_role>solver</suggested_role>")
        if role_name == "Verifier":
            self._verifications += 1
            return "ACCEPT — solution is complete." if self._verifications >= 2 else \
                   "REVISE — tighten the final step."
        return f"[{slot}] concrete work toward the solution."


class LiteLLMWorker:
    """Real worker pool via litellm as the provider-agnostic middle layer.
    litellm.completion() speaks one API to every backend, so each of the 7
    agent slots can be a different provider/model with no per-vendor code —
    which mirrors Fugu's own "swappable heterogeneous pool". [CODE]

    `slot_models` is a list of up to 7 litellm model ids (e.g.
    'openai/gpt-4o-mini', 'anthropic/claude-3-5-sonnet', 'gemini/gemini-1.5-pro').
    Credentials/base url are taken from litellm's normal env resolution, or
    passed through `api_key`/`api_base` (read from FUGU_API_KEY/FUGU_BASE_URL).
    Default points every slot at FUGU_WORKER_MODEL so the loop runs with one model."""
    def __init__(self, slot_models: list[str] | None = None,
                 api_key: str | None = None, api_base: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.2):
        import litellm
        self.litellm = litellm
        default_model = os.environ.get("FUGU_WORKER_MODEL", "openai/gpt-4o-mini")
        self.slot_models = slot_models or [default_model] * N_AGENTS
        self.api_key = api_key or os.environ.get("FUGU_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("FUGU_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.max_tokens, self.temperature = max_tokens, temperature

    def __call__(self, role_name: str, messages: list, agent_id: int) -> str:
        model = self.slot_models[agent_id % len(self.slot_models)]
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        kw = dict(model=model, messages=msgs,
                  max_tokens=self.max_tokens, temperature=self.temperature)
        if self.api_key:  kw["api_key"] = self.api_key
        if self.api_base: kw["api_base"] = self.api_base
        r = self.litellm.completion(**kw)
        return r.choices[0].message.content or ""


# ---- the coordination loop (step_trinity, faithful) -------------------------
@dataclass
class Turn:
    turn: int
    agent_id: int
    role_name: str
    reply: str


@dataclass
class RunResult:
    final: str
    turns: list[Turn] = field(default_factory=list)
    terminated_by: str = ""        # "verifier_accept" | "max_turns" | "verifier_no_response"


class Coordinator:
    """Runs the bounded multi-turn TRINITY loop over a FuguRouter + worker pool.

    Faithful to step_trinity (core.py):
      - each turn: route -> role -> inject role prompt -> dispatch worker
      - a Thinker may emit <suggested_role> that OVERRIDES next turn's role [CODE]
      - terminate when a Verifier replies ACCEPT, or at max_turns [CODE]
      - if a Verifier is picked before any worker response exists -> stop [CODE]

    `suppress_cold_verifier` is an IMPLEMENTATION choice (not checkpoint
    behavior): on turns with no worker response yet, a Verifier pick is a no-op
    that would end the run at turn 0, so we re-route it to Worker. The raw
    step_trinity behavior (terminate) is kept when this flag is False.
    """
    def __init__(self, router: FuguRouter, worker: WorkerFn,
                 max_turns: int = MAX_TURNS, stop_token: str = "ACCEPT",
                 sample: bool = True, suppress_cold_verifier: bool = True):
        self.router, self.worker = router, worker
        self.max_turns, self.stop_token, self.sample = max_turns, stop_token, sample
        self.suppress_cold_verifier = suppress_cold_verifier

    def run(self, query: str, verbose: bool = False) -> RunResult:
        messages = [{"role": "user", "content": query}]
        res = RunResult(final="")
        last_response: str | None = None
        suggested_role: str | None = None

        for t in range(self.max_turns):
            r = self.router.route(messages, sample=self.sample)
            role = r["role_name"]
            if suggested_role:                      # thinker override consumes here [CODE]
                role, suggested_role = suggested_role, None
            agent_id = r["agent_id"]

            if role == "Verifier" and last_response is None:
                if self.suppress_cold_verifier:
                    role = "Worker"                 # impl choice: no-op verifier -> work instead
                else:
                    res.terminated_by = "verifier_no_response"
                    break

            sys_prompt = {"role": "system", "content": ROLE_PROMPTS[role]}
            reply = self.worker(role, [sys_prompt] + messages, agent_id)
            messages.append({"role": "assistant", "content": reply})
            res.turns.append(Turn(t, agent_id, role, reply))
            if verbose:
                print(f"  turn {t}: agent={agent_id}({DEFAULT_SLOT_LABELS[agent_id]}) "
                      f"role={role}\n    {reply[:90]}")

            if role == "Worker":
                last_response = reply
            elif role == "Thinker":
                suggested_role = self._parse_suggested_role(reply)
            elif role == "Verifier":
                if reply.strip().upper().startswith(self.stop_token):
                    res.final = last_response or reply
                    res.terminated_by = "verifier_accept"
                    return res

        res.final = last_response or (res.turns[-1].reply if res.turns else "")
        if not res.terminated_by:
            res.terminated_by = "max_turns"
        return res

    @staticmethod
    def _parse_suggested_role(text: str) -> str | None:
        import re
        m = re.search(r"<suggested_role>\s*(solver|thinker|verifier)\s*</suggested_role>",
                      text, re.I)
        if not m:
            return None
        return {"solver": "Worker", "thinker": "Thinker", "verifier": "Verifier"}[m.group(1).lower()]

# CLI_MARKER

# ---- self-test: prove the implementation is faithful to the checkpoint ------
def self_test(router: FuguRouter, fixture_path: str) -> int:
    """Re-run the 37-case routing fixture. This is the regression guard: if the
    implementation drifts from model_iter_60.npy, accuracy collapses. Expect
    ~95% agent / 100% role (vs ~51% best-constant baseline). [EXEC]"""
    cases = json.load(open(fixture_path))["cases"]
    a_hit = r_hit = 0
    from collections import Counter
    ea = [c["expected"]["agent_id"] for c in cases]
    er = [c["expected"]["role_id"] for c in cases]
    base_a = Counter(ea).most_common(1)[0][1] / len(cases)
    base_r = Counter(er).most_common(1)[0][1] / len(cases)
    for c in cases:
        r = router.route(c["messages"], sample=False)   # argmax for eval
        a_hit += (r["agent_id"] == c["expected"]["agent_id"])
        r_hit += (r["role_id"] == c["expected"]["role_id"])
    n = len(cases)
    print(f"self-test on {n} cases:")
    print(f"  agent {a_hit}/{n} = {a_hit/n:.0%}   (baseline {base_a:.0%})")
    print(f"  role  {r_hit}/{n} = {r_hit/n:.0%}   (baseline {base_r:.0%})")
    ok = a_hit / n >= 0.90 and r_hit / n >= 0.95
    print("  PASS — implementation faithful to checkpoint" if ok else
          "  FAIL — implementation drifted from checkpoint")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Minimal faithful Fugu (TRINITY) inference.")
    ap.add_argument("--model", help="Qwen3-0.6B dir (env FUGU_MODEL)")
    ap.add_argument("--vector", help="model_iter_60.npy (env FUGU_VECTOR)")
    ap.add_argument("--fixture", help="qwen_router_prompt_eval_cases.json (env FUGU_FIXTURE)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--route", metavar="PROMPT", help="one routing decision for PROMPT")
    ap.add_argument("--live", action="store_true",
                    help="demo with a real worker pool via litellm (needs FUGU_API_KEY/_BASE_URL)")
    ap.add_argument("--slot-models", metavar="CSV",
                    help="comma-separated litellm model ids for the 7 agent slots")
    ap.add_argument("--query", help="override the --demo query")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    model = _resolve(args.model, "FUGU_MODEL", "Qwen/Qwen3-0.6B")
    vector = _resolve(args.vector, "FUGU_VECTOR", "model_iter_60.npy")
    fixture = _resolve(args.fixture, "FUGU_FIXTURE",
                       "trinity_coordinator/examples/fixtures/qwen_router_prompt_eval_cases.json")

    if not (args.self_test or args.demo or args.route):
        ap.error("choose one of --self-test / --demo / --route")

    router = FuguRouter(model, vector, seed=args.seed)

    if args.self_test:
        return self_test(router, fixture)

    if args.route:
        r = router.route([{"role": "user", "content": args.route}], sample=False)
        slot = DEFAULT_SLOT_LABELS[r["agent_id"]]
        print(f"agent {r['agent_id']} ({slot}), role {r['role_name']}")
        print(f"  agent_logits {np.round(r['agent_logits'], 2)}")
        print(f"  role_logits  {np.round(r['role_logits'], 2)}")
        return 0

    if args.demo:
        if args.live:
            models = args.slot_models.split(",") if args.slot_models else None
            worker = LiteLLMWorker(slot_models=models)
            print("worker pool: LiteLLMWorker (live, via litellm)")
        else:
            worker = MockWorker()
            print("worker pool: MockWorker (offline)")
        coord = Coordinator(router, worker, sample=True)
        q = args.query or "Implement binary search in Python and prove it terminates."
        print(f"query: {q}\n")
        res = coord.run(q, verbose=True)
        print(f"\nterminated_by: {res.terminated_by}  ({len(res.turns)} turns)")
        print(f"final: {res.final[:400]}")
        return 0


if __name__ == "__main__":
    sys.exit(main())


