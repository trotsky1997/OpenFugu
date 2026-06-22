#!/usr/bin/env python3
"""
fugu_ultra.py — a faithful, runnable reconstruction of Sakana Fugu-Ultra's
Conductor line: instead of routing one worker per turn (that's fugu_mini.py /
TRINITY), a Conductor LM emits an ENTIRE agentic workflow in one shot — three
equal-length lists (model_id / subtasks / access_list) forming a DAG over a
worker pool — which is then executed in topological order.

Provenance, stated honestly:
  [EXEC]  the execution engine — 3-list parse, DAG order, access-list visibility
          injection — is a faithful reimplementation of the TRINITY/Conductor
          authors' conductor_engine.py + conductor_utils.py.
  [DOC]   the GRPO-trained 7B Conductor weights are NOT public, so here the
          Conductor is a *prompted off-the-shelf model*. The Conductor paper's
          own claim is that prompting works (just below the RL-optimized model);
          this reproduces the mechanism, not the trained policy.

Workers (and the Conductor) run through litellm, so any provider pool works.

Usage:
  python fugu_ultra.py --query "..." --conductor novita/deepseek/deepseek-v4-pro \
      --slot-models <csv of 7 worker model ids>
  python fugu_ultra.py --self-test     # offline: parser + DAG executor on a canned workflow
"""
from __future__ import annotations
import argparse, ast, json, os, re, sys
from dataclasses import dataclass, field
from typing import Callable

N_AGENTS = 7
MAX_STEPS = 5                      # [DOC] Conductor workflows up to 5 steps
_SMART = str.maketrans("“”‘’", "\"\"''")

DEFAULT_SLOT_LABELS = [            # [DATA] training metadata; remappable to any provider
    "gpt-5", "claude-sonnet-4", "gemini-2.5-pro",
    "deepseek-r1-distill-qwen-32b", "gemma-3-27b-it",
    "qwen3-32b-reasoning", "qwen3-32b-direct",
]

# ---- 3-list parsing (faithful to conductor_utils._extract_any) [EXEC] --------
def _balanced_list(after: str) -> str | None:
    """Extract the first balanced [...] list, respecting quotes/escapes."""
    depth = 0; start = None; q = None; esc = False
    for i, ch in enumerate(after):
        if esc: esc = False; continue
        if ch == "\\": esc = True; continue
        if q:
            if ch == q: q = None
            continue
        if ch in "\"'": q = ch; continue
        if ch == "[":
            if depth == 0: start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start is not None:
                return after[start:i + 1]
    return None


def extract_list(text: str, labels: list[str]) -> list:
    """Find 'label: [ ... ]' then parse via ast -> json -> CSV fallback. [EXEC]"""
    tag = "|".join(re.escape(l) for l in labels)
    m = re.search(rf"({tag})\s*[:=]\s*", text, re.I)
    if not m:
        return []
    raw = _balanced_list(text[m.end():])
    if not raw:
        return []
    raw = raw.translate(_SMART).strip()
    try:
        return ast.literal_eval(raw)
    except Exception:
        pass
    try:
        return json.loads(re.sub(r"'", '"', raw))
    except Exception:
        pass
    items = [x.strip(" \"'") for x in raw.strip("[]").split(",") if x.strip()]
    return [int(x) if x.isdigit() else x for x in items]


def parse_workflow(text: str) -> tuple[list, list, list]:
    model_ids = extract_list(text, ["model_id", "model id", "model_ids", "model ids"])
    subtasks  = extract_list(text, ["subtasks", "subtask"])
    access    = extract_list(text, ["access_list", "access list", "access"])
    return model_ids, subtasks, access


def _is_all(x) -> bool:
    return isinstance(x, str) and x.strip().lower() in ("all", "[all]", "'all'")

# EXEC_MARKER

# ---- access-list visibility (choose_position: indices of earlier steps) [EXEC]
def visible_indices(access_list: list, step: int) -> list[int]:
    """Which earlier step outputs are visible to `step`. Forward references are
    rejected (topological order). 'all' => every earlier step. Faithful to
    _ascribe_history_positional_complex / _ascribe_history_binary."""
    if step == 0:
        return []
    a = access_list[step] if step < len(access_list) else []
    if _is_all(a):
        return list(range(step))
    if a in ([], "", None):
        return []
    out = []
    for pos in dict.fromkeys(a if isinstance(a, (list, tuple)) else [a]):
        if not isinstance(pos, int):
            continue
        if pos >= step:                                  # forward reference -> reject [EXEC]
            raise ValueError(f"step {step} references future/own step {pos} (not a DAG order)")
        if 0 <= pos < step:
            out.append(pos)
    return sorted(out)


# ---- the Conductor prompt (prompted stand-in for the RL-trained 7B) [DOC] ----
def conductor_prompt(query: str, slot_labels: list[str]) -> list[dict]:
    pool = "\n".join(f"  {i}: {name}" for i, name in enumerate(slot_labels))
    sys = (
        "You are a Conductor that orchestrates a pool of worker LLMs to solve a task. "
        "Design an agentic workflow as THREE equal-length Python lists:\n"
        "  model_id   = [int, ...]   # which worker (0-indexed) runs each step\n"
        "  subtasks   = [str, ...]   # the natural-language instruction for each step\n"
        "  access_list= [list, ...]  # for each step, the indices of EARLIER steps whose\n"
        "                            # outputs that step may see ([] = none, may use \"all\")\n"
        "Rules: lists must be equal length (<=5 steps); access_list may only reference "
        "strictly earlier steps (it is a DAG executed in order); the LAST step's output "
        "is the final answer. Pick workers to match each subtask's demands.\n\n"
        f"AVAILABLE LANGUAGE MODELS:\n{pool}\n\n"
        "Output the three lists explicitly as 'model_id: [...]', 'subtasks: [...]', "
        "'access_list: [...]'. You may reason first, but the three lists must appear."
    )
    return [{"role": "system", "content": sys},
            {"role": "user", "content": f"USER QUESTION: {query}"}]


WorkerFn = Callable[[str, list, int], str]   # (subtask, messages, agent_id) -> reply


@dataclass
class Step:
    idx: int
    agent_id: int
    subtask: str
    sees: list[int]
    reply: str


@dataclass
class UltraResult:
    final: str
    steps: list[Step] = field(default_factory=list)
    workflow: dict = field(default_factory=dict)


class ConductorExecutor:
    """Parse a Conductor completion into a workflow DAG and execute it. [EXEC]

    Each step prompts its worker with its subtask plus the outputs of the steps
    named in access_list, injected as <Agent N response> blocks (the engine's
    exact context-assembly format)."""
    def __init__(self, worker: WorkerFn, slot_labels=None, max_steps=MAX_STEPS):
        self.worker = worker
        self.slot_labels = slot_labels or DEFAULT_SLOT_LABELS
        self.max_steps = max_steps

    def validate(self, model_ids, subtasks, access):
        if not (subtasks and model_ids and access):
            raise ValueError("workflow missing one of model_id/subtasks/access_list")
        if not (len(model_ids) == len(subtasks) == len(access)):
            raise ValueError(f"lists unequal length: "
                             f"{len(model_ids)}/{len(subtasks)}/{len(access)}")
        if len(subtasks) > self.max_steps:
            subtasks, model_ids, access = (subtasks[:self.max_steps],
                                           model_ids[:self.max_steps], access[:self.max_steps])
        return model_ids, subtasks, access

    def execute(self, model_ids, subtasks, access, verbose=False) -> UltraResult:
        model_ids, subtasks, access = self.validate(model_ids, subtasks, access)
        res = UltraResult(final="", workflow={"model_id": model_ids,
                                              "subtasks": subtasks, "access_list": access})
        outputs: list[str] = []
        for t, (mid, sub) in enumerate(zip(model_ids, subtasks)):
            sees = visible_indices(access, t)
            ctx = ""
            for j in sees:
                ctx += (f"\n<Subtask assigned to Agent {model_ids[j]}>{subtasks[j]}"
                        f"</Subtask assigned to Agent {model_ids[j]}>"
                        f"\n<Agent {model_ids[j]} response>{outputs[j].strip()}"
                        f"</Agent {model_ids[j]} response>")
            user = (f"USER QUESTION context:\n{ctx}\n\nYour subtask: {sub}"
                    if ctx else f"Your subtask: {sub}")
            mid = int(mid) % len(self.slot_labels)
            reply = self.worker(sub, [{"role": "user", "content": user}], mid)
            outputs.append(reply)
            res.steps.append(Step(t, mid, sub, sees, reply))
            if verbose:
                print(f"  step {t}: agent={mid}({self.slot_labels[mid]}) sees={sees}")
                print(f"    subtask: {sub[:80]}")
                print(f"    -> {reply.strip()[:90]}")
        res.final = outputs[-1] if outputs else ""        # last step = answer [EXEC]
        return res

# CLI_MARKER

class LiteLLMWorker:
    """Provider-agnostic worker via litellm (same middle layer as fugu_mini)."""
    def __init__(self, slot_models=None, api_key=None, api_base=None,
                 max_tokens=1024, temperature=0.2):
        import litellm
        self.litellm = litellm
        default = os.environ.get("FUGU_WORKER_MODEL", "openai/gpt-4o-mini")
        self.slot_models = slot_models or [default] * N_AGENTS
        self.api_key = api_key or os.environ.get("FUGU_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("FUGU_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.max_tokens, self.temperature = max_tokens, temperature

    def _call(self, model, messages):
        kw = dict(model=model, messages=messages,
                  max_tokens=self.max_tokens, temperature=self.temperature)
        if self.api_key:  kw["api_key"] = self.api_key
        if self.api_base: kw["api_base"] = self.api_base
        return self.litellm.completion(**kw).choices[0].message.content or ""

    def __call__(self, subtask, messages, agent_id):
        return self._call(self.slot_models[agent_id % len(self.slot_models)], messages)

    def conduct(self, model, messages):     # the Conductor call (more tokens)
        old = self.max_tokens; self.max_tokens = 2048
        try:
            return self._call(model, messages)
        finally:
            self.max_tokens = old


class MockWorker:
    """Offline: deterministic replies so parser+DAG can be tested with no keys."""
    def __call__(self, subtask, messages, agent_id):
        return f"[agent {agent_id}] result for: {subtask[:50]}"


CANNED = (  # a Conductor-style completion for the offline self-test
    "Plan: derive then implement then verify.\n"
    "model_id: [2, 0, 1]\n"
    'subtasks: ["Devise an algorithm for the task", '
    '"Implement it in Python using the devised algorithm", '
    '"Verify the implementation is correct"]\n'
    "access_list: [[], [0], [0, 1]]\n"
)


def self_test() -> int:
    """Offline: parse a canned workflow and execute the DAG with a mock pool.
    Checks parsing, equal-length validation, topological visibility."""
    mids, subs, acc = parse_workflow(CANNED)
    print("parsed workflow:")
    print(f"  model_id   = {mids}")
    print(f"  subtasks   = {[s[:30]+'...' for s in subs]}")
    print(f"  access_list= {acc}")
    assert mids == [2, 0, 1], mids
    assert acc == [[], [0], [0, 1]], acc
    assert len(mids) == len(subs) == len(acc) == 3
    # visibility
    assert visible_indices(acc, 0) == []
    assert visible_indices(acc, 1) == [0]
    assert visible_indices(acc, 2) == [0, 1]
    # forward-ref rejection
    try:
        visible_indices([[], [2], []], 1); raise AssertionError("should have rejected")
    except ValueError:
        pass
    res = ConductorExecutor(MockWorker()).execute(mids, subs, acc, verbose=True)
    assert len(res.steps) == 3 and res.final
    print("\nPASS — parser, equal-length, DAG order, forward-ref ban, execution all OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Faithful Fugu-Ultra (Conductor) workflow executor.")
    ap.add_argument("--query")
    ap.add_argument("--conductor", help="litellm model id acting as the Conductor")
    ap.add_argument("--slot-models", metavar="CSV", help="7 litellm worker model ids")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not (args.query and args.conductor):
        ap.error("need --query and --conductor (or --self-test)")

    slots = args.slot_models.split(",") if args.slot_models else None
    worker = LiteLLMWorker(slot_models=slots)
    slot_labels = slots or DEFAULT_SLOT_LABELS

    print(f"conductor: {args.conductor}")
    print(f"query: {args.query}\n")
    completion = worker.conduct(args.conductor, conductor_prompt(args.query, slot_labels))
    mids, subs, acc = parse_workflow(completion)
    if not subs:
        print("Conductor did not emit a parseable workflow. Raw completion:\n")
        print(completion[:800]); return 1
    print(f"workflow: model_id={mids}  access_list={acc}")
    print(f"  ({len(subs)} steps)\n")
    res = ConductorExecutor(worker, slot_labels=slot_labels).execute(mids, subs, acc, verbose=True)
    print(f"\nfinal answer (step {len(res.steps)-1} output):\n{res.final.strip()[:600]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


