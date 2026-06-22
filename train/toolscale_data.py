"""ToolScale data + reward for training a Conductor (Fugu-Ultra line) with GRPO.

Drop-in for the conductor_supplementary Hydra stack: mirrors the signatures of
custom_data/countdown_data.py (make_datasets / make_reward_functions).

Design (honest scope):
  - nvidia/ToolScale gives each task a natural-language request plus
    `evaluation_criteria.actions` = the expected sequence of tool calls
    (name + arguments). It does NOT ship an executable tool environment.
  - So we DON'T execute tools. Instead the model is asked to emit the tool-call
    plan as JSON, and the reward is sequence-match against the expected actions:
    tool-name match (primary) + argument match (secondary) + format.
  This trains the orchestration/planning behaviour ToolScale targets, without a
  multi-thousand-line simulated environment the dataset omits.
"""
from __future__ import annotations
import json, re
from typing import List
from datasets import load_dataset

SYSTEM = (
    "You are a tool-use planner. Given a user request and the list of available "
    "tools, output the sequence of tool calls that fulfils it. Think briefly inside "
    "<think>...</think>, then output ONLY a JSON list inside <answer>...</answer>, "
    "where each item is {\"name\": <tool_name>, \"arguments\": {<arg>: <value>}}. "
    "Use the minimal correct sequence."
)


def _jsonable(v):
    # ToolScale args may contain datetime/other non-JSON types -> stringify
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def _expected_actions(ec) -> list[dict]:
    out = []
    for a in (ec or {}).get("actions", []) or []:
        args = {k: _jsonable(v) for k, v in (a.get("arguments") or {}).items() if v is not None}
        out.append({"name": a.get("name"), "arguments": args})
    return out


def make_datasets(data_limit=4000, seed=42, tokenizer=None,
                  dataset_id_or_path="nvidia/ToolScale", **kwargs):
    # base.yaml injects dataset_id_or_path / dataset_local_directory /
    # model_name_or_path / name — absorbed via kwargs; we honor the id.
    ds = load_dataset(dataset_id_or_path, split="train")
    ds = ds.shuffle(seed=seed)
    if data_limit and data_limit < len(ds):
        ds = ds.select(range(data_limit))

    def build(row):
        us = row.get("user_scenario") or {}
        instr = (us.get("instructions") or {})
        task = instr.get("task_instructions") or instr.get("reason_for_call") or ""
        expected = _expected_actions(row.get("evaluation_criteria"))
        # surface the tool names the task expects so the planner has a vocabulary
        tools = sorted({a["name"] for a in expected if a["name"]})
        tool_hint = ("Available tools: " + ", ".join(tools)) if tools else ""
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{tool_hint}\n\nUSER QUESTION: {task}".strip()},
            {"role": "assistant", "content": "Let me plan the tool calls.\n<think>"},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True),
            "expected_actions": json.dumps(expected),
        }

    ds = ds.map(build)
    # keep only rows that actually have expected actions (a learnable target)
    ds = ds.filter(lambda r: r["expected_actions"] != "[]")
    split = ds.train_test_split(test_size=0.1, seed=seed)
    return dict(train_dataset=split["train"], eval_dataset=split["test"])


# ---- reward ----------------------------------------------------------------
_ANSWER_RE = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.I)


def _parse_plan(completion: str) -> list[dict] | None:
    """The trainer strips the leading <think>, so prepend it for the format check."""
    m = _ANSWER_RE.search("<think>" + completion)
    if not m:
        return None
    body = m.group(1).strip()
    try:
        data = json.loads(body)
    except Exception:
        try:
            data = json.loads(re.sub(r"'", '"', body))
        except Exception:
            return None
    if not isinstance(data, list):
        return None
    norm = []
    for it in data:
        if isinstance(it, dict) and "name" in it:
            norm.append({"name": it.get("name"),
                         "arguments": it.get("arguments") or {}})
    return norm


def _score(pred: list[dict], gold: list[dict]) -> float:
    """Sequence match: name set/order (0.7) + argument overlap (0.3)."""
    if not gold:
        return 0.0
    gold_names = [g["name"] for g in gold]
    pred_names = [p["name"] for p in pred]
    # name recall over the expected set
    name_hit = sum(1 for n in gold_names if n in pred_names) / len(gold_names)
    # ordered bonus: longest matching prefix of names
    order = 0
    for a, b in zip(pred_names, gold_names):
        if a == b:
            order += 1
        else:
            break
    order_frac = order / len(gold_names)
    name_score = 0.6 * name_hit + 0.4 * order_frac
    # argument match on name-aligned calls
    arg_scores = []
    used = set()
    for g in gold:
        for i, p in enumerate(pred):
            if i in used or p["name"] != g["name"]:
                continue
            used.add(i)
            ga, pa = g["arguments"], p["arguments"]
            if not ga:
                arg_scores.append(1.0)
            else:
                hit = sum(1 for k, v in ga.items()
                          if str(pa.get(k)) == str(v))
                arg_scores.append(hit / len(ga))
            break
    arg_score = sum(arg_scores) / len(gold) if gold else 0.0
    return 0.7 * name_score + 0.3 * arg_score


def make_reward_functions(output_dir=None, include_format_reward=True, **kwargs):
    def format_reward(completions: List[str], **kw):
        out = []
        for c in completions:
            ok = bool(_ANSWER_RE.search("<think>" + c)) and ("</think>" in ("<think>" + c))
            out.append(1.0 if ok else 0.0)
        return out

    def action_reward(completions: List[str], expected_actions: List[str] = None, **kw):
        rewards = []
        exp = expected_actions or [None] * len(completions)
        for comp, gold_json in zip(completions, exp):
            try:
                gold = json.loads(gold_json) if gold_json else []
                pred = _parse_plan(comp)
                rewards.append(0.0 if pred is None else _score(pred, gold))
            except Exception:
                rewards.append(0.0)
        return rewards

    return [format_reward, action_reward] if include_format_reward else [action_reward]


if __name__ == "__main__":
    # offline unit test of the reward (no model, no GPU)
    gold = [{"name": "get_forecast_details", "arguments": {"forecast_id": "76"}},
            {"name": "get_current_conditions", "arguments": {"location_id": "76"}}]
    perfect = '<think>x</think><answer>' + json.dumps(gold) + '</answer>'
    partial = '<think>x</think><answer>[{"name":"get_forecast_details","arguments":{"forecast_id":"76"}}]</answer>'
    wrong   = '<think>x</think><answer>[{"name":"send_email","arguments":{}}]</answer>'
    fns = make_reward_functions()
    ej = [json.dumps(gold)] * 3
    fmt = fns[0]([perfect, partial, wrong])
    act = fns[1]([perfect, partial, wrong], expected_actions=ej)
    print("format rewards:", fmt)
    print("action rewards:", [round(x, 3) for x in act])
    assert fmt == [1.0, 1.0, 1.0]
    assert act[0] > 0.95 and act[1] > 0.3 and act[2] < 0.1, act
    print("PASS — reward discriminates perfect / partial / wrong plans")
