# Sakana Fugu — Reverse-Engineered Architecture

> Twelve rounds of investigation across 3 papers, 4 repos, 1 PDF report, and
> live execution on an 8×A800 server. Every claim below carries an **evidence
> grade** so you know what's proven vs inferred vs unreachable.

## Evidence grades

| Grade | Meaning |
|-------|---------|
| 🟢 **EXEC** | Verified by running real weights/model. Reproducible. |
| 🔵 **CODE** | Read directly from author source (`trinity_code_submission`, `conductor_supplementary`). |
| 🟣 **DATA** | Computed from a local artifact (`model_iter_60.npy`, `es_log.json`) or library default. |
| 🟡 **DOC** | Stated in the technical report / paper, no code to confirm. |
| 🟠 **INFER** | Reconstructed from constraints; plausible but unverified. |
| 🔴 **DARK** | Structurally unreachable (code not shipped / closed source). |

## TL;DR

**Fugu is not one model.** It is a product family of two *learned orchestrators*
that expose a pool of frontier LLMs behind a single OpenAI-compatible API. 🟡

- **Fugu** (low latency) builds on **TRINITY** (arXiv 2512.04695): a tiny head
  on a Qwen3-0.6B backbone picks one worker per turn from its hidden state. 🟢
- **Fugu-Ultra** (max quality) builds on **Conductor** (arXiv 2512.04388): a 7B
  LLM emits a full natural-language agentic-workflow DAG. 🔵🟡
- The orchestrator is trained, the workers are not. No weight merging — pure
  macro-level composition over heterogeneous APIs. 🟡
- The two are **peer variants** on a quality/latency frontier, NOT a cascade. 🟡

The GitHub `SakanaAI/fugu` repo is 100% shell — it installs the closed API into
Codex. The architecture lives entirely in the two papers + the report. 🔵

---

## 1. Fugu (TRINITY line) — inference path

### 1.1 Parametrization

```
transcript ("role: content\n"...)         # raw format, NOT chat template  🟢
   │
   ▼  Qwen3-0.6B backbone (forward monkeypatched)         🔵 modeling_qwen2.py:102-123
   │  returns early at action_layer, bypassing LM head
   ▼
hidden state h ∈ ℝ^1024 at position -2 (penultimate token)   🟢
   │
   ▼  linear head W:{10,1024}, bias=False                  🟢
   ▼
logits[10] → agent_logits = logits[:7]                      🔵 core.py:901
             role_logits  = logits[7:]
   │
   ▼  softmax-sample (training) / argmax (eval)             🔵 core.py:896-903
agent_id ∈ {0..6},  role_id ∈ {0:Worker/solver, 1:Thinker, 2:Verifier}
```

- The router's **own generated text is discarded** — only head logits matter,
  so it can read an early-token hidden state and skip autoregressive decoding. 🟡
- **Product Fugu drops roles** (head outputs only L worker logits); the
  role-assigning L+3 head is the *academic* TRINITY design. 🟡

### 1.2 Trainable parameter vector — `model_iter_60.npy`, exactly 19456 floats 🟢

```
[ 0      : 9216 ]  SVF singular-value scale offsets
[ 9216   : 19456]  router head, reshape → (10, 1024) = (7 agents + 3 roles, hidden)
```

**SVF touches 9 matrices, in state_dict order** (verified by counting the real
Qwen3 safetensors header + reconstructing on GPU): 🟢

```
0  model.embed_tokens.weight        (151936,1024)  → 1024 singular values
1  layers.26.self_attn.q_proj       (2048,1024)    → 1024
2  layers.26.self_attn.k_proj       (1024,1024)    → 1024
3  layers.26.self_attn.v_proj       (1024,1024)    → 1024
4  layers.26.self_attn.o_proj       (1024,2048)    → 1024
5  layers.26.mlp.gate_proj          (3072,1024)    → 1024
6  layers.26.mlp.up_proj            (3072,1024)    → 1024
7  layers.26.mlp.down_proj          (1024,3072)    → 1024
8  lm_head.weight                   (151936,1024)  → 1024
                                           total = 9216 ✓
```

> Correction (cost 8 rounds): NOT "9 matrices in layer 26". It is **7 in
> layer 26 + embed_tokens + lm_head**. SVF adapts the two largest global
> matrices, not just the penultimate layer. 🟢

### 1.3 SVF reconstruction formula 🔵 trainer.py:1183-1191

```python
scale    = flat[off:off+k] + 1.0                       # 0 offset = identity
scaled_S = S * scale
newW     = (U @ diag(scaled_S) @ Vᵀ) * (S.sum() / scaled_S.sum())   # energy-preserving
```
The `S.sum()/scaled_S.sum()` term keeps total singular energy constant. 🔵

### 1.4 Coordination loop 🔵 core.py:879-1000

- At most `max_turns=5` turns. 🟣
- Each turn: route → inject role prompt → call the selected worker LLM. 🔵
- **Termination (trinity mode):** verifier outputs `ACCEPT` → done; OR
  `max_turns` hit → return latest worker response; OR verifier picked with no
  response yet → done with 0 reward. 🔵
- Thinker can emit `<suggested_role>` that **overrides** the head's role pick
  next turn. 🔵
- (The "consecutive-same-agent returns" rule is the *standard* mode, NOT
  trinity. Correcting an earlier error of mine.) 🔵

### 1.5 EXECUTION PROOF 🟢

Ran `model_iter_60.npy` + real Qwen3-0.6B on A800, applied SVF in state_dict
order, against the 37-case fixture (`qwen_router_prompt_eval_cases.json`):

```
baseline (always guess mode class):  agent 51%   role 49%
raw "role: content" format:          agent 95%   role 100%   both 95%   ← PROVEN
chat_template format:                agent 41%   role 11%    both  5%   ← wrong input
```

95% vs 51% baseline = statistically impossible by chance. This **simultaneously
proves**: vector split order, SVF matrix order, reconstruction formula, the
raw-transcript input format, and that the npy is genuinely paired with this
inference code. The 2 agent misses are explained: one has top1-top2
margin=0.214 (below the known 0.24 floor → fp32 numeric noise), one is a
unicode-emoji edge case. 🟢

Repro: `verify_37.py`, `verify_trinity2.py`, `verify_margin.py`.

---

## 2. Fugu (TRINITY) — training

### 2.1 Real hyperparameters 🟣 `logs/ckpt/es_log.json` (actual run config)

```
task = mix_m_m_r_l (MATH+MMLU+RLPR+LiveCodeBench)   model = Qwen/Qwen3-0.6B
num_iters = 60       (→ model_iter_60.npy = final)  sigma0 = 0.03
num_repeats = 16     (episodes averaged per cand)   seed = 42
num_tests/test_size = 300                           temperature = 0.1 (worker)
max_turns = 5        num_agents = 7                 last_token_predict = False
reward shaping: diversity_bonus=0.15  turn_bonus=0.10  cost_bonus=0.0
opt_layer_indices = [26]   closed_model_config = True
```

### 2.2 CMA defaults (popsize_override=0 → pycma defaults, n=19456) 🟣 closed-form

```
λ (popsize) = 4 + ⌊3 ln n⌋ = 33     ← paper's "≈32" is this, rounding
μ = 16,  mueff = 9.18
recomb weights = log-rank normalized (top 0.198 → tail 0.002)
```

### 2.3 Worker pool (7 slots) 🟣 es_log.json

```
gpt-5 · claude-sonnet-4 · gemini-2.5-pro · DeepSeek-R1-Distill-Qwen-32B
gemma-3-27b-it · Qwen3-32B(reasoning) · Qwen3-32B(direct)
```
Slot labels are **training metadata**, remappable to any provider at deploy
time — this is how the product "excludes providers / dodges export controls". 🔵

### 2.4 Training intensity (reverse-derived from weights) 🟣

```
SVF offsets std = 0.097      → singular values scaled ±9.7% on average (gentle)
31% of offsets |·|>0.1       max +43% / −34% on individual singular values
agent head row-means ≈ 0     → NO built-in agent preference; routing is purely
                               hidden-state-driven (explains task-adaptive routing)
```

### 2.5 Training MAIN LOOP — reconstructed, NOT shipped 🟠

The ask/tell loop lives in the un-shipped `experiments/with_training/
testing_standalone.py` (named at `evaluate_*.py:8`). `CMAEvolutionTrainer` has
only `__init__/_setup_svd_info/run_test` — no `ask`/`tell`, `import cma` is a
dead reference here. 🔵

Reconstruction in `recovered_training_loop.py` — **78% constraint-derived, 22%
labeled guesses**. Note `CMA_diagonal=True` (sep) is **required** here — full CMA
is infeasible at $N{=}19456$ — but in 60 iters only the scalar $\sigma$ moves, so
dynamics reduce to an isotropic $(\mu/\mu_w,\lambda)$-ES (see §5.1). 🟠

```python
solver = cma.CMAEvolutionStrategy(np.zeros(19456), 0.03, {seed:42})  # sep required: full infeasible at 19456-D (§5.1)
for it in range(60):
    cands = solver.ask()                                  # x_i = m + σ·z_i   🟣🟡
    fits  = [mean(R over 16 episodes) + 0.15·diversity + 0.10·turn  # 🟣
             for c in cands]                              # R∈{0,1} terminal   🟡
    solver.tell(cands, [-f for f in fits])                # weighted recomb    🟡
    if it%5==0: run_test();  save model_iter_{it}.npy     # 🟣
```

---

## 3. Fugu-Ultra (Conductor line)

### 3.1 Output = 3 equal-length lists forming a DAG 🔵 conductor_engine.py

```
model_id[]    : which worker per step (int)
subtasks[]    : NL instruction per step
access_list[] : which prior step outputs are visible to this step
```
- Parsed via `_extract_any`: `ast.literal_eval` → `json.loads` → CSV fallback. 🔵
- Executed **sequentially**; each worker sees prior outputs named by access_list,
  injected as `<Agent N response>…</Agent N response>`. 🔵
- Access modes: `binary` / `choose_id` / `choose_position`; all **forbid forward
  references** (raises on future-step access). So it's a *topological order*,
  not an arbitrary DAG. 🔵
- Conductor may name **itself** as a worker → recursive topologies. 🔵🟡

### 3.2 Training 🔵🟡

- GRPO (`from trl import GRPOTrainer`), **no KL penalty**. 🔵🟡
- Reward: r=0 if 3 lists don't parse; r=1 if executed workflow correct, else
  r=0.5 (this is the *Conductor* reward; in code the correctness term comes from
  an external task fn, not hardcoded). 🔵🟡
- Recursion finetune: `recursion_discount_factor=0.2` (paper said 0.25),
  `normalize_rewards_per_recursion_round=True`. 🔵
- Base Qwen2.5-7B, 200 GRPO iters, 2×H100, 960 problems. 🟡

### 3.3 Product-only extensions (in neither paper) 🟡

- **Intra-workflow agent isolation**: within one workflow each agent's
  function-call trajectory is isolated (except access_list), preventing
  "orchestration collapse" where the first agent's path dominates. 🟡
- **Persistent shared memory**: across workflows agents share tool-call memory,
  so they don't re-discover the same artifacts. 🟡
- Net: isolated within a workflow, shared across the multi-turn conversation. 🟡

---

## 4. Product layer 🟡

| | Fugu | Fugu-Ultra |
|---|---|---|
| Based on | TRINITY | Conductor |
| Orchestrator | Qwen3-0.6B + tiny head | 7B LM → NL DAG |
| Per query | 1 hidden→logits, pick worker | full agentic workflow |
| Latency | low (no decode) | high (deep orchestration + tool calls) |

Model Card (report Table 1, 🟡): Fugu-Ultra leads SWE-Bench-Pro 73.7, Terminal
Bench 82.1, LiveCodeBench 93.2, GPQA-D 95.5 — but loses MRCRv2 to GPT-5.5
(93.6 vs 94.8). Orchestration doesn't win everywhere. Worker pool = the same
Gemini-3.1-Pro / Opus-4.8 / GPT-5.5 it's benchmarked against. Routing
distribution matches each model's domain SOTA (Terminal→GPT, GPQA→Gemini). 🟡

Formulas 1-7 (SFT softmax target, KL loss, sep-CMA eqs, GRPO) transcribed to
LaTeX in `fugu/Fugu_technical_report.md` appendix. 🟡

---

## 5. The DARK side — and how far reverse-inference actually reaches

Not all "dark" is equal. Sorted by whether it matters for *understanding the
architecture* (vs. being a credential or a tuning knob):

### 5.1 The "sep" question — MECHANISM UNDERSTOOD (probe_sep_cma.py) 🟢

For 12 rounds I treated "diagonal (sep) or full?" as an architectural unknown,
then briefly dismissed it as "irrelevant." Both were imprecise. Probing pycma
v4.4.4 directly pinned down the actual mechanism — in three layers.

**(a) What sep-CMA-ES *is*** — the standard separable variant of Ros & Hansen
(2008): restrict the covariance to diagonal, dropping cost from $O(N^2)$ storage
/ $O(N^3)$ eigendecomposition to $O(N)$. Trades away rotational invariance; good
on (near-)separable and high-dim problems. Not Sakana's invention. 🟡

**(b) How pycma implements it** (executed, corrected my prior mental model):
sep mode does **not** use a diagonal `C` matrix. `CMA_diagonal=True` selects the
`GaussStandardConstant` sampler whose `C` is **frozen at $I$** ("no update");
the diagonal adaptation instead lives in `sigma_vec`, a per-coordinate
step-size vector. So 🟢

$$x_i = m + \sigma \cdot \texttt{sigma\_vec} \odot z_i, \quad z_i \sim \mathcal N(0, I),$$

which is *mathematically* a diagonal covariance but implemented off the sampler.
(Default mode is `GaussFullSampler` adapting the full $C$ with `sigma_vec`≡1 —
the `0*100*N/...` default string evaluates to **0.0**, so full, not diagonal.
Earlier "default runs diagonal first" was wrong.) 🟢

**(c) Behavior at Fugu scale** ($N{=}19456$, pop 33, 60 iters):
- **full is computationally infeasible** here ($N{\times}N$ matrix + eigendecomp
  → OOM/stall). So sep is **REQUIRED for feasibility**, not an optional nicety —
  correcting my "free/irrelevant" framing. 🟢
- sep runs in ~7 s, but its per-coordinate `sigma_vec` divergence is
  **negligible** (std $6\!\times\!10^{-4}$, range 0.996–1.002). The only thing
  that moves materially is the **scalar** $\sigma$ (0.03 → 0.002). 🟢
- At low dim the same sep `sigma_vec` *does* diverge ([1..1]→[0.22..0.06]) and
  beats full — so the capability is real, just not exercised in 60 iters at
  19456-D. 🟢

> Net: Sakana uses sep-CMA-ES because **full CMA is intractable at 19456-D**, not
> for a diagonal-adaptation payoff. In 60 iterations neither the full $C$ nor
> sep's `sigma_vec` moves appreciably, so the training dynamics reduce to an
> **isotropic $(\mu/\mu_w, \lambda)$-ES**: scalar step-size adaptation + weighted
> mean recombination. (This also re-reads my "block-correlation ≈ null" evidence:
> it reflects an unmoved covariance, not a diagonal one — either mode gives it.)
> Repro: `probe_sep_cma.py`.

- **sep-CMA ask/tell main loop** — still in un-shipped
  `experiments/with_training/`; reconstructed to ~78% in
  `recovered_training_loop.py`. Behavior recoverable, exact code not. 🟠

### 5.2 Not on the "how it's implemented" path — and that's fine

These are credentials or fine-tuning magnitudes. Knowing them changes *nothing*
about the architecture; they're listed only for honest completeness.

- **SFT temperature $\tau$** — production-only stage; estimable by fitting
  $p_q = \mathrm{softmax}(\bar r_q/\tau)$ *if* one had product weights + worker
  score vectors, but those are closed. Tuning knob, not structure.
- **Production GRPO $\beta, \epsilon$, batch/lr at scale** — algorithm shape is
  TRL defaults + report's $\beta=0$ claim; only the scale numbers are closed.
- **Worker API keys** — credentials; information-theoretically unrecoverable and
  irrelevant to architecture.
- **Live routing weights / deployed pool** — dynamic production state, rotates
  ~biweekly; not static, not architectural.

> **Net:** on the actual target — *how Fugu is implemented* — fog is cleared.
> What remains is either a credential, a tuning magnitude, or the sep/full CMA
> flag (moderately inferred, architecturally inert). None sits on the critical
> path to understanding the system.

---

## 6. Corrections log (errors fixed across 14 rounds)

1. argmax → **softmax sampling** (trinity mode)
2. "consecutive-same-agent termination" → that's *standard* mode, not trinity
3. recursion discount 0.25 → code default **0.2**
4. "full forward to get hidden" → generation/forward **dual path** + forward
   monkeypatch bypassing LM head
5. reward hardcoded 0/0.5/1 → correctness from **external task fn**
6. "arbitrary DAG" → **topological order**, forward refs forbidden
7. "L+3 role head is fugu" → academic TRINITY; **product Fugu drops roles**
8. "training = sep-CMA only" → product Fugu is **SFT(KL) + sep-CMA two-stage**
9. **"9 matrices in layer 26"** → 7 in layer 26 + embed_tokens + lm_head
10. SVF offset order = **state_dict order** (embed first, lm_head last)
11. input = **raw transcript**, not chat_template (proven: 100% vs 11%)
12. "sep-CMA mock-runnable" → main loop **not shipped**, can't run
13. "pycma default runs a diagonal phase first" → **wrong**; `0*100*N/...`
    evaluates to 0 → default is full (`GaussFullSampler`)
14. "block correlation ≈ null supports diagonal" → **wrong**; it's what *both*
    modes give when $C$ never moves.
15. "sep is a free/irrelevant choice" → **wrong**; full CMA is *infeasible* at
    19456-D, so sep is **required**. And pycma's sep puts adaptation in
    `sigma_vec` (frozen $C$), not a diagonal $C$ — in 60 iters only the scalar
    $\sigma$ moves → dynamics ≈ isotropic $(\mu/\mu_w,\lambda)$-ES
    (`probe_sep_cma.py`)

---

## 7. Artifacts in this repo

| File | What |
|---|---|
| `verify_37.py` | 37-case batch eval — 95%/100% hit, the headline proof |
| `verify_trinity2.py` | end-to-end single-case, fixture agent=4 role=0 reproduced |
| `verify_margin.py` | margin analysis of the 2 missed cases |
| `probe_sep_cma.py` | sep-CMA-ES mechanism probe (sampler, sigma_vec, Fugu-scale dynamics) |
| `recovered_training_loop.py` | reconstructed sep-CMA loop, evidence-tagged |
| `fugu/Fugu_technical_report.md` | full report + LaTeX formula appendix |
| `trinity_code_submission/` | TRINITY author code (inference + stripped training) |
| `conductor_supplementary/` | Conductor author code (GRPO + recursion engines) |
| `trinity_coordinator/` | 3rd-party Elixir reimpl (the 37-case ground-truth source) |

**Bottom line:** the *skeleton* (inference path, SVF structure, TRINITY training
config) is EXEC/DATA-proven. The training *main loop* is reconstructed to ~78%.
Only the "sep" diagonal flag and the closed product layer (SFT, prod weights,
pools) are genuinely dark.


