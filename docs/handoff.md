# HANDOFF — openfugu reverse-engineering

Picking this up cold? Read this first. It tells you what was done, where the
truth lives, what to trust, and what's still open. It does **not** repeat the
findings — those are in `HOW_FUGU_IS_IMPLEMENTED.md` (the clean disclosure) and
`ARCHITECTURE.md` (the investigation log with evidence grades + a 15-entry
corrections log).

## What this is

A reverse-engineering of **Sakana Fugu** — a learned LLM-orchestrator product —
down to executable, weight-verified detail. Fugu = two variants: **Fugu**
(TRINITY paper, arXiv:2512.04695) and **Fugu-Ultra** (Conductor paper,
arXiv:2512.04388). We had: both papers, both authors' code submissions, the
technical report PDF, a third-party Elixir reimpl, and one released checkpoint.

## Read in this order

1. `HOW_FUGU_IS_IMPLEMENTED.md` — start here. §0/§0.1 = plain-language "what is
   it / how does it really work"; §1+ = the math, evidence-graded inline.
2. `ARCHITECTURE.md` — the same facts as an audit trail: evidence grades, the
   DARK-side analysis (§5), and a corrections log (§6) of every wrong turn taken
   and fixed. Read §6 before trusting any single claim — it shows the failure modes.
3. `fugu/Fugu_technical_report.md` — the product report, full text + a LaTeX
   formula appendix (formulas were screenshot-recovered; text extraction dropped them).

## Source material (the four repos)

| Dir | What it is | Trust |
|---|---|---|
| `trinity_code_submission/` | TRINITY authors' code. **Inference path is real; training loop is stripped** (lives in un-shipped `experiments/with_training/`). Holds `logs/ckpt/models/model_iter_60.npy` — the actual trained router. | high (for inference) |
| `conductor_supplementary/` | Conductor authors' code. GRPO + recursion engines present. | high |
| `trinity_coordinator/` | 3rd-party **Elixir** reimpl. Source of the **37-case routing fixture** we validated against (`examples/fixtures/qwen_router_prompt_eval_cases.json`). | medium — its README numbers (e.g. the "9 SVF tensors") are correct but its framing cost us rounds; verify against authors' code |
| `fugu/` | The public product repo: 100% shell installer + the technical report PDF. No architecture code. | n/a |

## The one artifact that matters most

`trinity_code_submission/logs/ckpt/models/model_iter_60.npy` — a **19456-float**
vector, the trained router. Everything EXEC-grade traces back to loading this and
running it on a real Qwen3-0.6B. If you verify nothing else, verify this:
`[:9216]` = SVF singular-value offsets (9 matrices × 1024), `[9216:]` = the
`(10,1024)` linear head.

## How to reproduce (the verification scripts)

Scripts assume a real Qwen3-0.6B + torch. **Local caveat:** the `.venv/` here is
Python 3.14, which has **no torch wheel** — torch work was done on a GPU box, not
locally. numpy-only checks (vector split, sep-CMA probe) run locally.

| Script | Needs | Proves |
|---|---|---|
| `verify_37.py` | GPU + Qwen3-0.6B | **headline:** 37-case routing 95% agent / 100% role (vs 51% baseline) |
| `verify_trinity2.py` | GPU + Qwen3-0.6B | single-case end-to-end, reproduces fixture agent=4/role=0 |
| `verify_trinity.py` | GPU + Qwen3-0.6B | first-pass structural check (superseded by verify_37) |
| `verify_margin.py` | GPU + Qwen3-0.6B | explains the 2 missed cases (margin 0.214<0.24, unicode edge) |
| `probe_sep_cma.py` | numpy + `cma` only | sep-CMA-ES mechanism + Fugu-scale dynamics (runs locally) |
| `recovered_training_loop.py` | reading only | reconstructed (~78%) training loop, evidence-tagged per line |

**GPU box used:** `ssh -p 8903 root@115.190.235.210` (8×A800-80GB, Python 3.12,
torch 2.11+cu129). Model pre-downloaded at
`/vePFS-Mindverse/share/huggingface/Qwen3-0.6B`. Repro flow: scp a verify script
+ `model_iter_60.npy` to `/root/`, edit `MODEL` to the local path, run. (The box
is borrowed — confirm it's still yours before relying on it.)

To run `verify_37.py` you also need the fixture:
`trinity_coordinator/examples/fixtures/qwen_router_prompt_eval_cases.json`.

## What to trust vs not

- **Trust (EXEC/DATA):** the entire Fugu inference path — vector split & order,
  9-matrix SVF layout, energy-preserving reconstruction, position-−2 hidden,
  raw-`"role: content"` input (NOT chat template), softmax-sample/argmax routing,
  verifier/max-turn termination. All reproduced at 95–100% on real weights.
- **Trust (DATA):** TRINITY training hyperparameters (from `es_log.json`), CMA
  λ=33/μ=16/μ_eff≈9.44 (matches `cma` v4.4.4).
- **Believe with care (DOC):** everything about the *product* (two-stage SFT+ES,
  Fugu-Ultra's memory model, the Model Card numbers) — stated in the report, no
  code to confirm. Product Fugu also differs from academic TRINITY (drops roles:
  L logits vs L+3).
- **Reconstruction (INFER):** the training ask/tell loop. ~78% constraint-pinned.
- **Do not chase (DARK / out of scope):** worker API keys (credentials), exact
  SFT τ / production GRPO β,ε / live pool weights (tuning magnitudes + dynamic
  state). None is on the path to understanding the architecture. Confirmed: the
  sep-CMA-ES "diagonal" question is resolved — at N=19456/60 iters the covariance
  barely moves, so training ≈ isotropic step-size ES regardless (see HOW §7).

## Known traps (learned the hard way — full list in ARCHITECTURE §6)

- SVF is **7 layer-26 matrices + embed_tokens + lm_head**, not "9 in layer 26".
- Router input is **raw transcript**, not a chat template (chat template scores
  11% role acc — looks broken, is just wrong formatting).
- pycma's `CMA_diagonal` default evaluates to **0** (full, not diagonal); and at
  this scale full vs sep is moot because the covariance never moves.
- `import cma` in `es.py` is a **dead reference** — the training loop isn't shipped.

## Open directions (if continuing)

1. **Build a minimal runnable Fugu** from the verified constants (single-file
   Python: Qwen3-0.6B + forward monkeypatch + position-−2 + linear(10,1024) head
   + softmax routing + verifier/max-turn loop). All constants are EXEC-confirmed;
   this is the natural next deliverable and was never built.
2. **Conductor/Fugu-Ultra has no execution proof yet** — only code-read + report.
   Reproducing a workflow-DAG execution would raise it from CODE/DOC to EXEC.
3. If product weights ever surface, the SFT temperature τ becomes fittable from
   `p_q = softmax(r̄_q/τ)`.

## Status

Architecture understanding: **complete to the reachable boundary.** Inference is
weight-verified; training config is data-grounded; the training loop is
reconstructed; the rest is either closed (credentials/tuning) or proven
irrelevant (sep flag). Nothing actionable remains dark.
