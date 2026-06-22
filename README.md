# OpenFugu

**An open, runnable reverse-engineering of Sakana AI's Fugu — the "one model to
command them all" LLM orchestrator.**

Fugu is sold as a single model; it is really a *policy over models* — a tiny
coordinator that, per query, routes work to a pool of frontier LLMs and returns
one answer. Sakana's product and trained weights are closed. OpenFugu rebuilds
the mechanism from the two papers + released artifacts, verifies it against real
weights, trains a Conductor of our own, and serves it behind one OpenAI-compatible
endpoint. Four stages, all working: **read → run → train → serve.**

> Independent reimplementation. Not affiliated with Sakana AI. No third-party
> code/weights are redistributed here — `scripts/fetch_artifacts.py` pulls them
> from their licensed sources. See `NOTICE`.

## What's inside

| Stage | What | Evidence |
|-------|------|----------|
| **read** | `docs/HOW_FUGU_IS_IMPLEMENTED.md` — full math; `docs/ARCHITECTURE.md` — investigation log, evidence-graded | reverse-engineered from papers + author code |
| **run** | `openfugu/mini.py` (TRINITY: hidden-state → linear head → worker); `openfugu/ultra.py` (Conductor: workflow-DAG) | `mini.py --self-test` = **95% agent / 100% role** on the 37-case fixture, real weights |
| **train** | `train/train_trinity.py` — self-train the **TRINITY** coordinator from scratch via sep-CMA-ES (no Sakana weights); `train/train_conductor.py` — GRPO a **Conductor** on `nvidia/ToolScale` | TRINITY: chance→optimal routing in ~5 generations (mock, runs anywhere); Conductor: reward **1.21 → 1.64** over 100 steps ([curve](results/)) |
| **serve** | `openfugu/serve.py` — one OpenAI-compatible `/v1/chat/completions`; internal TRINITY loop over a litellm pool | `curl` returns one answer; pool hidden |
| **eval** | `eval/eval_orchestration.py` — does **per-question** routing beat the best single model? | trained router **+107%** over best single worker (query-level routing, **not** per-step coordination — see [results caveat](results/)) |

## Quickstart

```bash
pip install -r requirements.txt           # torch, transformers, trl, litellm, ...
python scripts/fetch_artifacts.py         # pull Qwen3-0.6B + model_iter_60.npy + fixture (not redistributed)

export FUGU_MODEL=$(...Qwen3-0.6B path...)
export FUGU_VECTOR=$PWD/artifacts/model_iter_60.npy
export FUGU_FIXTURE=$PWD/artifacts/qwen_router_prompt_eval_cases.json

# READ:  the architecture, evidence-graded
less docs/HOW_FUGU_IS_IMPLEMENTED.md

# RUN:   prove the reconstruction is faithful to the checkpoint
python openfugu/mini.py --self-test       # -> 95% / 100%

# RUN:   route one query (offline mock pool)
python openfugu/mini.py --demo

# RUN (live): real worker pool via litellm
export FUGU_API_KEY=...  FUGU_BASE_URL=...
python openfugu/mini.py --demo --live \
  --slot-models "novita/deepseek/deepseek-v4-flash,novita/zai-org/glm-5,..."

# TRAIN: a Conductor on ToolScale (8x A800-class; HF generation, no vLLM)
python train/train_conductor.py           # reward climbs off zero; saves checkpoint

# TRAIN: Fugu-Ultra recursive topology — Conductor revises its own output (test-time scaling)
python train/train_recursion.py           # mock: +9% over one-shot (toy policy w/ headroom)
python train/train_recursion_real.py      # REAL recursion (round-0 fed back into round-1)
python eval/eval_recursion_real.py        # honest held-out: round-0 vs round-1 → TIE (see results/)

# TRAIN: adaptive k-of-n pool — generalize to arbitrary worker subsets (swap the pool)
python train/train_adaptive_pool.py            # mock: +44% over blind, 94% of oracle
python train/train_adaptive_pool_perstep.py    # REAL per-step: random k-of-n subset masked each turn,
                                               # base 0.625 -> 1.000 (n=8, overfit caveat), PASS

# TRAIN: self-train the TRINITY coordinator from scratch (sep-CMA-ES, mock — no GPU/API)
python train/train_trinity.py             # chance -> optimal routing; PASS in seconds

# SERVE: Fugu as one model (API worker pool via litellm)
python openfugu/serve.py --slot-models "<csv>" --port 8088
curl localhost:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"flatten a nested list in one line"}]}'

# SERVE (real end-to-end): TRAINED per-step head + REAL local worker pool, no API
python openfugu/serve.py --model <qwen3-0.6b dir> --vector model_iter_60.npy \
  --head trinity_perstep.npy --local-models "<llama dir>,<gemma dir>" --port 8088
# prove it end-to-end (boots the server, POSTs a real GSM8K question, checks the answer):
python eval/serve_e2e.py --model <qwen3-0.6b dir> --vector model_iter_60.npy \
  --head trinity_perstep.npy --local-models "<llama dir>,<gemma dir>"   # -> answer 72, PASS

# PIPELINE: train -> serve -> verify in ONE command (the head served is the head just trained)
python pipeline/e2e_train_serve.py --model <qwen3-0.6b dir> --vector model_iter_60.npy \
  --local-models "<llama dir>,<gemma dir>" --port 8097      # trains a fresh head, serves it, PASS
# serve+verify only, reusing an existing head:
python pipeline/e2e_train_serve.py --skip-train --head trinity_perstep.npy \
  --model <qwen3-0.6b dir> --local-models "<llama dir>,<gemma dir>"

# EVAL: does orchestration beat the best single model? (the central Fugu claim)
python eval/eval_orchestration.py        # trained coordinator +107% over best single, PASS
```

## The mechanism in one breath

A ~0.6B backbone (Qwen3-0.6B) never answers the user. It produces one hidden
state at the penultimate token; a **bias-free linear head** scores each worker;
the top worker is dispatched and *its* reply is returned. ~19.5K trainable
numbers (the head + singular-value-fine-tuning offsets on 9 matrices), optimized
gradient-free (sep-CMA-ES). **Fugu-Ultra** swaps the per-turn picker for a 7B
Conductor that emits a whole workflow DAG. No worker weights are ever touched —
it is macro-level composition over other people's models. Full math, with an
EXEC/CODE/DATA evidence grade on every claim, in `docs/`.

## Trained Conductor weights

The Conductor we trained on ToolScale (a fine-tune of Llama-3.2-3B-Instruct) is
published on HuggingFace, **not** in this repo (Llama 3.2 Community License
applies — see `NOTICE`):

    huggingface.co/di-zhang-fdu/openfugu-conductor-3b   (see model card)

## License

Apache-2.0 for all OpenFugu code (`LICENSE`). Third-party material is fetched,
not redistributed; trained weights carry the Llama 3.2 license. See `NOTICE`.
