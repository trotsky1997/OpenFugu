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
| **train** | `train/` — GRPO a Conductor on `nvidia/ToolScale` with a verifiable tool-call reward | reward **0.70 → 1.70** over 100 steps |
| **serve** | `openfugu/serve.py` — one OpenAI-compatible `/v1/chat/completions`; internal TRINITY loop over a litellm pool | `curl` returns one answer; pool hidden |

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

# SERVE: Fugu as one model
python openfugu/serve.py --slot-models "<csv>" --port 8088
curl localhost:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"flatten a nested list in one line"}]}'
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

    huggingface.co/openfugu/openfugu-conductor-3b   (see model card)

## License

Apache-2.0 for all OpenFugu code (`LICENSE`). Third-party material is fetched,
not redistributed; trained weights carry the Llama 3.2 license. See `NOTICE`.
