# Design — e2e-train-serve-pipeline

## Context

Training (`train/train_trinity_perstep.py`) and serving (`openfugu/serve.py` +
`eval/serve_e2e.py`, archived as `e2e-serving`) both work and are tested
independently. What's missing is a single entrypoint that composes them so the
head served is the head just trained — proving train→serve→infer on one fresh
artifact in one command.

## Goals / Non-Goals

- **Goal**: one process that trains a head, serves *that* head over the local
  pool, verifies a live request. No manual path hand-off.
- **Goal**: a `--skip-train` path to reuse an existing head (fast re-runs).
- **Non-Goal**: changing training or serving internals — they are proven; this
  only orchestrates them.
- **Non-Goal**: scale/accuracy claims — the per-step run's small-n caveat stands;
  this proves the *pipeline wiring*, not a new benchmark.

## Decisions

- **Orchestrate via subprocess, not imports.** The pipeline shells out to the
  existing `train_trinity_perstep.py` (writing `--out <head.npy>`) and then to
  the existing `serve_e2e.py` (passing `--head <head.npy>`). Rationale: reuses
  the exact, already-verified entrypoints with zero refactor risk; each stage's
  stdout streams through for a transparent log. Alternative: import and call
  `main()` of each — rejected, it couples the pipeline to internal signatures
  and risks CUDA state leaking across train→serve in one process.

- **Fresh head path is generated, not hard-coded.** The pipeline picks a temp
  `--out` path for training and feeds the same path to serving, so the artifact
  under test is provably the one just produced. Rationale: directly satisfies the
  spec's "head served is the head trained" requirement.

- **Reuse serve_e2e.py as the verification stage.** It already boots the server,
  waits for `/health`, POSTs a real question, and asserts answer + turns + not-
  mock. The pipeline delegates the serve+verify half to it wholesale. Rationale:
  one source of truth for "did the live request work," no duplicated HTTP logic.

## Risks / Trade-offs

- [Full pipeline is slow — training rollouts + worker loads] → `--skip-train`
  runs the serve+verify half in minutes against an existing head; the default
  path is the full proof.
- [A bad training run yields a weak head that fails verification] → That is the
  pipeline working as intended (non-zero exit surfaces it); the verify question
  is one the pool reliably solves, so a pass means the wiring and the head are
  both sound.
- [Subprocess GPU contention if training and serving overlap] → Stages run
  strictly sequentially; training fully exits (freeing its GPUs) before serving
  boots.

## Migration Plan

Purely additive: a new `pipeline/` entrypoint. No existing command changes. No
rollback needed beyond not running the pipeline.

## Open Questions

None blocking. Default verify question and worker pool mirror `serve_e2e.py`.
