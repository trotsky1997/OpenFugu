## Why

OpenFugu can now train a per-step TRINITY head (`train_trinity_perstep.py`) and
serve a trained head over a real local pool (`serve.py` + `serve_e2e.py`,
archived as `e2e-serving`). But these are still **separate manual steps**: a
human runs training, hand-copies the `.npy` path, then runs serving. There is no
single reproducible entrypoint that trains the head, serves *that freshly-trained
head* over the local pool, and verifies a live answer — which is what "real
end-to-end training and serving" means: one command, fresh artifact, proven
inference.

## What Changes

- A new pipeline entrypoint (`pipeline/e2e_train_serve.py`) that runs, in order:
  (1) per-step head training → a head `.npy`, (2) boot `serve.py` with that exact
  head + the local worker pool, (3) POST a real question and assert a real answer
  through the per-step loop. One process, no hand-copied paths.
- The pipeline SHALL pass the **freshly trained** head to serving (not a
  pre-baked file), so the artifact under test is the one just produced.
- A `--skip-train` escape hatch to reuse an existing head (fast re-runs / CI of
  the serve+verify half only), keeping the default path fully end-to-end.
- README + results updated with the one-command pipeline and its run evidence.

## Capabilities

### New Capabilities
- `e2e-train-serve`: a single reproducible pipeline that trains the per-step
  head, serves that exact head over the real local worker pool, and verifies a
  live request returns a real answer — proving train→serve→infer on one fresh
  artifact in one command.

### Modified Capabilities
<!-- None: e2e-serving (archived) covers serving a trained head; this composes
     training + that serving into one pipeline and does not change its requirements. -->

## Impact

- Code: new `pipeline/e2e_train_serve.py` orchestrating existing
  `train/train_trinity_perstep.py`, `openfugu/serve.py`, and the
  `eval/serve_e2e.py` verification logic. No changes to the training or serving
  internals (they already work and are tested).
- Docs: `README.md` (pipeline command), `results/README.md` (pipeline run
  evidence).
- Dependencies: none new — reuses the local worker pool + router already on the
  GPU server.
- Runtime: the full pipeline needs GPUs (training rollouts + resident workers);
  `--skip-train` reduces it to the serve+verify half for environments with a
  pre-trained head.
