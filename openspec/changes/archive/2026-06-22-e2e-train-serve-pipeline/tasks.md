## 1. Pipeline entrypoint

- [x] 1.1 Create `pipeline/e2e_train_serve.py` that accepts router model dir,
  base vector, local-models CSV, GPU/device options, and training knobs
  (n-train, iters, max-turns); generate a temp `--out` head path for the run.
- [x] 1.2 Stage 1 (train): subprocess `train/train_trinity_perstep.py` writing
  the head to the generated path; stream its stdout; fail the pipeline non-zero
  if training exits non-zero or the head file is missing.
- [x] 1.3 Stage 2 (serve+verify): subprocess `eval/serve_e2e.py` with
  `--head <generated path>` + the local pool; stream its stdout; the pipeline's
  exit code mirrors serve_e2e's (pass only on a correct live answer).
- [x] 1.4 Add `--skip-train --head <path>`: skip stage 1 and run stage 2 against
  the provided existing head.

## 2. End-to-end verification

- [x] 2.1 Run the full pipeline (train → serve → verify) on the GPU server with
  the real local pool; capture the run log to `results/e2e_pipeline_run.txt`.
- [x] 2.2 Confirm the served head is the freshly-trained one (the log shows the
  training stage writing the head path that the serve stage then loads).

## 3. Documentation

- [x] 3.1 Update `README.md` with the one-command pipeline invocation (and the
  `--skip-train` variant).
- [x] 3.2 Add the pipeline run evidence to `results/README.md`.
