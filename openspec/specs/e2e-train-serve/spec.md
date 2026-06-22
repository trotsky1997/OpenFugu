# e2e-train-serve Specification

## Purpose
TBD - created by archiving change e2e-train-serve-pipeline. Update Purpose after archive.
## Requirements
### Requirement: One-command train→serve→verify pipeline

The system SHALL provide a single pipeline entrypoint that trains a per-step
TRINITY head, serves that exact head over a real local worker pool behind the
OpenAI-compatible endpoint, and verifies a live request returns a real answer —
without any manual hand-off of artifact paths between steps.

#### Scenario: Default run trains then serves the fresh head

- **WHEN** the pipeline is invoked with no `--skip-train`
- **THEN** it SHALL run per-step head training to produce a head `.npy`, then
  boot the server with that exact freshly-produced head and the local worker
  pool, then issue a live request and assert a correct answer through the
  per-step loop
- **AND** the head served SHALL be the one produced by the training step in the
  same run, not a pre-existing file

#### Scenario: Pipeline reports a single pass/fail outcome

- **WHEN** the pipeline completes
- **THEN** it SHALL exit non-zero if any stage fails (training error, server not
  ready, wrong or missing answer) and zero only when the live request returns
  the expected answer through a real local worker

### Requirement: Skip-train escape hatch reuses an existing head

The pipeline SHALL support reusing a pre-trained head so the serve-and-verify
half can be run without repeating training, while the default path stays fully
end-to-end.

#### Scenario: Reuse a pre-trained head

- **WHEN** the pipeline is invoked with `--skip-train` and a path to an existing
  head `.npy`
- **THEN** it SHALL skip the training stage and serve the provided head over the
  local pool, then verify a live request as in the default path

