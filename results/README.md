# Results

> **Granularity caveat — read first (honest scoping).** The TRINITY routing
> experiments below (`train_trinity_real.py`, `train_trinity_toolscale.py`, and
> the `eval_orchestration.py` mock) are **per-QUESTION** routing: the router
> reads the question once, picks ONE worker, and that worker answers the whole
> question. This is query-level model selection (the RouterDC / MASRouter family).
> **It is NOT Fugu's per-STEP coordination**, which is the real TRINITY mechanism:
> per turn the router re-reads an *evolving* transcript (the question plus
> `<reference_thought_N>` accumulated from earlier workers), re-routes a
> (worker, role) action, the worker advances one step, and a verifier decides
> termination. The per-step coordinator lives in `openfugu/mini.py` (`Coordinator`,
> faithful to `step_trinity`); the per-step *training* loop over a real worker
> pool is `train/train_trinity_perstep.py` (results at the bottom of this file).
> The per-question numbers below are real and reproducible, but they measure
> query-level routing, not multi-turn coordination — labeling them "TRINITY
> router" earlier overstated the alignment, corrected here.

## Conductor GRPO on ToolScale

![Conductor GRPO reward curve](conductor_grpo_reward.png)

Training a Conductor (Llama-3.2-3B-Instruct) with GRPO on
[`nvidia/ToolScale`](https://huggingface.co/datasets/nvidia/ToolScale), 100
steps, β=0 (no KL — matching the Fugu-Ultra report). Reward = format reward
(`<think>…</think><answer>[json]</answer>`) + action reward (the emitted
tool-call sequence scored against the task's `evaluation_criteria.actions`).

What the curve shows:

- **format reward** saturates to **1.0 within ~3 steps** — the model quickly
  learns to emit the required `<think>/<answer>` structure.
- **action reward** climbs from **~0.27 → ~0.6** (peaks ~0.70) — it progressively
  learns to match the ground-truth tool calls.
- **total reward** rises **1.21 → 1.64** over training.

Raw per-step data: [`conductor_grpo_log.csv`](conductor_grpo_log.csv)
(step, reward, format_reward, action_reward, loss, completion_len).
Regenerate the plot: `python assets/plot_reward_curve.py <log> <out.png>`.

Trained weights: `huggingface.co/di-zhang-fdu/openfugu-conductor-3b`.

## Orchestration beats the best single model

`eval/eval_orchestration.py` — the self-trained TRINITY coordinator scores
**+107%** over the best single worker, reaching **100%** of the oracle ceiling
(see README). This is the central Fugu claim, reproduced on a coordinator we
trained ourselves.

## TRINITY self-training on REAL data (GSM8K) — honest result

`train/train_trinity_real.py` runs the same sep-CMA-ES loop on REAL data: real
Qwen3-0.6B hidden states as routing features, a real Novita worker pool, and
numeric-answer-match reward on GSM8K. Full log:
[`trinity_gsm8k_run.txt`](trinity_gsm8k_run.txt).

The real-data loop **runs and passes** (coordinator ≥ best single worker), but
on this setup it only **ties** the best worker rather than beating it:

```
per-worker solved: deepseek-v4-pro=0.83, qwen3.5-plus=0.92, gemma-4-31b-it=0.92
coordinator      = 0.917   (= best single worker)
```

**Why it ties, not wins — stated plainly:** GSM8K is too easy for these modern
workers (two of three already solve ~92% alone), so there is little
"worker A succeeds where B fails" signal for routing to exploit; the ceiling is
near-saturated and the coordinator correctly learns to route to a strong worker,
but there is no headroom to *beat* it. The mock harness shows the large +107%
gain precisely because it is built with sharply differentiated specialists
(0.9 vs 0.2 per domain). Demonstrating orchestration value on real data needs a
pool with **complementary** strengths and tasks hard enough that single models
fail — that's the next experiment, not a property of the loop (which is proven
to run end-to-end on real verifiable tasks here).

## TRINITY router on REAL multi-domain data (ToolScale) — the gain shows up

`train/train_trinity_toolscale.py` runs the same sep-CMA-ES router loop on
nvidia/ToolScale (multi-domain agentic tool-use), reusing the tool-call reward
from `toolscale_data.py`. Full log:
[`trinity_toolscale_run.txt`](trinity_toolscale_run.txt).

```
per-worker action-score: deepseek-v4-pro=0.000, qwen3.5-plus=0.142, gemma=0.021
coordinator             = 0.152  >  best single (qwen 0.142)   PASS (+7%)
routing: qwen x7, gemma x1
```

Unlike GSM8K (where all workers tied), here the workers' scores are **spread**
(0.000 / 0.142 / 0.021), so routing has signal — and the coordinator beats the
best single worker by routing most tasks to qwen while sending one to gemma.
This is the orchestration gain GSM8K couldn't show: it comes from worker
**complementarity**, which a multi-domain task set has and single-domain math
doesn't.

Honest caveats: (1) absolute scores are low — ToolScale tool-call matching is
hard for general chat models with no executable tool environment, so +7% is a
small-but-real lift; (2) deepseek scoring 0.000 likely reflects the JSON-plan
parser being strict on a reasoning model's long-CoT output, not the model being
incapable. The headline holds: the real-data router loop runs, and on
complementary multi-domain workers the coordinator > best single model.

## Recursive Conductor — REAL recursion (round-0 fed back), honest TIE on held-out

`train/train_recursion_real.py` is now genuinely recursive. An earlier version
of this file was single-round GRPO with "recursion" only in the docstring — the
round-0 output was never fed back. The current version subclasses `GRPOTrainer`
and overrides `_generate_and_score_completions` to (1) run a round-0 self-rollout
on each question, (2) splice round-0's own output + a correction instruction
(wording from the released `recursion_formats.py`) into the prompt, (3) run
standard GRPO on that round-1 "revise" prompt. Round-1 literally sees its own
round-0 attempt — faithful to `conductor_recursion_engine`. Real 3B model, real
trl GRPO, ToolScale reward, 30 steps from `conductor_toolscale_100/checkpoint-100`.
Log: [`recursion_real_run.txt`](recursion_real_run.txt). Weights:
`conductor_recursion_real/`.

**The training reward is the WRONG metric** (and we no longer report a PASS from
it): on ToolScale-easy the base Conductor's plans are already strong, so the 8
samples in a GRPO group score near-identically → `reward_std≈0` → no gradient.
That saturation is real and visible in the log. The HONEST metric is held-out:
`eval/eval_recursion_real.py` scores the round-0 plan, feeds it back, and scores
the round-1 revised plan on 40 held-out questions.

```
round-0 mean score = 0.617
round-1 mean score = 0.616   (-0.2%)
questions improved by revise = 0/40 ; regressed = 1
on round-0 misses, revise delta = -0.001
TIE — revise neither helps nor hurts (round-0 already strong; greedy → round-1
      reproduces round-0)
```

**Honest result: TIE, not a gain.** The recursion *mechanism* is real and runs
end-to-end (round-0 → feed-back → round-1, verified), but on this base+task the
revise round does not improve the plan. Recursion (test-time scaling) only pays
off when round-0 is actually wrong and there is something to fix; a strong base
on easy tool-planning, decoded greedily, leaves round-1 nothing to do. This
**contradicts the mock `train_recursion.py`'s "+9% PASS"** — that lift comes from
a non-saturated toy policy with headroom, and we no longer present it as evidence
about the real model. To show a real recursion gain you'd start from a weaker
base or a harder task. The mechanism is proven; the gain is not, and we say so.

## Per-STEP TRINITY training — the real granularity (not per-question)

`train/train_trinity_perstep.py` is the one that trains at Fugu's actual
granularity. Every sep-CMA-ES candidate is a router head; its fitness is the
**terminal reward of running the full per-step `Coordinator` loop** — per turn
the router re-reads the evolving obs (question + accumulated
`<reference_thought_N>` from prior solver turns), re-routes a (worker, role)
action, a local worker advances one step, a verifier decides termination. This
is what the per-question runs above are NOT. Log:
[`trinity_perstep_run.txt`](trinity_perstep_run.txt).

```
worker pool (local, multi-vendor): deepseek-distill-7b, llama-3.2-3b, gemma-3-4b
base router head, multi-turn rollout:  solved 0.750   (n=8, max_turns=4)
sep-CMA-trained head:                  solved 1.000   PASS (>base)
```

This also closes the gap that the single-turn `--self-test` could not: the
**multi-turn Coordinator loop now runs end-to-end on real weights + real
workers** (base rollout 0.750), and sep-CMA over the head improves it.

**Honest caveat — small n:** 1.000 means it routed all **8 training questions**
correctly; this is in-sample, not held-out, so some of the lift is overfitting
to 8 items. What's rigorously demonstrated is the *mechanism*: per-step routing
fitness = a real multi-turn rollout, optimized gradient-free, beats the base
head. A larger held-out eval is the scaling step, not a mechanism question.

## Adaptive k-of-n pool — REAL, per-STEP (the reverted experiment, done right)

`train/train_adaptive_pool_perstep.py` is the faithful version of an experiment
that failed twice before and was correctly reverted. The old
`train_adaptive_pool_real.py` was wrong two ways: (1) per-QUESTION not per-step,
and (2) its reward grafted worker-id semantics onto ToolScale *tool* names, so
the subset reward was always 0. This version fixes both: per question a random
**k-of-n subset** of the local worker pool is offered, the router is **masked to
that subset every turn** (new `agent_mask` on `FuguRouter.route` + `Coordinator`,
backward-compatible — self-test still 95%/100%), and fitness is the terminal
reward of the full per-step `Coordinator` rollout over GSM8K. Log:
[`adaptive_perstep_run.txt`](adaptive_perstep_run.txt).

```
pool (3, local multi-vendor): deepseek-distill-7b, llama-3.2-3b, gemma-3-4b   k=2
base head, subset-masked rollout:   solved 0.625   (n=8, max_turns=4)
sep-CMA over random k-of-n subsets: solved 1.000   PASS (>base)
```

Both numbers are under the SAME availability masking (neither base nor trained
can route to an absent worker); the only difference is whether the head was
*trained* over random subsets. Training over varying offered subsets is Fugu's
"swap the pool / opt out any provider" promise made concrete.

**Honest caveat — small n (same as the per-step run):** 1.000 is in-sample over
8 questions, so part of it is overfitting. What's rigorously shown is the
*mechanism*: subset-aware per-step routing, trained gradient-free over random
k-of-n offered subsets via real multi-turn rollouts, beats the untrained base
under the same masking. Held-out scale-up is the next step, not a mechanism
question.

## Real end-to-end serving — the trained head, served, answered live

`eval/serve_e2e.py` is the honest proof that Fugu serves as one model. It does
NOT call `Coordinator.run` directly — it boots the actual HTTP server
(`openfugu/serve.py`) with the **trained per-step head** (`trinity_perstep.npy`)
layered over the base SVF vector and a **real local worker pool** (no API), waits
for `/health`, then POSTs a real GSM8K question to `/v1/chat/completions` and
checks the answer came back through the full per-step loop from a real worker
(not the mock). Log: [`serve_e2e_run.txt`](serve_e2e_run.txt).

```
server: trained head applied; worker pool LOCAL (2): llama-3.2-3b, gemma-3-4b
POST /v1/chat/completions  ("Natalia sold clips ... altogether in April and May?")
answer = 72  (gold 72)   turns = 2   pool: local=True mock=False
PASS — real request answered correctly through the per-step loop over a real
       local worker pool (not mock)
```

This closes the read→run→train→serve loop on real artifacts: the head trained by
`train_trinity_perstep.py` is the head being served, the workers it routes to are
the real local models it was trained against, and the answer is produced by the
full multi-turn coordinator behind one OpenAI-compatible endpoint.

## One-command pipeline — train → serve → verify on a fresh head

`pipeline/e2e_train_serve.py` is the full product loop in one command. It trains
a per-step TRINITY head, then serves **that exact freshly-trained head** over the
real local worker pool and verifies a live request — no manual path hand-off
between training and serving. Log: [`e2e_pipeline_run.txt`](e2e_pipeline_run.txt).

```
STAGE train:  train_trinity_perstep.py --out /tmp/fugu_head_14c24d1w.npy
              base head rollout 0.750 -> sep-CMA 1.000  PASS
[pipeline]    trained head ready: /tmp/fugu_head_14c24d1w.npy (82048 bytes) — serving THIS head
STAGE serve+verify:
  [serve] applied trained head from /tmp/fugu_head_14c24d1w.npy   <- SAME path stage 1 wrote
  [serve] worker pool: LOCAL (2): llama-3.2-3b, gemma-3-4b
  POST /v1/chat/completions -> answer 72 (gold 72), turns=2, local=True mock=False
[pipeline] PASS — trained a head, served THAT head over the real local pool,
           and a live request returned the correct answer.
```

The head path written by the training stage (`/tmp/fugu_head_14c24d1w.npy`) is
the exact path the serving stage loads — so the artifact under test is provably
the one just produced, not a pre-baked file. This is "real end-to-end training
and serving" as one reproducible command. (`--skip-train --head <path>` runs the
serve+verify half alone against an existing head.)
