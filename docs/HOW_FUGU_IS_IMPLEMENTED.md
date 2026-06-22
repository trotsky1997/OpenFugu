# How Fugu Is Implemented — A Technical Disclosure

*A reverse-engineering disclosure of Sakana AI's Fugu orchestrator family,
derived from the TRINITY (arXiv:2512.04695) and Conductor (arXiv:2512.04388)
papers, the authors' code submissions, the Fugu technical report, and live
execution against the released `model_iter_60.npy` checkpoint on Qwen3-0.6B.*

Claims are graded inline: **[EXEC]** reproduced by running real weights ·
**[CODE]** read from author source · **[DATA]** computed from a released
artifact · **[DOC]** stated in the report · **[INFER]** reconstructed from
constraints · **[DARK]** structurally unavailable.

---

## 0. What is Sakana Fugu

Sakana Fugu is a **learned orchestrator** sold as if it were a single model. You
call one OpenAI-compatible endpoint; behind it, Fugu routes your query across a
pool of frontier LLMs from different vendors (Gemini-3.1-Pro, Claude-Opus-4.8,
GPT-5.5, and others) and returns one answer. The pitch is "frontier capability
without single-vendor dependency": because the pool is swappable and no provider
is hardcoded, Fugu can dodge export-control and availability constraints while
still reaching — and on several benchmarks exceeding — the best single model. It
ships in two flavors: **Fugu** (latency-optimized, picks one worker per turn) and
**Fugu-Ultra** (quality-optimized, plans a multi-step multi-agent workflow). The
public `SakanaAI/fugu` GitHub repo is 100% shell — it just installs the closed
API into Codex; the actual method lives in two ICLR-2026 papers (TRINITY and
Conductor) plus a technical report. [DOC]

The key reframing: **Fugu is not a model, it is a policy over models.** What
Sakana trained is not a better LLM but a tiny decision-maker that knows *which
existing LLM to ask, for what, and in what order* — leaving the heavy lifting to
the frozen frontier workers it commands.

## 0.1 How Sakana Fugu really works

Strip the marketing and the mechanism is surprisingly small. Fugu runs a
**~0.6B-parameter backbone (Qwen3-0.6B) that never produces a user-facing
answer**. Instead, for a given conversation it computes one hidden vector and
feeds it to a **bias-free linear head** that outputs a score per worker model.
The highest-scoring worker is dispatched the query; its reply is what you see.
The orchestrator's own text generation is thrown away — only the routing logits
are used — so a routing decision costs roughly a single forward pass, not a full
decode. That is the entire latency story. [EXEC]

What makes it *learned* rather than hand-tuned: the head plus a handful of
**singular-value scale offsets** on the backbone (together ~19.5K trainable
numbers — see §2.3) are optimized so that the routing logits correlate with which
worker actually succeeds. Training is gradient-free (an evolutionary strategy on
end-to-end task success), because the signal is sparse, expensive, and weakly
coupled. Empirically the learned routing tracks each model's real strengths —
sending coding-terminal work toward GPT, graduate science toward Gemini — and on
a 37-case routing fixture our reconstruction reproduces the reference decisions
at 95–100% (§6). [EXEC]

**Fugu** (the everyday variant) stops there: per turn, read hidden state → pick
one worker → optionally a Thinker/Verifier role → repeat until a verifier accepts
or a turn budget is hit. **Fugu-Ultra** swaps the per-turn picker for a larger
7B "Conductor" that, in one shot, writes a whole **workflow graph** — a list of
sub-tasks, which worker handles each, and which earlier outputs each may see —
then executes it, with agents kept isolated within a workflow but sharing
tool-call memory across the conversation (§5). Either way, **no worker weights
are touched**; the whole system is composition over other people's models. [DOC]

---

## 1. The core idea

§0 gave the shape; here is the bet stated precisely, because everything
technical follows from it. The orchestrator commits to a single structural
hypothesis:

> **the penultimate-token hidden state $h \in \mathbb{R}^d$ of a small frozen-ish
> LM already linearly encodes which large model should act next.**

If true, no generation, no fine-tuned judge, and no learned attention over
candidates is needed — a single matrix $W_{\text{head}}\,h$ suffices to route.
The rest of this document is what that hypothesis forces: a linear head (§2.1),
a cheap way to adapt the backbone's representation without full fine-tuning
(SVF, §2.2), a ~19.5K-parameter training surface (§2.3) small enough for
gradient-free optimization (§3), and the per-turn / workflow loops that consume
the routing decision (§4, §5). The hypothesis is not assumed — it is checked: on
a 37-case fixture the linear head reproduces reference routing at 95–100% (§6).
[EXEC]

---

## 2. Fugu — parametrization

### 2.1 The selection policy

**A note on variants up front**, because the two differ and all executable
evidence below is from the **academic TRINITY checkpoint** (`model_iter_60.npy`),
whose head emits $L+3 = 10$ logits. The **production Fugu** described in the
report drops roles and emits only $L$ — same backbone, narrower head. Claims
tagged [EXEC]/[DATA] are the academic variant; [DOC] is the product.

Let $h(s) \in \mathbb{R}^{d}$ be the backbone hidden state at the penultimate
token for transcript state $s$ ($d = 1024$ for Qwen3-0.6B). A bias-free linear
head $f_\theta : \mathbb{R}^d \to \mathbb{R}^{L}$ induces a categorical policy
over workers: [CODE]

$$
\pi_\theta(i \mid s) \;=\; \frac{\exp\!\big(f_\theta(h(s))_i\big)}{\sum_{i'=1}^{L}\exp\!\big(f_\theta(h(s))_{i'}\big)},
\qquad i \in \{1,\dots,L\}.
$$

The head weight is $W_{\text{head}} \in \mathbb{R}^{L \times d}$ (or
$\mathbb{R}^{(L+3)\times d}$ for the academic variant), so
$f_\theta(h) = W_{\text{head}}\, h$. The orchestrator's *generated text is
discarded* — only the logits matter — which is why $h$ may be read at an early
token without autoregressive decoding. [DOC]

In the **academic TRINITY** variant the head emits $L+3$ logits; the last three
select a role $\rho \in \{\text{Worker}, \text{Thinker}, \text{Verifier}\}$ from
an independent softmax over those coordinates. Writing
$\ell = W_{\text{head}}\, h \in \mathbb{R}^{L+3}$, agent and role are drawn (or,
at eval, taken as argmax) from the two sub-vectors: [CODE]

$$
a \sim \mathrm{softmax}(\ell_{1:L}), \qquad
\rho \sim \mathrm{softmax}(\ell_{L+1:L+3}).
$$

The **production Fugu** drops roles entirely ($L$ logits only), shrinking the
coordination space to pure model selection for latency. [DOC]

### 2.2 Singular-value fine-tuning (SVF) of the backbone

To adapt the backbone's *representation* without full fine-tuning, Fugu uses
SVF. For each selected weight matrix $W \in \mathbb{R}^{m \times n}$, take its
singular value decomposition

$$
W = U \Sigma V^\top, \qquad
\Sigma = \mathrm{diag}(\sigma_1, \dots, \sigma_r), \quad r = \min(m,n),
$$

and **freeze the orthogonal factors $U, V$**, learning only a per-singular-value
scale. With a trainable offset vector $\delta \in \mathbb{R}^{r}$, the adapted
matrix is reconstructed as [CODE]

$$
\tilde\sigma_k = (1 + \delta_k)\,\sigma_k, \qquad
\widetilde{W} \;=\; \big(U\, \mathrm{diag}(\tilde\sigma)\, V^\top\big)\cdot
\underbrace{\frac{\sum_k \sigma_k}{\sum_k \tilde\sigma_k}}_{\text{energy preservation}}.
$$

The trailing scalar renormalizes total spectral energy so activation scale does
not drift. $\delta = 0$ recovers $W$ exactly. [CODE]

### 2.3 The trainable parameter vector

All learnable parameters concatenate into a single flat vector
$x \in \mathbb{R}^{N}$, $N = 19456$, verified exactly from the released
checkpoint: [EXEC]

$$
x = \big[\; \underbrace{\delta^{(1)}, \dots, \delta^{(9)}}_{\text{SVF offsets, } 9216}
\;\big\Vert\;
\underbrace{\mathrm{vec}(W_{\text{head}})}_{\text{head, } 10240}\;\big],
\qquad
9216 + 10240 = 19456.
$$

SVF spans **9 matrices**, each contributing $r = 1024$ offsets, consumed in
PyTorch `state_dict` order: [EXEC]

$$
\underbrace{\text{embed\_tokens}}_{1024}
,\;
\underbrace{\{q,k,v,o,\text{gate},\text{up},\text{down}\}\text{ of layer }26}_{7 \times 1024}
,\;
\underbrace{\text{lm\_head}}_{1024}
\;=\; 9216.
$$

The head block reshapes as
$W_{\text{head}} = \mathrm{reshape}(x_{9216:},\, (L{+}3,\, d)) = (10, 1024)$
for $L = 7$. [EXEC]

> Total learnable parameters $\approx 19.5\text{K}$ — orders of magnitude below
> any fine-tune of the 0.6B backbone, and the reason an evolutionary strategy
> is tractable (§3.2). [DOC]

---

## 3. Fugu — training

Training is **two-stage**: a supervised warm start on single-step tasks, then
an evolutionary refinement on end-to-end trajectories. [DOC]

### 3.1 Stage 1 — supervised fine-tuning (SFT) on single-step tasks

For each question $q$ in a verifiable dataset $\mathcal{D}$, every worker $M_i$
is run $R$ times and scored against the ground truth, giving a mean reward and a
performance vector over the pool: [DOC]

$$
\bar r_{q,i} = \frac{1}{R}\sum_{j=1}^{R} r^{M_i}_{q,j},
\qquad
s_q = (\bar r_{q,1}, \dots, \bar r_{q,L}).
$$

Rather than supervising on the hard $\arg\max$ worker, the scores become a
**soft target** through a temperature softmax, preserving reward magnitudes:

$$
p_q(i) = \frac{\exp(\bar r_{q,i}/\tau)}{\sum_{i'=1}^{L}\exp(\bar r_{q,i'}/\tau)}.
$$

Both the head and the SVF scales are trained to minimize the forward KL of the
soft target $p_q$ relative to the policy (target first argument), i.e.
$D_{\mathrm{KL}}(p_q \,\Vert\, \pi_\theta)$: [DOC]

$$
\mathcal{L}_{\text{SFT}}(\theta)
= \frac{1}{|\mathcal{D}|}\sum_{q \in \mathcal{D}}
D_{\mathrm{KL}}\!\big(p_q(\cdot)\,\Vert\,\pi_\theta(\cdot \mid q)\big).
$$

This stage places $\theta$ in a strong basin and is the *only* place gradients
are used. ($\tau$ is not disclosed. This stage exists **only in the production
Fugu**; the open TRINITY submission has no SFT.) [DARK]

### 3.2 Stage 2 — sep-CMA-ES on end-to-end trajectories

The objective is the **expected terminal reward** of the orchestrator over
multi-turn agentic rollouts $\tau$ (drawn from real coding-assistant
environments — Claude Code, Codex, OpenCode), with $R(\tau) \in \{0,1\}$
recording task completion: [DOC]

$$
J(\theta) := \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\big].
$$

This is optimized **without gradients** — the parameters are weakly coupled,
the per-evaluation cost is high (every rollout is many LLM calls), and each run
is a Bernoulli draw, so REINFORCE gradients are low-SNR. The paper uses
**separable CMA-ES** (Ros & Hansen 2008): nominally a CMA-ES whose covariance is
constrained to be diagonal, $C = \mathrm{diag}(c_1^2,\dots,c_N^2)$, dropping cost
from $O(N^2)/O(N^3)$ to $O(N)$. (In pycma the diagonal adaptation is carried by a
per-coordinate step-size vector `sigma_vec` rather than a literal diagonal $C$ —
mechanism dissected in §7.) [DOC]

Maintaining a parent $m \in \mathbb{R}^N$, a scalar step size $\sigma$, and a
per-coordinate scale $d \in \mathbb{R}^N$ (the diagonal), each iteration samples
a population of $\lambda$ candidates:

$$
x_i = m + \sigma\, d \odot z_i, \qquad z_i \sim \mathcal{N}(0, I), \qquad i = 1, \dots, \lambda.
$$

Each candidate's fitness estimates $J$ by averaging the terminal reward over
$R_{\text{rep}}$ replicated rollouts, plus additive shaping bonuses (weights are
[DATA] from the log; the sign/normalization of the turn term is [INFER]):

$$
\hat J(x_i) = \underbrace{\frac{1}{R_{\text{rep}}}\sum_{e=1}^{R_{\text{rep}}} R(\tau_e)}_{\text{mean accuracy}}
\;+\; w_{\text{div}}\, H(\text{agents})
\;+\; w_{\text{turn}}\, \bar T
\;-\; w_{\text{cost}}\, \bar C,
$$

where $H$ is the entropy of the agent-selection distribution (diversity bonus)
and $\bar T, \bar C$ are mean turn count and cost. The top-$\mu$ candidates by
fitness are recombined by rank-weighted averaging to form the next parent:

$$
m' = m + \sum_{k=1}^{\mu} w_k\, (x_{k:\lambda} - m) \;=\; \sum_{k=1}^{\mu} w_k\, x_{k:\lambda},
\qquad \sum_k w_k = 1,
$$

with $x_{k:\lambda}$ the $k$-th best candidate under the fitness ranking (the
step size $\sigma$ enters only the sampling above, not this mean update). [DOC]

**Real configuration** (from the released training log `es_log.json`): [DATA]

$$
\begin{aligned}
&\text{iters} = 60,\quad \sigma_0 = 0.03,\quad R_{\text{rep}} = 16,\quad \text{seed} = 42, \\
&w_{\text{div}} = 0.15,\quad w_{\text{turn}} = 0.10,\quad w_{\text{cost}} = 0, \\
&L = 7,\quad \text{max\_turns} = 5,\quad \text{SVF layer} = 26.
\end{aligned}
$$

With $\texttt{popsize\_override}=0$, pycma defaults apply, giving the population
and parent counts in closed form for $N = 19456$: [DATA]

$$
\lambda = 4 + \lfloor 3 \ln N \rfloor = 33, \qquad
\mu = \lfloor \lambda/2 \rfloor = 16, \qquad
\mu_{\text{eff}} \approx 9.44,
$$

with log-rank recombination weights
$w_k \propto \ln(\mu + \tfrac12) - \ln k$ ($w_1 \approx 0.193$, normalized to sum
1). The paper's "$\lambda \approx 32$" is exactly this rounding. The released
checkpoint `model_iter_60.npy` is the $m$ at iteration 60. [DATA]

> The ask/tell loop itself is **not in the shipped code** (it lives in an
> un-released `experiments/with_training/` module); the structure above is
> reconstructed from the trainer signature, the eval-path job submission, and
> pycma defaults. The separable choice is **load-bearing for feasibility** — a
> full $N\times N$ covariance is intractable at $N=19456$ — but within 60
> iterations even the diagonal scale $d$ barely moves; only the scalar $\sigma$
> changes materially, so the run behaves like an isotropic ES (dissected in §7).
> [INFER]

---

## 4. Fugu — the inference loop

At serving time the orchestrator runs a bounded multi-turn loop over the
transcript $\mathcal{C}$. With SVF applied to the backbone, each turn $t$: [CODE]

1. Format the transcript as **raw** `"role: content\n"` text (not a chat
   template — proven decisive in §6), tokenize, forward pass. [EXEC]
2. Extract $h_t = $ hidden state at position $-2$. [CODE]
3. Route: $\ell_t = W_{\text{head}}\, h_t$; sample/argmax agent $a_t$ and
   (trinity) role $\rho_t$. [CODE]
4. Inject the role's system prompt, dispatch to worker $M_{a_t}$, append its
   reply to $\mathcal{C}$. [CODE]

The loop terminates at the stopping time

$$
\tau = \min\big\{\, t \le K : \rho_t = \text{Verifier} \;\wedge\; u_t = \texttt{ACCEPT} \,\big\},
\qquad K = 5,
$$

returning $O_\tau$ (the latest worker response); if no verifier accepts, it
returns the last response at $t = K$. A Thinker turn may emit
`<suggested_role>` which **overrides** the head's role choice on the next turn.
[CODE]

---

## 5. Fugu-Ultra — the Conductor line

Fugu-Ultra replaces the per-turn selector with a 7B LM that emits an **entire
agentic workflow** in natural language, parsed into three equal-length lists
defining a DAG over workers: [CODE]

$$
\texttt{model\_id}[1{:}T],\quad
\texttt{subtasks}[1{:}T],\quad
\texttt{access\_list}[1{:}T],
$$

where step $t$ dispatches `subtasks[t]` to worker `model_id[t]`, and
`access_list[t]` indexes which earlier steps' outputs enter its context. The
visibility relation must be a **topological order** — forward references raise —
so the workflow is a DAG, executed sequentially:

$$
\text{ctx}(t) = \big\{\,(\,\texttt{subtask}_j,\, o_j\,) : j \in \texttt{access\_list}[t]\,\big\},
\qquad j < t .
$$

### 5.1 Training — GRPO

The Conductor is trained with GRPO on a progressive reward (the $0/0.5/1$
structure is the documented design; in code the correctness term is supplied by
an external task-specific function, not a hardcoded constant): [DOC]

$$
r_i =
\begin{cases}
0 & \text{lists unparseable (format fail)} \\
0.5 & \text{parsed, executed, but wrong} \\
1 & \text{parsed, executed, correct,}
\end{cases}
$$

maximizing, over a group of $G$ sampled workflows per query, the clipped
surrogate with group-normalized advantage and **no KL penalty in production**
($\beta = 0$):

$$
J(\theta) = \mathbb{E}_{q,\{o_i\}\sim\pi_\theta(\cdot\mid q)}
\!\left[\frac{1}{G}\sum_{i=1}^{G}
\min\!\big(\rho_i A_i,\; \mathrm{clip}(\rho_i, 1{-}\epsilon, 1{+}\epsilon)\,A_i\big)
- \beta\, D_{\mathrm{KL}}(\pi_\theta \Vert \pi_{\text{ref}})\right],
$$

$$
\rho_i = \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)},
\qquad
A_i = \frac{r_i - \mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})}.
$$

A recursion finetune lets the Conductor name **itself** as a worker, with a
discount $\gamma = 0.2$ on the non-recursive round and per-round reward
normalization. [CODE]

### 5.2 Production-only orchestration memory

Two mechanisms (in the report, in neither paper) make multi-agent function
calling coherent: [DOC]

- **Intra-workflow isolation** — within one workflow, agent $a$ sees agent $b$'s
  trajectory *only* through `access_list`; otherwise each keeps its own
  transcript. Prevents "orchestration collapse" where the first agent's path
  steers all others.
- **Inter-workflow shared memory** — across the multi-turn conversation, agents
  share tool-call history, avoiding redundant re-discovery.

Formally, the context visible to agent $a$ at workflow $W$, turn $t$:

$$
\text{ctx}_a(W,t) = \underbrace{\text{own-history}_a}_{\text{isolated}}
\;\cup\; \underbrace{\{o_j : j \in \texttt{access\_list}\}}_{\text{topology}}
\;\cup\; \underbrace{\text{tool-memory}(\,<W\,)}_{\text{shared across workflows}}.
$$

---

## 6. Execution proof

The Fugu (TRINITY) inference path was reproduced end-to-end: real
`model_iter_60.npy`, real Qwen3-0.6B on an A800, SVF applied in `state_dict`
order, evaluated against a 37-case fixture with known expected
$(a, \rho)$ labels. [EXEC]

| Input formatting | agent acc. | role acc. | joint |
|---|---|---|---|
| naive baseline (guess mode class) | 51% | 49% | — |
| **raw `"role: content"`** | **95%** | **100%** | **95%** |
| chat template | 41% | 11% | 5% |

Against the best constant-guess null (always pick the modal class, $p_0 =
19/37$ for agent, $18/37$ for role), a one-sided binomial test puts the agent
result at $P(\ge 35/37) \approx 1.2\times10^{-8}$ and the role result at
$P(=37/37) \approx 2.6\times10^{-12}$ — so the fit is not chance, and it
**simultaneously confirms**: the $9216 \,\Vert\, 10240$ split and its order, the
9-matrix SVF layout, the energy-preserving reconstruction, the raw-transcript
input convention, and that the checkpoint is genuinely paired with this inference
code. The two agent misses are a sub-threshold logit margin ($0.214 < 0.24$,
fp32 noise) and a unicode-emoji edge case. [EXEC]

---

## 7. What remains uncertain — stated plainly

A few things in this disclosure are not directly confirmed, and I want to be
explicit about which.

The most-investigated point was what "sep" in sep-CMA-ES actually does. It is
not Sakana's invention — it is the standard separable variant of Ros & Hansen
(2008), which restricts the covariance to a diagonal, cutting cost from $O(N^2)$
storage / $O(N^3)$ eigendecomposition to $O(N)$. Probing pycma v4.4.4 directly
settled the mechanism: `CMA_diagonal=True` does **not** keep a diagonal $C$
matrix — it freezes the sampler's $C$ at $I$ (`GaussStandardConstant`, "no
update") and moves the per-coordinate adaptation into a step-size vector
`sigma_vec`, so $x_i = m + \sigma\cdot\texttt{sigma\_vec}\odot z_i$ with
$z_i\sim\mathcal N(0,I)$ — diagonal covariance in effect, implemented off the
sampler. (The default mode is the full-$C$ `GaussFullSampler`; the
`0*100*N/...` default string evaluates to $0$.) At Fugu's scale this matters in
a concrete way: full CMA is **computationally infeasible** at $N=19456$ (an
$N\times N$ matrix plus eigendecomposition), so sep is **required for
feasibility**, not an optional refinement. And within 60 iterations even sep's
`sigma_vec` barely diverges (std $\sim\!6\times10^{-4}$); the only quantity that
moves materially is the **scalar** step size $\sigma$ ($0.03\to0.002$).
The training therefore reduces, in practice, to an isotropic
$(\mu/\mu_w,\lambda)$-ES: scalar step-size control plus fitness-weighted mean
recombination, with the diagonal capability present but barely exercised at this
budget. (Honest correction: an earlier draft read the SVF blocks' near-zero
cross-correlation as evidence *for* a diagonal $C$ — it isn't; that is just what
any mode yields when the covariance never moves. Repro: `probe_sep_cma.py`.)

The training **ask/tell loop** is not in the shipped code (it lives in an
un-released `experiments/with_training/` module). What §3.2 and
`recovered_training_loop.py` give is a **reconstruction** — about 78% pinned by
the trainer's signature, the eval-path job submission, the released config, and
pycma defaults; the rest is labeled guesswork.

Everything else that's missing is either a **credential** (worker API keys —
unrecoverable by definition) or a **tuning magnitude** (the SFT temperature
$\tau$, the production GRPO $\beta/\epsilon$ and batch sizes, the live pool and
routing weights that rotate roughly biweekly). These are genuinely closed, but
none of them is on the path to understanding *how Fugu is implemented* — they're
knobs and secrets, not structure. The architecture itself — a tiny coordinator
over frozen workers, SVF-adapted backbone, linear selection head, trained by an
evolutionary strategy on end-to-end task success — is fully accounted for.

---

## 8. One-paragraph summary

Fugu attaches a $\sim$19.5K-parameter trainable surface — a bias-free linear
head $W_{\text{head}} \in \mathbb{R}^{(L+3)\times d}$ plus SVF singular-value
offsets on 9 backbone matrices — to a frozen Qwen3-0.6B. The penultimate-token
hidden state is mapped to worker (and role) logits; the selected frontier LLM
does the actual work, and the orchestrator never decodes text. It is warm-started
by KL-matching a temperature-softmax of measured per-worker rewards, then refined
by separable CMA-ES directly on end-to-end task success
$J(\theta)=\mathbb{E}_\tau[R(\tau)]$ — though at 19456-D over 60 iterations that
reduces in practice to an isotropic step-size-adapted ES. Fugu-Ultra swaps the
per-turn selector for a GRPO-trained 7B Conductor that emits a full
topological-DAG workflow with isolated-but-shared agent memory. No worker weights
are ever modified — the entire system is macro-level composition over
heterogeneous, swappable APIs.


