"""sep-CMA-ES mechanism probe — what 'separable' actually means in pycma,
and how it behaves at Fugu's real scale. All results are reproducible.

Findings (executed against cma v4.4.4):
  1. sep mode does NOT use a diagonal C matrix. CMA_diagonal=True selects the
     GaussStandardConstant sampler (C frozen = I, "no update"); the diagonal
     adaptation lives in `sigma_vec` (per-coordinate step-size), which is
     mathematically equivalent to diagonal C but implemented separately.
       x_i = m + sigma * sigma_vec ⊙ z_i,   z_i ~ N(0, I)
  2. full mode (default) adapts the N×N C via GaussFullSampler; sigma_vec ≡ 1.
  3. At low dim on a separable ellipsoid, sep's sigma_vec genuinely diverges
     ([1..1] -> [0.22..0.06]) and beats full (3.8e-5 vs 6.2e-4).
  4. At Fugu scale (N=19456, pop=33, 60 iters): full is computationally
     infeasible (N×N matrix + eigendecomp -> OOM/stall). sep runs in ~7s but
     its per-coordinate sigma_vec divergence is negligible (std 6e-4, range
     0.996–1.002). The only thing that moves materially is the SCALAR sigma
     (0.03 -> 0.002). => training ≈ isotropic (μ/μ_w, λ)-ES; sep is REQUIRED
     for feasibility, but its diagonal capability is barely exercised in 60 iters.
"""
import cma, numpy as np, time

def sampler_of(opts):
    es = cma.CMAEvolutionStrategy([0.]*10, 0.3, {**opts, 'verbose': -9, 'maxiter': 1})
    return type(es.sm).__name__

print("[1] which sampler does each mode use?")
print("    full default     ->", sampler_of({}))
print("    CMA_diagonal=True ->", sampler_of({'CMA_diagonal': True}))

def run(opts, n=10, iters=80, seed=1):
    es = cma.CMAEvolutionStrategy([3.]*n, 1.0, {**opts, 'verbose': -9, 'seed': seed})
    f = lambda x: sum((i+1)*xi**2 for i, xi in enumerate(x))   # separable ill-conditioned
    for _ in range(iters):
        X = es.ask(); es.tell(X, [f(x) for x in X])
    return es.result.fbest, np.array(es.sigma_vec*np.ones(n))

print("\n[2] low-dim separable ellipsoid (n=10, 80 iters): where does adaptation live?")
for label, opts in [('full', {}), ('sep', {'CMA_diagonal': True})]:
    fb, sv = run(opts)
    print(f"    {label:4s} fbest={fb:.2e}  sigma_vec std={sv.std():.4f}  range[{sv.min():.3f},{sv.max():.3f}]")

print("\n[3] Fugu scale (N=19456, pop=33, 60 iters) — sep only (full is infeasible here):")
N, pop, iters = 19456, 33, 60
rng = np.random.default_rng(0); w = rng.exponential(1.0, N)
t = time.time()
es = cma.CMAEvolutionStrategy(np.zeros(N), 0.03,
        {'CMA_diagonal': True, 'popsize': pop, 'verbose': -9, 'seed': 1, 'maxiter': iters})
f = lambda x: float(np.sum(w*x*x))
for _ in range(iters):
    X = es.ask(); es.tell(X, [f(x) for x in X])
sv = np.array(es.sigma_vec*np.ones(N))
print(f"    wall={time.time()-t:.1f}s  scalar sigma 0.03->{es.sigma:.5f}  "
      f"sigma_vec std={sv.std():.5f} range[{sv.min():.3f},{sv.max():.3f}]")
print("    => scalar sigma is the mover; per-coord diagonal adaptation negligible in 60 iters.")
