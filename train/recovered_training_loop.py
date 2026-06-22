# sep-CMA-ES Training Loop — Reconstructed Pseudocode
#
# This loop is NOT in trinity_code_submission (it lives in the un-shipped
# experiments/with_training/testing_standalone.py, per evaluate_*.py:8).
# Reconstructed from hard constraints. Evidence tags per line:
#   [SIG]  = forced by CMAEvolutionTrainer.__init__ signature (es.py:106-135)
#   [RUN]  = mirrors run_test() submit/collect structure (es.py:442-543)
#   [APPLY]= apply_params_to_model SVF reconstruction (trainer.py:1150-1221)
#   [LOG]  = real values in logs/ckpt/es_log.json
#   [PAPER]= TRINITY paper / report eqs 3-5 (J=E[R]; x=m+σz; weighted recomb)
#   [CMA]  = pycma library default behavior (popsize_override=0)
#   [INFER]= my inference from surrounding code; NOT directly verified

import cma                      # [SIG] import cma present at es.py:15
import numpy as np

class CMAEvolutionTrainer:
    def train(self):
        # ---- init SVF dims + flat param vector ----
        _, n_params, svd_weights = self._setup_svd_info()   # [RUN] es.py:382; n_params=19456
        x0 = np.zeros(n_params)                              # [INFER] offsets are +1.0 deltas -> start at 0 (=identity SVF), head at 0
        # ---- cma solver ----
        opts = {"seed": self.seed, "popsize": self.popsize_override or None}  # [SIG][LOG] seed=42
        # popsize_override=0 -> None -> pycma default lambda = 4 + floor(3 ln n) = 33  [CMA]
        self.solver = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)  # [PAPER][LOG] sigma0=0.03
        #   NOTE: paper says "sep-CMA-ES" (Ros & Hansen 2008, diagonal C), enabled
        #   via {"CMA_diagonal": True}. But it's MOOT here: at n=19456, pycma's
        #   covariance LRs are c1~5.3e-9, cmu~4.1e-8, so over 60 iters total
        #   covariance update ~ (c1+cmu)*60 ~ 0, and accumulated rank 60*33=1980
        #   << 19456. C stays ~I regardless of the flag -> sep and full give
        #   indistinguishable end-states. The result is driven by step-size
        #   adaptation + mean recombination, not covariance structure. [DATA, verified]

        for it in range(self.num_iters):                     # [LOG] num_iters=60
            solutions = self.solver.ask()                    # [CMA][PAPER eq4] x_i = m + sigma * z_i, i=1..lambda
            fitnesses = []
            for cand in solutions:                           # evaluate each of the ~33 candidates
                fitnesses.append(self._evaluate_candidate(cand, svd_weights, it))
            # pycma MINIMIZES -> negate (we maximize expected reward) [CMA][PAPER eq3]
            self.solver.tell(solutions, [-f for f in fitnesses])   # [PAPER eq5] weighted recomb of top-mu

            best = max(fitnesses)                            # [RUN] best_score tracking es.py:261,329
            if best > self.best_score:
                self.best_score = best
                self.best_solution = solutions[int(np.argmax(fitnesses))]
            if it % self.save_interval == 0:                 # [SIG] save_interval=1
                np.save(f"model_iter_{it}.npy", self.solver.result.xfavorite)  # [INFER] matches model_iter_60.npy artifact
            if it % self.test_interval == 0:                 # [SIG][LOG] test_interval=5
                self.run_test(self.best_solution)            # [RUN] es.py:442

    def _evaluate_candidate(self, flat_params, svd_weights, it):
        # ---- submit num_repeats episodes, average terminal reward ----  [PAPER eq3] fitness = E[R(tau)]
        futures = []
        rng = np.random.RandomState(self.seed + it)          # [INFER] seeded task sampling, cf run_test es.py:485
        for _ in range(self.num_repeats):                    # [LOG] num_repeats=16
            tid = rng.randint(0, self.infra.train_dataset_size)   # [INFER] train split (run_test uses test split)
            futures.append(self.job_manager.submit_training_job(  # [RUN] es.py:494 exact call
                task_id=int(tid),
                split="train",                               # [INFER] mirror of run_test split="test"
                flat_params=flat_params.astype(np.float32),  # [RUN] es.py:497
                svd_weights_cpu=svd_weights,
                iteration_idx=it,                            # [RUN] (run_test passes -1)
                eps_explore=self._eps(it),                   # [INFER] run_test sets 0.0; training likely anneals exploration
                servers_dict=self.servers,
                use_structured_router=self.use_structured_router,
                closed_model_config=self.closed_model_config,
                agent_configs=self.agent_configs))
        # inside the worker (trainer.py): apply_params_to_model() rebuilds SVF then runs TRINITY rollout
        #   SVF rebuild [APPLY] trainer.py:1183-1191:
        #     scale = flat[off:off+k] + 1.0
        #     newW  = (U @ diag(S*scale) @ V^T) * (S.sum() / (S*scale).sum())   # energy-preserving
        #   then head = flat[9216:].reshape(num_agents+3, hidden)              # [APPLY] trainer.py:1207
        #   rollout = step_trinity loop, terminal reward R in {0,1}            # [PAPER] core.py:879

        results = [f.get(timeout=600) for f in futures]      # [RUN] es.py:515
        # result tuple: (score, _, _, agent_ids, response, token_stats, _, _)  [RUN] es.py:521,536,539,542
        scores    = [r[0] for r in results if r[0] != -1.0]  # [RUN] -1.0 = infra failure, excluded
        agent_ids = [a for r in results if r[0] != -1.0 for a in r[3]]
        episodes  = [r[3] for r in results if r[0] != -1.0]

        base = float(np.mean(scores)) if scores else -1.0    # [RUN] es.py:543 mean accuracy

        # ---- reward shaping (additive bonuses) ----  [SIG][LOG] weights below
        div  = _calculate_diversity_metrics(agent_ids, self.infra.num_agents)["entropy"]  # [RUN] es.py:46
        turn = np.mean([len(e) for e in episodes]) if episodes else 0.0                    # [INFER] turn count proxy
        fitness = (base
                   + self.diversity_bonus_weight * div       # [LOG] 0.15
                   + self.turn_bonus_weight      * turn       # [LOG] 0.10  [INFER] sign/normalization unverified
                   - self.cost_bonus_weight      * 0.0)       # [LOG] 0.0 (disabled this run)
        return fitness

    def _eps(self, it):
        return 0.0   # [INFER] pure guess; run_test uses 0.0, training anneal schedule NOT in any shipped file
