from __future__ import annotations
from iohblade.problem import Problem
from iohblade.solution import Solution
import math
import os, time, traceback
import numpy as np
from collections import deque, Counter, defaultdict
import random, itertools, heapq

class BestSoFarWrapper:
    def __init__(self, problem):
        self.p = problem
        self.best = float("inf")
        self.best_curve = []

    def __getattr__(self, name):
        return getattr(self.p, name)

    def __call__(self, x):
        # Reject nested / ragged / object arrays early
        arr = np.asarray(x)

        # If arr becomes dtype=object, it's ragged / nested → reject
        if arr.dtype == object:
            raise ValueError(f"Ragged/object x passed to problem. Example element: {arr.flat[0]!r}")

        # Must be 1D after flatten
        arr = np.asarray(arr, dtype=float).reshape(-1)

        dim = len(self.p.bounds.lb)
        if arr.size != dim:
            raise ValueError(f"x size {arr.size} != expected dim {dim}")

        y = float(self.p(arr))
        if y < self.best:
            self.best = y
        self.best_curve.append(self.best)
        return y

# could be passed directly but this works fine
def load_kmeans_min_baselines(path="KMeansCompare.csv"):
        import pandas as pd
        df = pd.read_csv(path)
        return {(int(r.dataset), int(r.k)): float(r.mse) for r in df.itertuples(index=False)}

class IOHClusteringProblemAdapter(Problem):
    def __init__(
        self,
        logger=None,
        training_instances=None,   # list[(fid, iid, k)]
        test_instances=None,       # ^ and use these values to get clustering problem
        name="IOH_CLUSTERING",
        data_set_name = None,  # for analysis
        eval_timeout=1000,
        base_budget=1000,
        budget_scale_with_dim=True,
        # This might be good to have and change later 
        full_ioh_log=True,
        ioh_dir=None,
        dependencies=None,
        imports=None,
    ):
        # To fix the dependency headache & the issue of BLADE not having the right imports - rm -rf /tmp/blade_env_*
        # you removed numpy might be an issue
        if dependencies is None:
            dependencies = ["git+https://github.com/IOHprofiler/iohclustering.git", "ioh", "pandas"]
        if imports is None:
            imports = "import numpy as np\nimport ioh\nimport iohclustering\nimport pandas\n"

        super().__init__(logger, training_instances, test_instances, name, eval_timeout, dependencies=dependencies,
    imports=imports,)


        # maybe this works maybe it does not 
        self.training_instances = training_instances
        self.test_instances = test_instances
        self.data_set_name = data_set_name
        self.base_budget = int(base_budget)
        self.budget_scale_with_dim = bool(budget_scale_with_dim)
        self.name = name
        self.eval_timeout = eval_timeout
        
        self.full_ioh_log = bool(full_ioh_log)
        self.ioh_dir = ioh_dir

        self.kmeans_min = load_kmeans_min_baselines("KMeansCompare.csv")

        # LLaMEA / iohblade convention fields
        self.func_name = "__call__"
        self.init_inputs = ["budget"]      
        self.func_inputs = ["func"]
        self.func_outputs = ["f_opt", "x_opt"]

        # current eval has provided domain knowledge on representation of clusters
        # and is trying to ensure diveristy of algorithms

        self.role_prompt = "You are an excellent Python programmer."

        self.task_prompt = r"""You are designing a black-box optimization algorithm for an IOH clustering problem (minimizing MSE).

        INTERFACE:
        - Implement a class with __init__(self, budget, dim=None, seed=None) and __call__(self, func).
        - func(x) evaluates a 1D NumPy float64 array and returns a Python float MSE (lower is better).
        - Infer dim from len(func.bounds.lb). Enforce bounds with np.clip(x, func.bounds.lb, func.bounds.ub).
        - Do NOT instantiate the problem yourself. Never call func(x) after budget is exhausted.
        - Use // or int(...) for any integer. Before reshape, verify compatibility with x.size.

        ALGORITHM:
        - Base your design on exactly one of: PSO, DE, UMDA, or ES.
        - You may extend or adapt it, but the core mechanism must remain recognisable.

        OUTPUT: Return (f_opt, x_opt)."""

        # order of the example prompts
        # [RandomSearch, NeighborhoodAdaptiveDE, ParticleSwarmOptimiser, UMDA, GreedyHillClimber]
        
        # Uncomment for reg mutation prompts
        
        # self.mutation_prompts = [
        #         "Refine the strategy of the selected solution to improve it.",  # small mutation
        #         "Generate a new algorithm that is different from the algorithms you have tried before." 
        #     ]

        self.mutation_prompts = ["""An example of such code (a simple random search) for an IOH *clustering* objective is as follows.
Note: for clustering problems. Use the type of algorithm in the name of the program.
```python
import numpy as np
class RandomSearch:
    def __init__(self, budget, dim, seed=None):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        f_opt = np.inf
        x_opt = None

        lb = np.asarray(func.bounds.lb, dtype=float)
        ub = np.asarray(func.bounds.ub, dtype=float)

        while func.state.evaluations < self.budget:
            x = self.rng.uniform(lb, ub, size=self.dim)
            f = func(x)

            if f < f_opt:
                f_opt = f
                x_opt = x.copy()

        return f_opt, x_opt```""", 
"""An example of such code NeighborhoodAdaptiveDE for an IOH *clustering* objective is as follows.
Note: for clustering problems. Use the type of algorithm in the name of the program.
```python
import numpy as np
class NeighborhoodAdaptiveDE:
    def __init__(self, budget=2000, dim=None, seed=None):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.success_f = []
        self.success_cr = []
        self.best_fitness_history = [] 
        self.pop_size = 50
        self.adapt_freq = 50
        self.stagnation_threshold = 1000
        self.learning_rate = 0.1
        self.neighborhood_size = 5
        self.f = 0.5
        self.cr = 0.9

    def __call__(self, func):
        np.random.seed(self.seed)
        f_opt = np.Inf
        x_opt = None
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        self.fitness = np.array([func(x) for x in self.population])
        self.evals = self.pop_size
        self.last_improvement = 0

        best_idx = np.argmin(self.fitness)
        f_opt = self.fitness[best_idx]
        x_opt = self.population[best_idx]
        self.best_fitness_history.append(f_opt)

        while self.evals < self.budget:
            for i in range(self.pop_size):
                neighbors = np.random.choice(np.arange(self.pop_size), self.neighborhood_size, replace=False)
                best_neighbor_idx = neighbors[np.argmin(self.fitness[neighbors])]

                idxs = np.random.choice(np.arange(self.pop_size), 2, replace=False)
                x1, x2 = self.population[idxs]

                v = self.population[best_neighbor_idx] + self.f * (x1 - x2)
                v = np.clip(v, func.bounds.lb, func.bounds.ub)

                u = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == j_rand:
                        u[j] = v[j]

                f_new = func(u)
                self.evals += 1

                if f_new < self.fitness[i]:
                    self.success_f.append(self.f)
                    self.success_cr.append(self.cr)
                    self.fitness[i] = f_new
                    self.population[i] = u

                    if f_new < f_opt:
                        f_opt = f_new
                        x_opt = u
                        self.best_fitness_history.append(f_opt)
                        self.last_improvement = self.evals

            if self.evals % self.adapt_freq == 0:
                if self.success_f:
                    self.f = (1 - self.learning_rate) * self.f + self.learning_rate * np.mean(self.success_f)
                    self.cr = (1 - self.learning_rate) * self.cr + self.learning_rate * np.mean(self.success_cr)
                    self.f = np.clip(self.f, 0.1, 0.9)
                    self.cr = np.clip(self.cr, 0.1, 1.0)
                    self.success_f = []
                    self.success_cr = []

            if self.evals - self.last_improvement > self.stagnation_threshold:
                self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
                self.fitness = np.array([func(x) for x in self.population])
                self.evals += self.pop_size
                best_idx = np.argmin(self.fitness)
                f_opt = self.fitness[best_idx]
                x_opt = self.population[best_idx]
                self.last_improvement = self.evals
                self.best_fitness_history.append(f_opt)

        return f_opt, x_opt```""", 
"""An example of such code a particle swarm optimiser for an IOH *clustering* objective is as follows.
Note: for clustering problems. Use the type of algorithm in the name of the program.

```python
import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=None, seed=None):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(seed)

        self.s = 50
        self.inertia = 0.5
        self.psi_p = 1.5
        self.psi_g = 1.5

    def __call__(self, func):
        positions = self.rng.uniform(func.bounds.lb, func.bounds.ub, size=(self.s, self.dim))
        domain = np.abs(np.array(func.bounds.ub) - np.array(func.bounds.lb))
        velocity = self.rng.uniform(-domain, domain, size=(self.s, self.dim))

        f = np.array([func(x) for x in positions])
        best_known_positions = positions.copy()
        best_known_values = f.copy()
        idx = np.argmin(f)
        f_opt, x_opt = f[idx], positions[idx, :]

        while func.state.evaluations + self.s <= self.budget:
            rp = self.rng.uniform(size=(self.s, 1))
            rg = self.rng.uniform(size=(self.s, 1))
            velocity = (
                (self.inertia * velocity)
                + (self.psi_p * rp * (best_known_positions - positions))
                + (self.psi_g * rg * (x_opt - positions))
            )

            positions += velocity
            f = np.array([func(x) for x in positions])
            improvements = f < best_known_values
            best_known_positions[improvements] = positions[improvements]
            best_known_values[improvements] = f[improvements]

            idy = np.argmin(f)
            if f[idy] < f_opt:
                f_opt = f[idy]
                x_opt = positions[idy]

            if abs(func.optimum.y - func.state.current_best.y) < 1e-8:
                break

        return f_opt, x_opt```""", 

"""An example of such code a UMDA solution for an IOH *clustering* objective is as follows.
Note: for clustering problems.

```python
import numpy as np
import math

class UMDA:
    def __init__(self, budget, dim, seed=None):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.population_size = 50

    def __call__(self, func):
        n = self.dim
        s = self.population_size

        pop = self.rng.integers(0, 2, size=(s, n), dtype=np.uint8)
        fitness = np.empty(s, dtype=float)
        for i in range(s):
            fitness[i] = func(pop[i])

        best_idx = int(np.argmax(fitness))
        best_x = pop[best_idx].copy()
        best_f = float(fitness[best_idx])

        mu = s // 2
        idx_sorted = np.argsort(-fitness)
        parents = pop[idx_sorted[:mu]]

        while func.state.evaluations + s <= self.budget:
            counts = parents.sum(axis=0)
            p = np.clip(2.0 * counts / s, 1.0 / n, 1.0 - 1.0 / n)

            pop = (self.rng.random((s, n)) < p).astype(np.uint8)
            fitness = np.empty(s, dtype=float)
            for i in range(s):
                if func.state.evaluations >= self.budget:
                    return best_x, best_f
                fitness[i] = func(pop[i])
                if fitness[i] > best_f:
                    best_f, best_x = fitness[i], pop[i].copy()

            if best_f >= func.optimum.y:
                break

            idx_sorted = np.argsort(-fitness)
            parents = pop[idx_sorted[:mu]]

        return best_f, best_x```"""]

        self.example_prompt = """An example of such code (a simple random search) for an IOH *clustering* objective is as follows.
        Note: for clustering problems.

        ```python
            import numpy as np

            class RandomSearch:
                def __init__(self, budget, dim, seed=None):
                    self.budget = budget
                    self.dim = dim
                    self.rng = np.random.default_rng(seed)

                def __call__(self, func):
                    f_opt = np.inf
                    x_opt = None

                    lb = np.asarray(func.bounds.lb, dtype=float)
                    ub = np.asarray(func.bounds.ub, dtype=float)

                    while func.state.evaluations < self.budget:
                        x = self.rng.uniform(lb, ub, size=self.dim)
                        f = func(x)

                        if f < f_opt:
                            f_opt = f
                            x_opt = x.copy()

        return f_opt, x_opt"""

       

        self.format_prompt = """
            Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
            # Description: <short-description>
            # Code: 
            ```python
            <code>
            ```
            """
        
        
    def __call__(self, solution: Solution, logger=None):
        self._debug_log(f"__call__ entered for solution={solution.id} gen={solution.generation}")

        out = super().__call__(solution, logger)

        # debug log
        self._debug_log(
            f"__call__ exiting for solution={solution.id} fitness={out.fitness}"
            f"feedback={repr(out.feedback)} error={repr(out.error)}"
        )

        # NEW: also log subprocess stdout/stderr captured by Problem.__call__
        stdout = getattr(self, "_last_stdout", "")
        stderr = getattr(self, "_last_stderr", "")
        if stdout:
            self._debug_log(f"subprocess stdout (truncated)={repr(stdout[-500:])}")
        if stderr:
            self._debug_log(f"subprocess stderr (truncated)={repr(stderr[-500:])}")

        return out

    def _debug_log(self, msg: str):
        # Prefer Experiment's directory if available; otherwise use CWD
        # I want to add current points or dataset to the log
        # Dataset would be sufficient
        base = getattr(getattr(self, "exp_logger", None), "dirname", None)
        if not base:
            base = os.environ.get("IOHBLADE_LOG_DIR", None)  # optional override
        if not base:
            base = os.getcwd()

        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, "debug_eval.log")

        with open(path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")

        return path


    # -------------------------
    # Prompt handling (no wording included here)
    # -------------------------
    def set_prompts(self, task_prompt: str, example_prompt: str, format_prompt: str):
        self.task_prompt = task_prompt
        self.example_prompt = example_prompt
        self.format_prompt = format_prompt

    def get_prompt(self):
        if not (self.task_prompt and self.example_prompt and self.format_prompt):
            raise ValueError(
                "Prompts are not set. Call problem.set_prompts(task_prompt, example_prompt, format_prompt) "
                "or set task_prompt/example_prompt/format_prompt attributes before running LLaMEA."
            )

        return self.task_prompt + self.example_prompt + self.format_prompt

    # -------------------------
    # IOH clustering specifics
    # -------------------------
    @staticmethod
    def _make_problem(fid: int, iid: int, k: int):
        import iohclustering
        p, _ = iohclustering.get_problem(fid=fid, instance=iid, k=k)
        
        return p

    @staticmethod
    def _infer_dim(p):
        md = getattr(p, "meta_data", None)
        if md is not None and hasattr(md, "n_variables"):
            return int(md.n_variables)
        if hasattr(p, "dimension"):
            return int(p.dimension)
        b = getattr(p, "bounds", None)
        if b is not None and hasattr(b, "lb"):
            return int(np.asarray(b.lb).size)
        return None

    def _budget(self, p):
        if not self.budget_scale_with_dim:
            return self.base_budget
        dim = self._infer_dim(p)
        return self.base_budget if dim is None else int(self.base_budget * dim)

    # -------------------------
    # Evaluation (AOCC-style, like BBOB_SBOX)
    # -------------------------
    def evaluate(self, solution: Solution, test: bool = False):
        # Lazy imports for dependencies only needed during evaluation
        import ioh
        from iohblade.solution import Solution as SolutionClass
        from iohblade.utils import OverBudgetException, aoc_logger, correct_aoc
        
        self._eval_calls = getattr(self, "_eval_calls", 0) + 1
        solution.add_metadata("eval_call_index", self._eval_calls)
        
        log_path = self._debug_log(f"evaluate() called for solution={solution.id} gen={solution.generation} test={test} dataset={self.data_set_name}")


        code = solution.code
        algorithm_name = solution.name

        # execute generated code in a controlled namespace
        safe_globals = {"np": np, "math": math, "ioh": ioh, "deque": deque, "Counter": Counter, "defaultdict": defaultdict, "random": random, "itertools": itertools, "heapq": heapq,}
        
        local_env = {}
        try:
            exec(code, safe_globals, local_env)
        except Exception as e:
            tb = traceback.format_exc()
            with open(log_path, "a") as f:
                f.write(f"exec() failed: {type(e).__name__}: {e}\n{tb}\n")
            solution.set_scores(-np.inf, f"exec() failed: {type(e).__name__}: {e}")
            return solution

        if not algorithm_name or algorithm_name not in local_env:
            with open(log_path, "a") as f:
                f.write(f"Algorithm class '{algorithm_name}' not found in local_env. Keys: {list(local_env.keys())}\n")
            solution.set_scores(-np.inf, f"Algorithm class '{algorithm_name}' not found after exec().")
            return solution

        Algo = local_env[algorithm_name]

        # NOTE: SMOKE TEST

        # is a smoke test needed?
        # Check this it's important to know and realise 
        # inst0 = (self.test_instances[0] if test else self.training_instances[0])
        # fid0, iid0, k0 = inst0
        # p0 = self._make_problem(fid0, iid0, k0)
        # b0 = max(20, min(50, self._budget(p0)))
        # self.data_set_name= repr(p0.meta_data)  
        # l2_tmp = aoc_logger(b0, upper=1e2, triggers=[ioh.logger.trigger.ALWAYS])

        # try:
        #     p0.attach_logger(l2_tmp)
        #     # Try instantiating Algo with common signatures; fallback if needed
        #     try:
        #         alg0 = Algo(budget=b0)
        #     except TypeError:
        #         try:
        #             dim0 = self._infer_dim(p0) or 0
        #             alg0 = Algo(budget=b0, dim=dim0)
        #         except Exception as e2:
        #             raise
        #     alg0(p0)
        # except OverBudgetException:
        #     pass
        # except Exception as e:
        #     tb = traceback.format_exc()
        #     with open(log_path, "a") as f:
        #         f.write(f"Smoke test failed: {type(e).__name__}: {e}\n{tb}\n")
        #     try:
        #         p0.reset()
        #     except Exception:
        #         pass
        #     solution.set_scores(-np.inf, f"Smoke test failed: {type(e).__name__}: {e}")
        #     return solution
        # finally:
        #     try:
        #         p0.reset()
        #     except Exception:
        #         pass


        # ---- main evaluation ----
        # I am unhappy with how you are currently analsying these results 
        instances = self.test_instances if test else self.training_instances

        aucs = []
        performance_data = []
        best_curve = np.array([], dtype=float)   
        pw = None                                

        for (fid, iid, k) in instances:
            
            p = self._make_problem(fid, iid, k)
            budget = self._budget(p)
            l2 = aoc_logger(budget, upper=1e2, triggers=[ioh.logger.trigger.ALWAYS])
            baseline_mse = self.kmeans_min.get((int(fid), int(k)))

            try:
                if test or self.full_ioh_log:
                    l1 = ioh.logger.Analyzer(
                        root=self.ioh_dir if self.ioh_dir else "ioh_logs",
                        folder_name=solution.id,
                        algorithm_name=solution.id,
                        store_positions=True,
                        triggers=[ioh.logger.trigger.ALWAYS],
                    )
                    p.attach_logger(ioh.logger.Combine([l1, l2]))
                else:
                    p.attach_logger(l2)
                p_dim = self._infer_dim(p) or k * 2
                try:
                    alg = Algo(budget=budget, dim=p_dim)
                except TypeError:
                    alg = Algo(budget=budget)            
                
                pw = BestSoFarWrapper(p)
                alg(pw)
                best_curve = np.asarray(pw.best_curve, dtype=float)

            except OverBudgetException:
                # count it anyway; AOCC will handle based on recorded evaluations
                pass
            except Exception as e:
                # Maybe I should not penalise errors as heavily
                # Errors tend to happen when diversifying the number of problems
                tb = traceback.format_exc()
                with open(log_path, "a") as f:
                    f.write(f"Evaluation failed for fid={fid}, iid={iid}, k={k}: {type(e).__name__}: {e}\n{tb}\n")
                performance_data.append({"fid": fid, "iid": iid, "k": k, "auc": float(-1e9), "error": f"{type(e).__name__}: {e}"})
                aucs.append(-1e9)
                try:
                    p.reset()
                except Exception:
                    pass
                continue
            finally:
                # best-effort reset
                try:
                    p.reset()
                except Exception:
                    pass
            

            if best_curve.size == 0:
                solution.set_scores(-np.inf, "Algorithm did not call func(x); no evaluations recorded.")
                return solution
            else:
                if best_curve.size < budget:
                    best_curve = np.pad(best_curve, (0, budget - best_curve.size), mode="edge")
                else: 
                    best_curve = best_curve[:budget]

                improvement_curve = baseline_mse - best_curve  # baseline minus best-so-far MSE
                auc_imp = float(np.mean(improvement_curve))    # “AUC” as average over budget

            performance_data.append({
                "fid": fid, "iid": iid, "k": k,
                "problem_name": str(p),
                "baseline_mse": baseline_mse,
                "auc_imp": auc_imp,
            })
            aucs.append(auc_imp if auc_imp is not None else 0.0)

            try:
                l2.reset(p)
            except Exception:
                pass

        auc_mean = float(np.mean(aucs)) if aucs else 0.0
        # Log final aggregated values for debugging why fitness becomes -inf
        with open(log_path, "a") as f:
            f.write(f"Final aucs: {aucs} | auc_mean: {auc_mean}\n")
        solution.add_metadata("performance_data", performance_data)
        solution.add_metadata("aucs", aucs)
        solution.set_scores(auc_mean, f"{algorithm_name} mean improvement-AUC vs KMeans(min) = {auc_mean:.6f}")
        return solution

    def test(self, solution: Solution):
        return self.evaluate(solution, test=True)

    def to_dict(self):
        return {
            "name": self.name,
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
            "base_budget": self.base_budget,
            "budget_scale_with_dim": self.budget_scale_with_dim,
            "full_ioh_log": self.full_ioh_log,
            "ioh_dir": self.ioh_dir,
        }

        
