import glob
import json
import os
import importlib.util
import traceback

import ioh
import iohclustering
import numpy as np

# configs - best four from the baseline compare

EXPERIMENTS = {
    "E2_domain_knowledge": {
        "label": "E2: Domain knowledge",
        "runs": sorted(glob.glob("results/domain_knowledge/*/log.jsonl")),
    },
    "E3_no_domain_mutation": {
        "label": "E3: No Domain + mutation prompts",
        "runs": sorted(glob.glob("results/mutation_5_runs_elitism_on/*/log.jsonl")),
    },
    "E4_no_domain_warm_start": {
        "label": "E4: No Domain + warm start",
        "runs": sorted(glob.glob("results/warm_start_5_runs/*/log.jsonl")),
    },
    "E6_domain_warm_start": {
        "label": "E6: Domain + warm start",
        "runs": sorted(glob.glob("results/5_runs_warm_start_WITH_DOMAIN_KNOWLEDGE/*/log.jsonl")),
    },
}

# matching cma-es params
REPS        = 25
BUDGET      = 5000
FIDS        = range(1, 11)
KS          = [2, 3, 5, 10]
ROOT        = "Data"
BEST_ALG_DIR = "3_ranged_algorithms"
SENTINEL_THRESHOLD = -1e9   # fitness values at or below this are excluded

os.makedirs(BEST_ALG_DIR, exist_ok=True)

RANKS = ["worst", "middle", "best"]


# prevent generated algs from terminating early mistmatch of log files

class BudgetEnforcer:
    def __init__(self, problem, budget: int):
        self.problem      = problem
        self.budget       = budget
        self._best_x      = None
        self._eval_count  = 0
        self.meta_data    = problem.meta_data   

    def __call__(self, x):
        y = self.problem(x)
        self._eval_count += 1
        # track best x seen so far 
        if self._best_x is None or y <= self.problem.state.current_best.y:
            self._best_x = x.copy()
        return y

    def exhaust(self):
        """Fill remaining budget by re-evaluating the best solution found."""
        while self._eval_count < self.budget:
            self.problem(self._best_x)
            self._eval_count += 1

    # forward any attribute lookups to the underlying problem
    def __getattr__(self, name):
        return getattr(self.problem, name)


# exract best worst and median 

def extract_ranked_algorithms(config_key: str, run_files: list) -> dict:
    entries = []   # list of (fitness, entry)
    seen_ids = set()

    for log_path in run_files:
        if not os.path.isfile(log_path):
            print(f"  [warn] missing: {log_path}")
            continue

        with open(log_path, "r") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                fitness = entry.get("fitness")
                code    = entry.get("code")
                uid     = entry.get("id")

                if fitness is None or code is None:
                    continue
                try:
                    fitness = float(fitness)
                except (TypeError, ValueError):
                    continue

                # exclude non-finite and sentinel values
                if not np.isfinite(fitness):
                    continue
                if fitness <= SENTINEL_THRESHOLD:
                    continue

                dedup_key = uid if uid else (fitness, entry.get("name", ""))
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)

                entries.append((fitness, entry))

    if not entries:
        print(f"  [warn] No valid entries found for {config_key}.")
        return {}

    # sort ascending by fitness
    entries.sort(key=lambda x: x[0])
    n = len(entries)

    print(f"  {n} valid unique entries  |  "
          f"worst={entries[0][0]:.6f}  "
          f"best={entries[-1][0]:.6f}")

    mid_idx = n // 2

    selected = {
        "worst" : entries[0],
        "middle": entries[mid_idx],
        "best"  : entries[-1],
    }

    # show selections and vals
    for rank, (fit, ent) in selected.items():
        print(f"    {rank:6s}: fitness={fit:.6f}  name={ent.get('name', 'unknown')}")

    return {rank: entry for rank, (_, entry) in selected.items()}


# save algorithm

def save_algorithm(config_key: str, rank: str, entry: dict) -> str:
    fname    = f"{config_key}_{rank}.py"
    out_path = os.path.join(BEST_ALG_DIR, fname)
    with open(out_path, "w") as f:
        f.write(entry["code"])
    print(f"    Saved: {out_path}")
    return out_path


# load algorithm

def load_algorithm_class(py_path: str, class_name: str):
    module_name = os.path.splitext(os.path.basename(py_path))[0]
    spec   = importlib.util.spec_from_file_location(module_name, py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cls = getattr(module, class_name, None)
    if cls is None:
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if isinstance(obj, type) and not attr_name.startswith("_"):
                print(f"    [info] '{class_name}' not found; using '{attr_name}'.")
                cls = obj
                break

    if cls is None:
        raise RuntimeError(f"No usable class found in {py_path}")
    return cls


# run the current algorithm on the benchmark suite

def run_algorithm_on_benchmark(config_key: str, rank: str, py_path: str, class_name: str):
    """
    Logs to Data/<config_key>_<rank>/fid<f>_k<k>/
    so all three ranks per config are kept separate.
    """
    run_key = f"{config_key}_{rank}"
    print(f"\n    Running {run_key} ...")

    try:
        AlgClass = load_algorithm_class(py_path, class_name)
    except Exception as e:
        print(f"    [error] Could not load: {e}")
        traceback.print_exc()
        return

    for fid in FIDS:
        for k in KS:
            folder = f"{run_key}/fid{fid}_k{k}"
            print(f"      fid={fid}, k={k} ...", end=" ", flush=True)

            logger = ioh.logger.Analyzer(
                root=ROOT,
                folder_name=folder,
                algorithm_name=run_key,
                store_positions=True,
                triggers=[ioh.logger.trigger.ALWAYS],
            )

            problem, _ = iohclustering.get_problem(fid=fid, instance=1, k=k)
            problem.reset()
            problem.attach_logger(logger)

            for rep in range(REPS):
                problem.reset()
                dim      = problem.meta_data.n_variables
                enforcer = BudgetEnforcer(problem, BUDGET)

                try:
                    alg = AlgClass(budget=BUDGET, dim=dim, seed=rep)
                    alg(enforcer)
                except Exception as e:
                    print(f"\n        [error] rep {rep}: {e}")
                    traceback.print_exc()
                finally:
                    enforcer.exhaust()   # always pad to exactly BUDGET evals

            problem.detach_logger()
            print("done")

def main():
    print("=" * 70)
    print("LLaMEA ranked-algorithm (worst/middle/best) benchmark runner")
    print("=" * 70)

    for config_key, cfg in EXPERIMENTS.items():
        label     = cfg["label"]
        run_files = cfg["runs"]

        print(f"\n{'─' * 60}")
        print(f"Config: {label}  ({len(run_files)} run files found)")

        if not run_files:
            print("  [skip] No log files found.")
            continue

        ranked = extract_ranked_algorithms(config_key, run_files)
        if not ranked:
            continue

        for rank in RANKS:
            if rank not in ranked:
                print(f"  [skip] rank '{rank}' not available.")
                continue

            entry      = ranked[rank]
            py_path    = save_algorithm(config_key, rank, entry)
            class_name = entry.get("name", "Algorithm")
            run_algorithm_on_benchmark(config_key, rank, py_path, class_name)

    print("\n" + "=" * 70)
    print("Done. IOH logs:", os.path.abspath(ROOT))
    print("Algorithm files:", os.path.abspath(BEST_ALG_DIR))
    print("=" * 70)


if __name__ == "__main__":
    main()