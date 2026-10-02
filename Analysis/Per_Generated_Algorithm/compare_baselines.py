import glob
import json
import os
import importlib.util
import traceback
import ioh
import iohclustering
import numpy as np


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


# experiment configs

EXPERIMENTS = {
    "E1_no_domain_knowledge": {
        "label": "E1: No domain knowledge",
        "runs": sorted(glob.glob("results/without_domain_knowledge/*/log.jsonl")),
    },
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
    "E5_domain_mutation": {
        "label": "E5: Domain + mutation prompts",
        "runs": sorted(glob.glob("results/5_runs_mutation_prompts_WITH_DOMAIN_KNOWLEDGE/*/log.jsonl")),
    },
    "E6_domain_warm_start": {
        "label": "E6: Domain + warm start",
        "runs": sorted(glob.glob("results/5_runs_warm_start_WITH_DOMAIN_KNOWLEDGE/*/log.jsonl")),
    },
}

# matching cma-es params

REPS   = 25
BUDGET = 5000
FIDS   = range(1, 11)
KS     = [2, 3, 5, 10]
ROOT   = "Data"
BEST_ALG_DIR = "best_algorithms"

os.makedirs(BEST_ALG_DIR, exist_ok=True)


# find best algorithm

def extract_best_algorithm(config_key: str, run_files: list[str]) -> dict | None:
    """
    Scans all log.jsonl files for a configuration and returns the entry
    with the highest fitness value (higher = better, i.e. least-negative MSE).
    """
    best_entry = None
    best_fitness = -np.inf

    for log_path in run_files:
        if not os.path.isfile(log_path):
            print(f"  [warn] missing file: {log_path}")
            continue
        with open(log_path, "r") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  [warn] JSON parse error in {log_path}:{lineno}: {e}")
                    continue

                fitness = entry.get("fitness")
                code    = entry.get("code")
                # skip entries that had errors or missing code
                if fitness is None or code is None:
                    continue
                try:
                    fitness = float(fitness)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(fitness):
                    continue

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_entry   = entry
                    best_entry["_source_file"] = log_path

    if best_entry is None:
        print(f"  [warn] No valid entries found for {config_key}.")
    else:
        print(
            f"  Best fitness: {best_fitness:.6f}  "
            f"| name: {best_entry.get('name', 'unknown')}  "
            f"| source: {best_entry['_source_file']}"
        )
    return best_entry


# save best generated algorithm code

def save_algorithm(config_key: str, entry: dict) -> str:
    """Writes the algorithm code to best_algorithms/<config_key>.py."""
    out_path = os.path.join(BEST_ALG_DIR, f"{config_key}.py")
    with open(out_path, "w") as f:
        f.write(entry["code"])
    print(f"  Saved to {out_path}")
    return out_path


# load the algorithm class

def load_algorithm_class(py_path: str, class_name: str):
    """
    Dynamically imports a .py file and returns the class named `class_name`.
    Falls back to the first class defined in the module if the name doesn't match.
    """
    module_name = os.path.splitext(os.path.basename(py_path))[0]
    spec   = importlib.util.spec_from_file_location(module_name, py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # try exact name first, then fall back to any class in the module
    cls = getattr(module, class_name, None)
    if cls is None:
        # find the first class that looks like an algorithm (has __call__ or __init__)
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if isinstance(obj, type) and not attr_name.startswith("_"):
                cls = obj
                print(f"    [info] class '{class_name}' not found; using '{attr_name}' instead.")
                break

    if cls is None:
        raise RuntimeError(f"Could not find any usable class in {py_path}")
    return cls


# run the current algorithm on the benchmark suite

def run_algorithm_on_benchmark(config_key, alg_label, py_path, class_name):
    print(f"\n  Running benchmark for {config_key} ({alg_label}) ...")

    try:
        AlgClass = load_algorithm_class(py_path, class_name)
    except Exception as e:
        print(f"  [error] Could not load algorithm: {e}")
        traceback.print_exc()
        return

    for fid in FIDS:
        for k in KS:
            folder = f"{config_key}/fid{fid}_k{k}"
            print(f"    fid={fid}, k={k} ...", end=" ", flush=True)

            logger = ioh.logger.Analyzer(
                root=ROOT,
                folder_name=folder,
                algorithm_name=config_key,
                store_positions=True,
                triggers=[ioh.logger.trigger.ALWAYS],
            )

            problem, _ = iohclustering.get_problem(fid=fid, instance=1, k=k)
            problem.reset()
            problem.attach_logger(logger)

            for rep in range(REPS):
                problem.reset()
                dim = problem.meta_data.n_variables

                # wrap in budget enforcer (because of problems before)
                enforcer = BudgetEnforcer(problem, BUDGET)

                try:
                    alg = AlgClass(budget=BUDGET, dim=dim, seed=rep)
                    alg(enforcer)          # algorithm runs against the enforcer
                except Exception as e:
                    print(f"\n      [error] rep {rep}: {e}")
                    traceback.print_exc()
                finally:
                    enforcer.exhaust()     # fill any remaining budget unconditionally

            problem.detach_logger()
            print("done")


def main():
    print("=" * 70)
    print("LLaMEA best-algorithm extraction and benchmark runner")
    print("=" * 70)

    for config_key, cfg in EXPERIMENTS.items():
        label     = cfg["label"]
        run_files = cfg["runs"]

        print(f"\n{'─' * 60}")
        print(f"Config: {label}  ({len(run_files)} run files found)")

        if not run_files:
            print("  [skip] No log files found — check your glob paths.")
            continue

        # find best entry
        best = extract_best_algorithm(config_key, run_files)
        if best is None:
            continue

        # save
        py_path    = save_algorithm(config_key, best)
        class_name = best.get("name", "Algorithm")

        # then run
        run_algorithm_on_benchmark(config_key, label, py_path, class_name)

    print("\n" + "=" * 70)
    print("All done. IOH logs written to:", os.path.abspath(ROOT))
    print("Extracted algorithms written to:", os.path.abspath(BEST_ALG_DIR))
    print("=" * 70)


if __name__ == "__main__":
    main()