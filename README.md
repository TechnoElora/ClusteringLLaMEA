# Automated Algorithm Design for Clustering: How Prompting Strategies Shape LLM-Generated Optimiser

**Prompting strategies and selection pressure in LLM-driven automated algorithm design for continuous clustering problems**

Author - Anon, Affliation - Anon, 2026

[Paper](LINK_TO_PAPER) | [Data archive](https://zenodo.org/records/24244235?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjA4N2NlNTIyLTVlYmMtNGE0Yy04ZWU5LTljYzI1OGYxY2QzNiIsImRhdGEiOnt9LCJyYW5kb20iOiJkYmRlNGViMzU3NTJhN2MzZmRjZWM3OTZlMzgwN2U1ZiJ9.lLMlLAnRxBaAwQciW5RnL_sF83heD9xgP-QcgOZ2hWnofB9HRkqC4fA44VBb-m_oCO6NIV-tPluSdBGnkJGcag) | [Citation](#citation)

> [!NOTE]
> This repository contains the code, analysis scripts and result tables for the paper above. Raw run logs and generated algorithms are archived separately; see [Data and results](#data-and-results).

## Table of Contents

- [Overview](#overview)
- [Experimental design](#experimental-design)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the experiments](#reproducing-the-experiments)
  - [Prompting-level experiments](#prompting-level-experiments)
  - [Candidate-algorithm-level analyses](#candidate-algorithm-level-analyses)
- [Data and results](#data-and-results)
- [Reproducibility notes](#reproducibility-notes)
- [Citation](#citation)

## Overview

We study how prompting choices and elitism affect the optimisation algorithms that [LLaMEA](https://github.com/XAI-liacs/LLaMEA) generates for the clustering benchmark of Vermetten, Dinu and Gallagher, a continuous black-box optimisation suite over cluster-centre coordinates. Experiment were run in [BLADE](https://github.com/XAI-liacs/BLADE) with a local [Ollama](https://ollama.com/) model, but a different model can be chosen with its associated API key.

The repository supports two levels of analysis:

| Level | Explanation | Entry point |
|---|---|---|
| **Prompting level** | The code runs the blade evaluation with 5 runs per chosen configuration and 200 generated candidates. | [`ClusteringProblems.ipynb`](ClusteringProblems.ipynb) |
| **Candidate-algorithm level** | Evaluates the best generated (for a range_of_vals_compare it's best, worst and median candidate) candidates compared over 5000 evaluations  | Scripts in [`Analysis/Per_Generated_Algorithm`](Analysis/Per_Generated_Algorithm) |

## Experimental design

Six prompting configurations are compared, each with and without elitism (200 evaluations per run).

| Configuration | Domain knowledge | Strategy |
|---|---|---|
| E1 | No | Standard |
| E2 | Yes | Standard |
| E3 | No | Exemplar injection |
| E4 | No | Warm start |
| E5 | Yes | Exemplar injection |
| E6 | Yes | Warm start |

Fitness is reported relative to the KMeans++ reference values ([`kmeans_baseline.csv`](kmeans_baseline.csv)). Selected candidates are additionally compared with CMA-ES ([`overview_cmaes.csv`](overview_cmaes.csv)) on held-out instances.

## Repository structure

```
.
├── ClusteringProblems.ipynb            # prompting-level experiments (E1-E6)
├── ClusteringProblems_executed.ipynb   # same notebook with the saved outputs of the reported run
├── IOHClusteringAdapter.py             # determines fitness of generated algorithms on (fid, iid, k)
├── llamea_warm_start_changes.py        # for warm start when using LLaMEA substitute llamea.py with modifications in this file
├── Analysis/
│   ├── Per_Generated_Algorithm/        # candidate-level scripts
│   └── Per_LLaMEA_run/                 # run-level notebooks and tables
├── kmeans_baseline.csv, KMeansCompare.csv        # KMeans++ reference values
├── overview_cmaes.csv                            # CMA-ES results
├── generalizability_*.csv                        # results on held-out instances
├── pyproject.toml, uv.lock, .python-version      # environment
└── LICENSE
```

## Installation

**Requirements**

- Python 3.13.12 for all files except `Analysis/Per_LLaMEA_run/CodeBleu.ipynb`, which needs Python 3.11.15 due to dependency conflicts
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com/) with the model `qwen3-coder:30b` 

**Setup**

```bash
git clone https://github.com/<USER>/<REPO>.git
cd <REPO>
uv sync
```

> [!IMPORTANT]
> `CodeBleu.ipynb` needs its own Python 3.11 environment:
> ```bash
> uv venv .venv-codebleu --python 3.11.15
> source .venv-codebleu/bin/activate        # Windows: .venv-codebleu\Scripts\activate
> uv pip install -r requirements-codebleu.txt
> ```

## Replicating the experiments

### Prompting-level experiments

1. Make sure Ollama is running with the model - `qwen3-coder:30b`.
2. Open [`ClusteringProblems.ipynb`](ClusteringProblems.ipynb).
3. Run all cells. Adjust exemplar prompts based on configuration and adjust prompts in [`IOHClusteringAdapter.py](IOHClusteringAdapter.py) is run for 200 evaluations. When using elistism ensure `n_parents`= 4 and `n_offspring`= 8, for without elitism use `n_parents`= 4 and `n_offspring`= 16. 

The warm-start configuration relies on the modified LLaMEA functions in [`llamea_warm_start_changes.py`](llamea_warm_start_changes.py), 

### Candidate-algorithm-level analyses

These analyses use the algorithms generated in the runs above (or the archived runs, see [Data and results](#data-and-results)).

| Script / notebook | Purpose |
|---|---|
| [`Analysis/Per_Generated_Algorithm/compare_baselines.py`](Analysis/Per_Generated_Algorithm/compare_baselines.py) | Run best candidates (per configuration) on the clustering benchmark |
| [`Analysis/Per_Generated_Algorithm/range_of_vals_compare.py`](Analysis/Per_Generated_Algorithm/range_of_vals_compare.py) | Run best median and worst candidates (per configuration) on the clustering benchmark |
| [`Analysis/Per_LLaMEA_run/Evaluation.ipynb`](Analysis/Per_LLaMEA_run/Evaluation.ipynb) | Contains the majority of graphs used in the paper |
| [`Analysis/Per_LLaMEA_run/Generalisability_alg_cat_prop.ipynb`](Analysis/Per_LLaMEA_run/Generalisability_alg_cat_prop.ipynb) | Contains the generalisability ability of the configurations + algorithm category proportion with violin plots |
| [`Analysis/Per_LLaMEA_run/CodeBleu.ipynb`](Analysis/Per_LLaMEA_run/CodeBleu.ipynb) | CodeBLEU similarity versus fitness difference (Python 3.11 environment) |

```bash
uv run python Analysis/Per_Generated_Algorithm/compare_baselines.py
uv run python Analysis/Per_Generated_Algorithm/range_of_vals_compare.py
```

## Data and results

| Item | Download |
|---|---|
| KMeans++ reference values | [`kmeans_baseline.csv`](kmeans_baseline.csv), [`KMeansCompare.csv`](KMeansCompare.csv) |
| CMA-ES results | [`overview_cmaes.csv`](overview_cmaes.csv) |
| Generalisation results | [`generalizability_mse_by_fid_k.csv`](generalizability_mse_by_fid_k.csv), [`generalizability_results_domain.csv`](generalizability_results_domain.csv), [`generalizability_results_overwrite.csv`](generalizability_results_overwrite.csv) |
| Effect sizes (Cliff's delta) | [`Analysis/Per_LLaMEA_run/cliffs_delta_vs_reference.csv`](Analysis/Per_LLaMEA_run/cliffs_delta_vs_reference.csv) |
| Raw LLaMEA run logs and generated algorithms | [Zenodo archive](https://zenodo.org/records/24244235?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjA4N2NlNTIyLTVlYmMtNGE0Yy04ZWU5LTljYzI1OGYxY2QzNiIsImRhdGEiOnt9LCJyYW5kb20iOiJkYmRlNGViMzU3NTJhN2MzZmRjZWM3OTZlMzgwN2U1ZiJ9.lLMlLAnRxBaAwQciW5RnL_sF83heD9xgP-QcgOZ2hWnofB9HRkqC4fA44VBb-m_oCO6NIV-tPluSdBGnkJGcag) |

When importing the run log generated algorithms, extract the zip file and add the internal folder to your local version of this repo (do not change the extracted folder names otherwise the analysis files will not find the logs.).

## Reproducibility notes

All experiments used the Ollama model `qwen3-coder:30b`. LLM sampling is stochastic, so re-running the pipeline reproduces the procedure but not the exact generated algorithms or fitness values. The numbers in the paper come from the archived runs; the analysis scripts regenerate the paper's tables and figures from them. Running the pipeline with a different LLM or on different clustering instances would be a replication of the study rather than a reproduction.

## Citation
This work builds on the following:

```bibtex
@article{vanstein2025llamea,
  author  = {van Stein, Niki and B{\"a}ck, Thomas},
  title   = {{LLaMEA}: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics},
  journal = {IEEE Transactions on Evolutionary Computation},
  volume  = {29},
  number  = {2},
  pages   = {331--345},
  year    = {2025},
  doi     = {10.1109/TEVC.2024.3497793}
}

@inproceedings{vanstein2025blade,
  author    = {van Stein, Niki and Kononova, Anna V. and Yin, Haoran and B{\"a}ck, Thomas},
  title     = {{BLADE}: Benchmark suite for {LLM}-driven Automated Design and Evolution of iterative optimisation heuristics},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference Companion},
  series    = {GECCO '25 Companion},
  year      = {2025},
  pages     = {2336--2344},
  doi       = {10.1145/3712255.3734347}
}

@misc{vermetten2025clustering,
  author = {Vermetten, Diederick and Dinu, Catalin-Viorel and Gallagher, Marcus},
  title  = {A Standardized Benchmark Set of Clustering Problem Instances for Comparing Black-Box Optimizers},
  year   = {2025},
  note   = {arXiv:2505.09233},
  doi    = {10.1145/3729878.3746699}
}
```


