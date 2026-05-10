#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Import necessary libraries
# #%pip install git+https://github.com/IOHprofiler/iohclustering.git
# #%pip install modcma

# from iohclustering.utils import get_dataset
# from iohclustering.cluster_baseline_problems import CLUSTER_BASELINE_DATASETS

# from iohclustering import create_cluster_problem, general_cluster_metric, get_problem
# import ioh
# import numpy as np
# import os


# In[ ]:


# from iohclustering import load_problems
# import iohclustering


# p, _= iohclustering.get_problem(fid=1, instance=1, k=2)
# print("All attributes of p:", dir(p))


# # problems = load_problems()

# # for problem in problems.keys():
# #     print(problem)

# # Not needed but to check how the downloads work


# ## Using the CMA-ES Algorithm on clustering problems to see what your current configuration of the algorithm gets as a result

# In[ ]:


# import numpy as np
# from types import SimpleNamespace
# import json
# import ioh
# import iohclustering
# from modcma.modularcmaes import ModularCMAES
# from modcma.parameters import Parameters

# # 1) tiny metadata object (so IOH can store these as columns)
# cfg = SimpleNamespace(
#     lambda_=20,
#     mu=10,
#     elitist=True,
#     covariance= False,          
#     bound_correction= None,           
# )

# # 2) IOH logger: generates Data/<folder>/... logs
# cfg_summary = {
#     "lambda_": cfg.lambda_,
#     "mu": cfg.mu,
#     "elitist": cfg.elitist,
#     "covariance": str(cfg.covariance),
#     "bound_correction": str(cfg.bound_correction),
# }

# # logger = ioh.logger.Analyzer(
# #     root="Data",
# #     folder_name="CMA_REPLICATION_STUDY",
# #     algorithm_name="ModCMA_CMAES",
# #     algorithm_info=json.dumps(cfg_summary),  # record config, no folder splitting
# # )


# def run_modcma_on_problem(problem, seed: int, budget: int):
#     dim = problem.meta_data.n_variables
#     rng = np.random.default_rng(seed)

#     # clustering problems in your code are normalized to [0,1], so keep bounds consistent
#     lb = np.zeros(dim)
#     ub = np.ones(dim)

#     x0 = rng.uniform(0.0, 1.0, size=dim)
#     sigma0 = 0.3

#     # Map your "nan" -> None for ModCMA
#     bc = None if cfg.bound_correction == "nan" else cfg.bound_correction

#     params = Parameters(
#         d=dim,
#         x0=x0,
#         sigma0=sigma0,
#         lambda_=cfg.lambda_,
#         mu=cfg.mu,
#         elitist=cfg.elitist,
#         bound_correction=bc,
#         lb=lb,
#         ub=ub,
#     )

#     opt = ModularCMAES(fitness_func=problem, parameters=params)

#     while opt.parameters.used_budget < budget:
#         if not opt.step():
#             break


# # 3) your experiment loop (this is what creates the "Data/CMA_REPLICATION_STUDY/..." folder)
# REPS = 25
# BUDGET = 5000

# ROOT = "Data"
# ALG_NAME = "ModCMA_CMAES"  # <-- rename the algorithm here (shown in IOH Analyzer)

# for fid in range(1, 11):
#     for k in [2, 3, 5, 10]:

#         # One folder per (fid, k)
#         folder = f"fid{fid}_k{k}"

#         logger = ioh.logger.Analyzer(
#             root=ROOT,
#             folder_name=folder,
#             algorithm_name=ALG_NAME,
#             store_positions=True,
#             triggers=[ioh.logger.trigger.ALWAYS],
#         )

#         problem, _ = iohclustering.get_problem(fid=fid, instance=1, k=k)

#         problem.reset()
#         problem.attach_logger(logger)

#         for r in range(REPS):
#             problem.reset()
#             run_modcma_on_problem(problem, seed=r, budget=BUDGET)

#         problem.detach_logger()


# ## Analysis

# In[ ]:


# import iohinspector
# import pandas as pd
# DATA_PATH = "Data"

# dm = iohinspector.DataManager()
# dm.add_folder(DATA_PATH)

# dt = dm.overview.to_pandas()

# # The overview gridcma3 I had downloaded externally 
# # What does it do? 

# dt.to_csv("overview_cmaes.csv", index=False)
# dt = pd.read_csv("overview_cmaes.csv")
# print (dt.head(1000))
# # dt_agg = dt.groupby(['function_id', 'dimension', 'algorithm_name', 'lambda_', 'mu', 'elitist', 'bound_correction', 'covariance'])['best_y'].agg('mean').reset_index()
# # # The main problem with your current method is that you use the exact results from them which is not what we want 

# # # different aggregations of the following will lead to different results 
# # dt_agg.to_csv("CSVS/overview_gridcma3_agg.csv", index=False)
# # dt_agg = pd.read_csv("CSVS/overview_gridcma3_agg.csv")


# In[ ]:


# import iohinspector
# import inspect

# # Check DataManager class signature
# print("DataManager signature:")
# print(inspect.signature(iohinspector.DataManager))
# print("\n" + "="*50 + "\n")

# # Check if there's a docstring
# print("DataManager docstring:")
# print(iohinspector.DataManager.__doc__)
# print("\n" + "="*50 + "\n")

# # List available classes/functions
# print("Available in iohinspector:")
# print([item for item in dir(iohinspector) if not item.startswith('_')])


# In[ ]:


# import pandas as pd
# import json

# # First, parse algorithm_info JSON column
# dt_analysis = dt.copy()
# # The algorithm_info has double braces, so we need to replace them first
# dt_analysis['algorithm_info_parsed'] = dt_analysis['algorithm_info'].apply(lambda x: json.loads(x.replace('{{', '{').replace('}}', '}')))

# # Extract hyperparameters into separate columns
# for key in ['lambda_', 'mu', 'elitist', 'covariance', 'bound_correction']:
#     dt_analysis[key] = dt_analysis['algorithm_info_parsed'].apply(lambda x: x.get(key))

# # Replace NaN with "off" string
# dt_analysis = dt_analysis.fillna("off")

# # Analyze performance extremes across functions
# # Note: Since all hyperparameters are constant, grouping yields a single group per function
# dts = []
# for fid in dt_analysis['function_name'].unique():
#     dt_func = dt_analysis[dt_analysis['function_name'] == fid]

#     # Group by hyperparameters (should only have one group since all params are constant)
#     dt_agg2 = dt_func.groupby(["lambda_", "mu", "elitist", "covariance", "bound_correction"])['best_y'].mean().reset_index().sort_values("best_y", ascending=True)

#     best = dt_agg2['best_y'].min()
#     worst = dt_agg2['best_y'].max()

#     # Create dataframe with best and worst extremes
#     dt_extremes = pd.DataFrame({
#         "best_y": [best, worst],
#         "lambda_": [dt_agg2['lambda_'].iloc[0], dt_agg2['lambda_'].iloc[-1]],
#         "mu": [dt_agg2['mu'].iloc[0], dt_agg2['mu'].iloc[-1]],
#         "elitist": [dt_agg2['elitist'].iloc[0], dt_agg2['elitist'].iloc[-1]],
#         "covariance": [dt_agg2['covariance'].iloc[0], dt_agg2['covariance'].iloc[-1]],
#         "bound_correction": [dt_agg2['bound_correction'].iloc[0], dt_agg2['bound_correction'].iloc[-1]]
#     })

#     dt_extremes['fid'] = dt_func['function_id'].iloc[0]
#     dt_extremes['dim'] = dt_func['dimension'].iloc[0]
#     dt_extremes['dataset'] = fid.split("_")[1]
#     dts.append(dt_extremes)

# dt_extremes = pd.concat(dts, ignore_index=True)
# print("Performance extremes by function:")
# print(dt_extremes)


# ## The problem with the testing I'm doing 
# 
# It links back to the issue of being unable to **generate json file per each k value.**
# You've kept trying but for some reason each IOH profiler is linked to to single k value instead of an overall k value. 
# 
# Then the next problem with the data heat map difference 
# This is associated with only having a single `CMA-ES configuration` to work with. However could be possible to use the difference with `KMEANS` as a essentially a compromise with what you already have. 
# 
# ## Summary: Why Only One Dimension?
# 
# **The Problem:** Your original analysis only showed dimension=4 (19 unique values).
# 
# **The Root Cause:** You were grouping by `function_name` alone, which aggregated across ALL dimensions for each function. Since each function had 4 dimensions (4, 6, 10, 20), grouping by hyperparameters only created a single group per function that averaged performance across ALL dimensions.
# 
# **The Fix:** Group by **dimension** in addition to the hyperparameters. This preserves dimension-specific performance.
# 
# **Results:** After grouping by dimension, you now have:
# - **20 rows** (10 functions × 2 for best/worst)
# - **2 unique dimensions**: 4 and 20
#   - Dimension 4 tends to have worse (higher) `best_y` values
#   - Dimension 20 tends to have better (lower) `best_y` values
# 
# The analysis shows that for each dataset, performance varies significantly with dimensionality!

# In[ ]:


# import matplotlib.pyplot as plt
# import seaborn as sbs
# difference_data = (dt_extremes.pivot_table(index = "fid", columns = "dim", values = "best_y", aggfunc = "max") - dt_extremes.pivot_table(index = "fid", columns = "dim", values = "best_y", aggfunc = "min")) / (dt_extremes.pivot_table(index = "fid", columns = "dim", values = "best_y", aggfunc = "max"))
# plt.figure(figsize=(16, 9))
# sbs.heatmap(difference_data, annot=True, fmt=".4f", cmap="viridis")
# plt.xlabel('Dimension')
# plt.ylabel('Function ID')
# plt.yticks(rotation=0)
# plt.tight_layout()


# In[ ]:


# # KMeans calculation of values for the same functions and dimensions, to compare with the CMA-ES results.

# import pandas as pd
# dt_kmean = pd.read_csv("kmeans_baseline.csv")
# # dt_kmean['function_id'] = dt_kmean['dataset'].astype(int)
# dt_kmean['dimension'] = dt_kmean['k'].astype(int)*2

# dt_kmean.head()


# In[ ]:


# # NOTE: The comparison with Baseline KMeans is not worth indicating as the actual difference is very minimal between results 
# # Hence the main focus is gonna be the best value per each function and dimension

# print("="*70)
# print("COMPARISON: Before vs After Adding Dimension to Groupby")
# print("="*70)

# # ---- ORIGINAL APPROACH (loses dimension info) ----
# print("\n1. ORIGINAL: Groupby WITHOUT dimension")
# print("-" * 70)

# dt_analysis = dt.copy()
# dt_analysis['algorithm_info_parsed'] = dt_analysis['algorithm_info'].apply(lambda x: json.loads(x.replace('{{', '{').replace('}}', '}')))
# for key in ['lambda_', 'mu', 'elitist', 'covariance', 'bound_correction']:
#     dt_analysis[key] = dt_analysis['algorithm_info_parsed'].apply(lambda x: x.get(key))
# dt_analysis = dt_analysis.fillna("off")

# dts_original = []
# for fid in dt_analysis['function_name'].unique():
#     dt_func = dt_analysis[dt_analysis['function_name'] == fid]

#     # ORIGINAL: Only group by hyperparameters
#     dt_agg = dt_func.groupby(["lambda_", "mu", "elitist", "covariance", "bound_correction"])['best_y'].mean().reset_index().sort_values("best_y", ascending=True)

#     dt_ext = pd.DataFrame({
#         "best_y": [dt_agg['best_y'].min()],
#         "lambda_": [dt_agg['lambda_'].iloc[0]],
#         "dimension": [dt_func['dimension'].iloc[0]]  # Gets FIRST dimension, not meaningful!
#     })
#     dt_ext['function'] = fid
#     dts_original.append(dt_ext)

# dt_result_original = pd.concat(dts_original, ignore_index=True)
# print(f"Result: {len(dt_result_original)} rows")
# print(f"Unique dimensions: {dt_result_original['dimension'].unique()}")
# print(f"\nProblem: All rows have dimension={dt_result_original['dimension'].iloc[0]} because we pick .iloc[0]")
# print(dt_result_original[['function', 'dimension', 'best_y']].head(3))

# # ---- IMPROVED APPROACH (preserves dimension info) ----
# print("\n" + "="*70)
# print("2. IMPROVED: Groupby WITH dimension")
# print("-" * 70)

# dts_improved = []
# for fid in dt_analysis['function_name'].unique():
#     dt_func = dt_analysis[dt_analysis['function_name'] == fid]

#     # IMPROVED: Include dimension in groupby!
#     dt_agg = dt_func.groupby(["dimension", "lambda_", "mu", "elitist", "covariance", "bound_correction"])['best_y'].mean().reset_index().sort_values("best_y", ascending=True)

#     best = dt_agg['best_y'].min()
#     worst = dt_agg['best_y'].max()

#     # Now you get separate rows for each dimension!
#     for idx, row in dt_agg.iterrows():
#         dt_ext = pd.DataFrame({
#             "best_y": [row['best_y']],
#             "dimension": [row['dimension']],
#             "lambda_": [row['lambda_']],
#         })
#         dt_ext['function'] = fid
#         dts_improved.append(dt_ext)

# dt_result_improved = pd.concat(dts_improved, ignore_index=True)
# print(f"Result: {len(dt_result_improved)} rows")
# print(f"Unique dimensions: {sorted(dt_result_improved['dimension'].unique())}")
# print(f"\nNow you capture performance for EACH dimension!")
# print(dt_result_improved[['function', 'dimension', 'best_y']].head(8))

# # Currently not the difference between values but just the best_y for each dimension and function


# difference_data = dt_result_improved.pivot_table(index = "function", columns = "dimension", values = "best_y", aggfunc = "max") 
# plt.figure(figsize=(16, 9))
# # sbs.heatmap(difference_data, annot=True, fmt=".4f", cmap="viridis")
# plt.xlabel('Dimension')
# plt.ylabel('Function ID')
# plt.yticks(rotation=0)
# plt.tight_layout()


# ## Blade Function check how it works with clustering

# In[ ]:


# %pip install iohblade 


# In[ ]:


# NOTE: Make sure you never share this
# export OPENAI_API_KEY='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'


# In[ ]:


# if dependencies is None:
#     dependencies = ["numpy==2.2.3", "ioh==0.3.22", "iohclustering==0.0.3"]


# In[ ]:


# import iohblade, iohblade.problem as p
# print("iohblade version:", getattr(iohblade, "__version__", "unknown"))
# print("problem module:", p.__file__)
# print("Has WrappedProblem?", hasattr(p, "WrappedProblem"))
# print("Has wrap_problem?", hasattr(p, "wrap_problem"))
# print([name for name in dir(p) if "wrap" in name.lower() or "wrapped" in name.lower()])


# # The comparison csv file for kMeans 
# Feel free to adjust it _however_ you choose, primarily by picking mean vs min

# In[ ]:


# import pandas as pd
# df_Km = pd.read_csv("kmeans_baseline.csv")

# df_min = df_Km.groupby(['dataset', 'k']).agg('min')
# df_min.to_csv("KMeansCompare.csv")


# ## THIS IS THE CODE THAT WORKS

# In[ ]:


# 


# In[ ]:


# IMPORTANT to use ollama pull qwen3-coder:30b


# In[ ]:


# %pip install virtualenv


# In[ ]:


from iohblade.experiment import Experiment
from iohblade.llm import Ollama_LLM
from iohblade.methods import LLaMEA
import os
from iohblade.loggers import ExperimentLogger
# from iohblade import Problem, Solution, wrap_problem
import numpy as np
from pathlib import Path
import os
import logging

from IOHClusteringAdapter import IOHClusteringProblemAdapter  # <-- key change

# (optional) reduce joblib multi-proc weirdness in notebooks
# os.environ["JOBLIB_MULTIPROCESSING"] = "0"

mutation_prompts = ["""An example of such code (a simple random search) for an IOH *clustering* objective is as follows.
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

return f_opt, x_opt ```
"""
,""" ```python
import numpy as np

class NeighborhoodAdaptiveDE:
def __init__(self, budget = 2000, dim, seed = None):
self.budget: int = budget
self.dim: int = dim
self.seed = seed

self.success_f: list = []
self.success_cr: list = []
self.best_fitness_history: list = []

# Hyperparameters that can be tuned
self.pop_size: int = 50
self.adapt_freq: int = 50
self.stagnation_threshold: int = 1000
self.learning_rate: float = 0.1
self.neighborhood_size: int = 5
self.f: float = 0.5
self.cr: float = 0.9

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
# Neighborhood-based Mutation
neighbors = np.random.choice(np.arange(self.pop_size), self.neighborhood_size, replace=False)
best_neighbor_idx = neighbors[np.argmin(self.fitness[neighbors])]

idxs = np.random.choice(np.arange(self.pop_size), 2, replace=False)
x1, x2 = self.population[idxs]

v = self.population[best_neighbor_idx] + self.f * (x1 - x2)
v = np.clip(v, func.bounds.lb, func.bounds.ub)

# Orthogonal Crossover
u = np.copy(self.population[i])
j_rand = np.random.randint(self.dim)
for j in range(self.dim):
    if np.random.rand() < self.cr or j == j_rand:
        u[j] = v[j]

f_new = func(u)
self.evals += 1

# Selection
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

# Adaptive Parameter Control
if self.evals % self.adapt_freq == 0:
if self.success_f:
self.f = (1 - self.learning_rate) * self.f + self.learning_rate * np.mean(self.success_f)
self.cr = (1 - self.learning_rate) * self.cr + self.learning_rate * np.mean(self.success_cr)
self.f = np.clip(self.f, 0.1, 0.9)
self.cr = np.clip(self.cr, 0.1, 1.0)
self.success_f = []
self.success_cr = []

# Stagnation Check and Restart
if self.evals - self.last_improvement > self.stagnation_threshold:
self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
self.fitness = np.array([func(x) for x in self.population])
self.evals += self.pop_size
best_idx = np.argmin(self.fitness)
f_opt = self.fitness[best_idx]
x_opt = self.population[best_idx]
self.last_improvement = self.evals
self.best_fitness_history.append(f_opt)

return f_opt, x_opt```
""", 
"""```python
import numpy as np

class ParticleSwarmOptimization:
def __init__(self, budget=10000, dim, seed=None):
self.budget = budget
self.dim = dim
self.rng = np.random.default_rng(seed)

# Hyperparameters, can be tuned
self.s: int = 50
self.inertia: float = 0.5
self.psi_p: float = 1.5
self.psi_g: float = 1.5

def __call__(self, func) :
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
velocity = ((self.inertia * velocity) + (self.psi_p * rp * (best_known_positions - positions)) + (self.psi_g * rg * (x_opt - positions)))

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

return f_opt, x_opt```""", """```python
import numpy as np
import math

class UMDA:
def __init__(self, budget, dim, seed=None):
self.budget = budget
self.dim = dim
self.rng = np.random.default_rng(seed)
# Hyperparameters
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

mu = s // 2  # number of parents selected
idx_sorted = np.argsort(-fitness)
parents = pop[idx_sorted[:mu]]

while func.state.evaluations + s <= self.budget:
# Estimate marginal probabilities
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

return best_f, best_x```""", """```python 
import numpy as np

class GreedyHillClimber:
def __init__(self, budget, dim, seed=None):
self.budget = budget
self.dim = dim
self.rng = np.random.default_rng(seed)

def __call__(self, func):  # Run the algorithm on one problem instance, `func` is a function object (callable).
self.mutation_rate = (1.0 / self.dim)
x = self.rng.integers(0, 2, self.dim)
fx = func(x)
idx = 0
while func.state.evaluations < self.budget:
y = x.copy()
y[idx] = 1 - y[idx]  # flip bits for 0/1 domain
idx = (idx + 1) % self.dim
fy = func(y)

if fy > fx:
x, fx = y, fy

if fy >= func.optimum.y:
break
return fx, x```"""]

llm = Ollama_LLM("qwen3-coder:30b")

method = LLaMEA(
    llm,
    budget=200,          # LLaMEA algorithm-count budget
    name="LLaMEA",
    n_parents=6,
    mutation_prompts = mutation_prompts,
    n_offspring=6,    # How many children should I have? 
    elitism=True, # Important (suggested by Niki to change)
    minimization=False, 
)

methods = [method]

FIDS = range(1, 3)
K_ALL = [2, 3, 5, 10]
# I just checked, with your current eval method you never run testing
# so maybe we just check how the best algorithms perform on different instances of the train and test set? 
# Remains to be seen, but regardless you should run all the datasets at least once 
train_fid = range(1, 9)
test_fid  = range(9, 11)

# keeping the instances simple for now as it will be my first time running over the entire dataset
IID = 1

train_instances = [(fid, IID, k) for fid in train_fid for k in K_ALL]
test_instances  = [(fid, IID, k) for fid in test_fid for k in K_ALL]

# What's up with the iids stuff? 
# Just change the fid for test instances so it doesn't correspond to the same data set
# You don't know how the distribution changes of using iid with clustering problems 


# While you are assigning the training and test instances here you need to reset these to be the actual data points in your python file 

problem = IOHClusteringProblemAdapter(
    training_instances=train_instances,
    test_instances=test_instances,
    name="CLUSTERING",
    base_budget=25,
    budget_scale_with_dim=True,
    eval_timeout=120,
    full_ioh_log=False,
    ioh_dir="",  # or "ioh_logs"
)

logger = ExperimentLogger("results/clustering_mutation2")
problems = [problem] # our dummy problem

experiment = Experiment(
    methods=methods,
    problems=problems,
    runs=3,
    show_stdout=True,     # note: LLaMEA redirects stdout internally; your debug_eval.log is the truth
    exp_logger=logger,
    budget=200,           # IMPORTANT: must be >= LLaMEA_method.budget, ideally a bit higher
)

experiment()


# In[ ]:


# import os, time, traceback, math
# import numpy as np
# import ioh
# import iohclustering

# from iohblade.solution import Solution
# from iohblade.utils import OverBudgetException, aoc_logger, correct_aoc
# from my_wrapped_problem import WrappedProblem  # from the .py file

# def make_clustering_wrapped_problem(
#     training_instances,
#     test_instances,
#     *,
#     name="CLUSTERING",
#     eval_timeout=120,
#     base_budget=5,
#     budget_scale_with_dim=True,
#     full_ioh_log=False,
#     ioh_dir="",
#     dependencies=None,
#     imports=None,
# ):
#     # IMPORTANT: these deps/imports are installed/used inside iohblade's evaluation venv
#     if dependencies is None:
#         dependencies = ["numpy==2.2.3", "ioh==0.3.22", "iohclustering"]
#     if imports is None:
#         imports = "import numpy as np\nimport math\nimport ioh\nimport iohclustering\nimport traceback\nimport os\nimport time\n"

#     def _make_problem(fid: int, iid: int, k: int):
#         p, _ = iohclustering.get_problem(fid=fid, instance=iid, k=k)
#         return p

#     def _infer_dim(p):
#         md = getattr(p, "meta_data", None)
#         if md is not None and hasattr(md, "n_variables"):
#             return int(md.n_variables)
#         if hasattr(p, "dimension"):
#             return int(p.dimension)
#         b = getattr(p, "bounds", None)
#         if b is not None and hasattr(b, "lb"):
#             return int(np.asarray(b.lb).size)
#         return None

#     def _budget(p):
#         if not budget_scale_with_dim:
#             return int(base_budget)
#         dim = _infer_dim(p)
#         return int(base_budget if dim is None else base_budget * dim)

#     def _debug_path(self):
#         # Where to write debug logs: prefer logger dir if available
#         base = getattr(getattr(self, "logger", None), "get_log_dir", lambda: "")()
#         if not base:
#             base = os.getcwd()
#         os.makedirs(base, exist_ok=True)
#         return os.path.join(base, "debug_eval.log")

#     def evaluate_fn(self, solution: Solution):
#         log_path = _debug_path(self)
#         def log(msg):
#             with open(log_path, "a") as f:
#                 f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")

#         log(f"evaluate() called solution={solution.id} gen={solution.generation}")

#         # ---- 1) exec generated code ----
#         code = solution.code or ""
#         algo_name = solution.name or ""

#         safe_globals = {"np": np, "math": math, "ioh": ioh}
#         local_env = {}
#         try:
#             exec(code, safe_globals, local_env)
#         except Exception as e:
#             log(f"exec failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
#             solution.set_scores(-np.inf, f"exec() failed: {type(e).__name__}: {e}")
#             return solution

#         if algo_name not in local_env:
#             log(f"Algo '{algo_name}' not found. local_env keys={list(local_env.keys())}")
#             solution.set_scores(-np.inf, f"Algorithm class '{algo_name}' not found after exec().")
#             return solution

#         Algo = local_env[algo_name]

#         # ---- 2) smoke test on first training instance ----
#         try:
#             fid0, iid0, k0 = training_instances[0]
#             p0 = _make_problem(fid0, iid0, k0)
#             b0 = max(10, min(50, _budget(p0)))

#             l2_tmp = aoc_logger(b0, upper=1e2, triggers=[ioh.logger.trigger.ALWAYS])
#             p0.attach_logger(l2_tmp)

#             try:
#                 alg0 = Algo(budget=b0)
#             except TypeError:
#                 dim0 = _infer_dim(p0) or 0
#                 alg0 = Algo(budget=b0, dim=dim0)

#             alg0(p0)

#         except OverBudgetException:
#             # ok; still “ran”
#             pass
#         except Exception as e:
#             log(f"smoke failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
#             try: p0.reset()
#             except Exception: pass
#             solution.set_scores(-np.inf, f"Smoke test failed: {type(e).__name__}: {e}")
#             return solution
#         finally:
#             try: p0.reset()
#             except Exception: pass

#         # ---- 3) main evaluation on all training instances ----
#         aucs = []
#         perf = []

#         for (fid, iid, k) in training_instances:
#             p = _make_problem(fid, iid, k)
#             b = _budget(p)

#             l2 = aoc_logger(b, upper=1e2, triggers=[ioh.logger.trigger.ALWAYS])

#             try:
#                 if full_ioh_log:
#                     l1 = ioh.logger.Analyzer(
#                         root=ioh_dir if ioh_dir else "ioh_logs",
#                         folder_name=solution.id,
#                         algorithm_name=solution.id,
#                         store_positions=True,
#                         triggers=[ioh.logger.trigger.ALWAYS],
#                     )
#                     p.attach_logger(ioh.logger.Combine([l1, l2]))
#                 else:
#                     p.attach_logger(l2)

#                 try:
#                     alg = Algo(budget=b)
#                 except TypeError:
#                     dim = _infer_dim(p) or 0
#                     alg = Algo(budget=b, dim=dim)

#                 alg(p)

#             except OverBudgetException:
#                 pass
#             except Exception as e:
#                 log(f"eval failed fid={fid} iid={iid} k={k}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
#                 perf.append({"fid": fid, "iid": iid, "k": k, "auc": 0.0, "error": f"{type(e).__name__}: {e}"})
#                 aucs.append(0.0)
#                 try: p.reset()
#                 except Exception: pass
#                 continue
#             finally:
#                 try: p.reset()
#                 except Exception: pass

#             try:
#                 corrected = correct_aoc(p, l2, b)
#             except Exception as e:
#                 log(f"correct_aoc failed fid={fid} iid={iid} k={k}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
#                 corrected = 0.0

#             perf.append({"fid": fid, "iid": iid, "k": k, "auc": float(corrected)})
#             aucs.append(float(corrected))

#             try: l2.reset(p)
#             except Exception: pass

#         auc_mean = float(np.mean(aucs)) if aucs else 0.0
#         solution.add_metadata("performance_data", perf)
#         solution.add_metadata("aucs", aucs)
#         solution.set_scores(auc_mean, f"AOCC mean={auc_mean:.4f} over {len(aucs)} instances (higher is better).")
#         log(f"evaluate() done solution={solution.id} fitness={solution.fitness}")

#         forbidden = ["import ioh", "from ioh", "ioh.ioh", "iohcpp", "ProblemAdapter"]
#         if any(tok in code for tok in forbidden):
#             solution.set_scores(-np.inf, "Forbidden IOH import detected; use func only.")
#             return solution
#         return solution

#     return WrappedProblem(
#         evaluate_fn,
#         name=name,
#         eval_timeout=eval_timeout,
#         training_instances=list(training_instances),
#         test_instances=list(test_instances),
#         dependencies=dependencies,
#         imports=imports,
#         task_prompt="You are given an IOH clustering optimization problem adapter. Additionally the domain knowledge you have of clustering problems is that the major challenge of these types of problems is regions of neutrality and symmetry. Implement a robust evaluate_solution(x, attach_logger=None) method that: validates x is a numeric vector of length equal to the problem dimension; obtains or creates the underlying IOH problem instance (use get_problem/get_problem(fid,iid,k) or the adapter's _make_problem_for_run()); call the IOH problem via problem(x) to evaluate the objective; reset the problem state after evaluation (call problem.reset() if available); return the objective as a Python float. Keep side-effects minimal, handle exceptions (reset on error), and raise ValueError for wrong input size. ",     # you can set prompts later if you want
#         example_prompt="""An example of such code (a simple random search) for an IOH *clustering* objective is as follows.
#         Note: for clustering problems, the dimension can be inferred from the bounds (`len(func.bounds.lb)`), so `dim` is optional.

#         ```python
#         import numpy as np
#         import math

#         class RandomSearch:
#             def __init__(self, budget=10000, dim=None):
#                 self.budget = budget
#                 self.dim = dim  # optional; can be inferred from func.bounds

#             def __call__(self, func):
#                 # infer dimension if not provided
#                 if self.dim is None:
#                     self.dim = len(func.bounds.lb)

#                 self.f_opt = np.inf
#                 self.x_opt = None

#                 # uniform sampling within the continuous search space
#                 for _ in range(self.budget):
#                     x = np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim)
#                     f = func(x)  # clustering objective evaluation

#                     if f < self.f_opt:
#                         self.f_opt = f
#                         self.x_opt = x

#         return self.f_opt, self.x_opt)""" )


# In[ ]:


# from iohblade.experiment import Experiment
# from iohblade.llm import Ollama_LLM
# from iohblade.methods import LLaMEA
# import os
# #
# os.environ["JOBLIB_MULTIPROCESSING"] = "0"

# llm = Ollama_LLM("qwen3-coder:30b")

# LLaMEA_method = LLaMEA(
#     llm,
#     budget=124,
#     name="LLaMEA",
#     n_parents=2,
#     n_offspring=6,
#     elitism=False,
#     minimization=True,
# )

# methods = [LLaMEA_method]

# FIDS = range(1, 3)
# K_ALL = [2, 3]
# train_iids = range(1, 3)
# test_iids  = range(3, 5)

# train_instances = [(fid, iid, k) for fid in FIDS for k in K_ALL for iid in train_iids]
# test_instances  = [(fid, iid, k) for fid in FIDS for k in K_ALL for iid in test_iids]

# problem = make_clustering_wrapped_problem(
#     train_instances,
#     test_instances,
#     name="CLUSTERING",
#     base_budget=5,
#     budget_scale_with_dim=True,
#     eval_timeout=120,
#     full_ioh_log=False,
#     ioh_dir="",
# )

# experiment = Experiment(
#     methods=methods,
#     problems=[problem],      # <-- WrappedProblem instance
#     llm=llm,
#     runs=1,
#     show_stdout=True,
#     log_dir="Results",
#     budget=80,               # Experiment budget >= method budget is fine, but not required to match exactly
# )

# experiment()

