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
