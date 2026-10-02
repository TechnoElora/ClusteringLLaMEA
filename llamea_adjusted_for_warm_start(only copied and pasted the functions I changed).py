"""LLaMEA - LLM powered Evolutionary Algorithm for code optimization
This module integrates OpenAI's language models to generate and evolve
algorithms to automatically evaluate (for example metaheuristics evaluated on BBOB).
"""

import concurrent.futures
import contextlib
import logging
import math
import os
import random
import re
import traceback
import warnings
from typing import Callable, Optional
import pickle
import jsonlines

import numpy as np
from joblib import Parallel, delayed

from .llm import LLM
from .feature_guidance import FeatureGuidance, compute_feature_guidance
from .ast_features import extract_ast_features

try:
    from ConfigSpace import ConfigurationSpace
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ConfigurationSpace = None

from .loggers import ExperimentLogger
from .solution import Solution
from .utils import (
    NoCodeException,
    code_distance,
    discrete_power_law_distribution,
    handle_timeout,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class LLaMEA:
    """
    A class that represents the Language Model powered Evolutionary Algorithm (LLaMEA).
    This class handles the initialization, evolution, and interaction with a language model
    to generate and refine algorithms.
    """

    def __init__(
        self,
        f,
        llm,
        n_parents=5,
        n_offspring=5,
        role_prompt="",
        task_prompt="",
        example_prompt=None,
        example_prompts = None,
        output_format_prompt=None,
        experiment_name="",
        elitism=True,
        HPO=False,
        mutation_prompts=None,
        adaptive_mutation=False,
        adaptive_prompt=False,
        feature_guided_mutation: bool = False,
        budget=100,
        eval_timeout=3600,
        max_workers=10,
        parallel_backend="loky",
        log=True,
        minimization=False,
        _random=False,
        niching: Optional[str] = None,
        distance_metric: Optional[Callable[[Solution, Solution], float]] = None,
        niche_radius: Optional[float] = None,
        adaptive_niche_radius: bool = False,
        clearing_interval: Optional[int] = None,
        behavior_descriptor: Optional[Callable[[Solution], tuple]] = None,
        map_elites_bins: Optional[tuple[int, ...] | int] = None,
        novelty_k: int = 5,
        novelty_archive_size: Optional[int] = 100,
        evaluate_population=False,
        diff_mode: bool = False,
        parent_selection: str = "random",
        tournament_size: int = 3,
    ):
        """
        Initializes the LLaMEA instance with provided parameters. Note that by default LLaMEA maximizes the objective.

        Args:
            f (callable): The evaluation function to measure the fitness of algorithms.
            llm (object): An instance of a language model that will be used to generate and evolve algorithms.
            n_parents (int): The number of parents in the population.
            n_offspring (int): The number of offspring each iteration.
            elitism (bool): Flag to decide if elitism (plus strategy) should be used in the evolutionary process or comma strategy.
            role_prompt (str): A prompt that defines the role of the language model in the optimization task.
            task_prompt (str): A prompt describing the task for the language model to generate optimization algorithms.
            example_prompt (str): An example prompt to guide the language model in generating code (or None for default).
            output_format_prompt (str): A prompt that specifies the output format of the language model's response.
            experiment_name (str): The name of the experiment for logging purposes.
            elitism (bool): Flag to decide if elitism should be used in the evolutionary process.
            HPO (bool): Flag to decide if hyper-parameter optimization is part of the evaluation function.
                In case it is, a configuration space should be asked from the LLM as additional output in json format.
            mutation_prompts (list): A list of prompts to specify mutation operators to the LLM model. Each mutation, a random choice from this list is made.
            adaptive_mutation (bool): If set to True, the mutation prompt 'Change X% of the lines of code' will be used in an adaptive control setting.
                This overwrites mutation_prompts.
            adaptive_prompt (bool): If True, the task prompt is optimized before each mutation, allowing it to co-evolve with the individuals.
            feature_guided_mutation (bool): Enable archive based mutation guidance that
                augments mutation prompts using XGBoost and TreeSHAP insights.
            budget (int): The number of generations to run the evolutionary algorithm.
            eval_timeout (int): The number of seconds one evaluation can maximum take (to counter infinite loops etc.). Defaults to 1 hour.
            max_workers (int): The maximum number of parallel workers to use for evaluating individuals.
            parallel_backend (str): The backend to use for parallel processing (e.g., 'loky', 'threading').
            log (bool): Flag to switch of the logging of experiments.
            minimization (bool): Whether we minimize or maximize the objective function. Defaults to False.
            _random (bool): Flag to switch to random search (purely for debugging).
            niching (str | None): Niching strategy to use. Supports "sharing",
                "clearing", and "map_elites". If ``None``, niching is disabled.
            distance_metric (callable | None): Function that computes a distance
                between two :class:`Solution` objects. Defaults to a simple AST
                based distance if not supplied.
            niche_radius (float | None): Radius for niche determination when
                using fitness sharing or clearing. If ``None`` a default of ``0.5``
                is used when a niching method is active.
            adaptive_niche_radius (bool): If ``True`` the niche radius adapts to
                the population each generation.
            clearing_interval (int | None): Interval (in generations) at which
                clearing is applied when ``niching`` is set to ``"clearing"``.
            behavior_descriptor (callable | None): Function returning a tuple
                that characterizes a :class:`Solution` for MAP-Elites. If not
                provided the descriptor is derived from basic code statistics.
            map_elites_bins (tuple[int, ...] | int | None): Number of bins per
                dimension for the MAP-Elites archive. If ``None`` a default of
                ten bins per descriptor dimension is used.
            novelty_k (int): Number of nearest neighbours considered when
                computing novelty scores if ``niching`` is set to
                ``"novelty"``.
            novelty_archive_size (int | None): Maximum size of the novelty
                archive when ``niching`` is set to ``"novelty"``. ``None``
                keeps all past individuals.
            evaluate_population (bool): If True, the evaluation function `f` should
                accept a list with the new population and a list of parents (optionally, to deal with elitism)
                and return a list of solutions that are evaluated, also the parents may receive new fitness values and should be returned.
                So `f` should have the signature
                `f(population, parents=None, logger=None) -> (evaluated_offspring, evaluated_parents)`.
            diff_mode (bool): If ``True``, the LLM is asked to generate unified diff
                patches instead of complete code when evolving solutions.
            parent_selection (str): Strategy for selecting parents to produce
                offspring. Options are ``"random"``, ``"tournament"``, which
                picks the best from a sampled subset, and fitness-proportionate
                roulette wheel selection ``"roulette"``.
            tournament_size (int): Number of candidates sampled in each
                tournament when using ``"tournament"`` selection.
        """
        self.llm = llm
        self.model = llm.model
        self.diff_mode = diff_mode

        self.eval_timeout = eval_timeout
        self.f = f  # evaluation function, provides an individual as output.
        self.role_prompt = role_prompt
        self.parallel_backend = parallel_backend
        if HPO and ConfigurationSpace is None:
            warnings.warn(
                "ConfigSpace is not installed. Install ConfigSpace to enable HPO.",
                stacklevel=2,
            )
            HPO = False
        if role_prompt == "":
            self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
        if task_prompt == "":
            self.task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code to minimize the function value. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.

Give an excellent and novel heuristic algorithm to solve this task.
"""
        else:
            self.task_prompt = task_prompt

        if example_prompt == None:
            self.example_prompt = """
An example of such code (a simple random search), is as follows:
```
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)

            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt
```
"""
        else:
            self.example_prompt = example_prompt

        self.example_prompts = example_prompts

        if output_format_prompt is None:
            self.output_format_prompt = """
Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
"""
            if HPO:
                self.output_format_prompt = """
Provide the Python code, a one-line description with the main idea (without enters) and the SMAC3 Configuration space to optimize the code (in Python dictionary format). Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
Space: <configuration_space>"""
        else:
            self.output_format_prompt = output_format_prompt
        self.diff_output_format_prompt = """
        ---
You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:
```
# Original code to find and replace (must match exactly)
```

Example of valid diff format:
```
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
```
"""
        self.mutation_prompts = mutation_prompts
        self.adaptive_mutation = adaptive_mutation
        if mutation_prompts == None:
            self.mutation_prompts = [
                "Refine the strategy of the selected solution to improve it.",  # small mutation
                # "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
            ]
        self.budget = budget
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.population = []
        self.elitism = elitism
        self.generation = 0
        self.run_history = []
        self.log = log
        self._random = _random
        self.HPO = HPO
        self.minimization = minimization
        self.evaluate_population = evaluate_population
        self.adaptive_prompt = adaptive_prompt
        self.feature_guided_mutation = feature_guided_mutation
        self.feature_guidance: FeatureGuidance | None = None
        self.feature_guidance_message = ""
        self.worst_value = -np.inf
        if minimization:
            self.worst_value = np.inf
        if niching == "novelty" and self.minimization:
            raise ValueError("Novelty niching only supports maximization.")
        self.niching = niching
        self.distance_metric = distance_metric or code_distance
        self.niche_radius = niche_radius if niche_radius is not None else 0.5
        self.adaptive_niche_radius = adaptive_niche_radius
        self.clearing_interval = clearing_interval
        self.behavior_descriptor = behavior_descriptor
        self.map_elites_bins = map_elites_bins
        self.map_elites_archive = {}
        self.map_elites_bounds = None
        self.map_elites_descriptor_cache = {}
        self.map_elites_solutions = {}
        self.novelty_k = max(1, int(novelty_k))
        self.novelty_archive_size = (
            None if novelty_archive_size is None else max(1, int(novelty_archive_size))
        )
        self.novelty_archive: list[Solution] = []
        self.best_so_far = Solution(name="", code="")
        self.best_so_far.set_scores(self.worst_value, "")
        self.experiment_name = experiment_name
        self.parent_selection = parent_selection  # "random" | "roulette" | "tournament"
        self.tournament_size = tournament_size

        if self.log:
            modelname = self.model.replace(":", "_")
            modelname = self.model.replace("/", "_")
            self.logger = ExperimentLogger(f"LLaMEA-{modelname}-{experiment_name}")
            self.llm.set_logger(self.logger)
        else:
            self.logger = None
        if max_workers > self.n_offspring:
            max_workers = self.n_offspring
        self.max_workers = max_workers
        self.pickle_archive()

    @classmethod
    def warm_start(cls, path_to_archive_dir):
        """
        Class method for warm starts, takes a archive directory, and finds pickle archieve stored at path_to_archieve_dir/llamea_config.pkl,
        generates the object from it, and return it for warm start.
        Args:
            path_to_archive_dir: Directory of instance for which warm start needs to be executed.
        """
        try:
            with open(f"{path_to_archive_dir}/llamea_config.pkl", "rb") as file:
                obj = pickle.load(file)
            return obj
        except Exception as e:
            print(
                f"Error unarchiving object from {path_to_archive_dir}/llamea_config.pkl: {e.__repr__()}"
            )
            return None

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, to_state):
        self.__dict__.update(to_state)

    def logevent(self, event):
        print(event)

    def initialize_single(self, example_prompt=None, example_prompt_index=0):
        """
        Initializes a multiple prompts/single if not using multiple.
        """
        new_individual = Solution(name="", code="", generation=self.generation)


        # Fall back to original single prompt if not provided
        example_prompt = self.example_prompt if example_prompt is None else example_prompt


        session_messages = [
            {
                "role": "user",
                "content": self.role_prompt
                    + self.task_prompt
                    + example_prompt
                    + self.output_format_prompt,
            },
        ]
        try:
            new_individual = self.llm.sample_solution(session_messages, HPO=self.HPO)
            new_individual.generation = self.generation
            new_individual.task_prompt = self.task_prompt
            new_individual.add_metadata("example_prompt_index", example_prompt_index)
            if not self.evaluate_population:
                new_individual = self.evaluate_fitness(new_individual)
        except Exception as e:
            new_individual.set_scores(self.worst_value, "", e)
            self.logevent(f"An exception occured: {traceback.format_exc()}.")
            if hasattr(self.f, "log_individual"):
                self.f.log_individual(new_individual)
        print("Releasing individual", new_individual)
        return new_individual


    def initialize(self):
        """
        Initializes the evolutionary process by generating the first parent population.
        """
        population = self.population
        population_gen = []


        # new change to work with multiple ex prompts
        # pre-assign example prompts to each slot before parallelism


        n_needed = self.n_parents - len(population)


        # fall back to single example_prompt if example_prompts is missing/empty
        example_prompts = getattr(self, "example_prompts", None) or [self.example_prompt]


        assigned_examples = [
            (i % len(example_prompts), example_prompts[i % len(example_prompts)])
            for i in range(n_needed)
        ]


        try:
            timeout = self.eval_timeout
            population_gen = Parallel(
                n_jobs=self.max_workers,
                backend=self.parallel_backend,
                timeout=timeout + 15,
                return_as="generator_unordered",
            )(
                delayed(self.initialize_single)(example_prompt=example, example_prompt_index=idx)
                for idx, example in assigned_examples
            )
        except Exception as e:
            print(f"Parallel time out in initialization {e}, retrying.")


        for p in population_gen:
            if math.isnan(p.fitness):
                p.fitness = self.worst_value
            population.append(p)


        if self.evaluate_population:
            population = self.evaluate_population_fitness(population)


        for p in population:
            self.run_history.append(p)


        if self.niching == "novelty":
            population = self.apply_niching(population)


        self.generation += 1
        self.population = population  # Save the entire population
        self.update_best()