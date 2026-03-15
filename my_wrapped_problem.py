# my_wrapped_problem.py
import inspect
import cloudpickle
import iohblade
from iohblade.problem import Problem
from iohblade.solution import Solution

class WrappedProblem(Problem):
    def __init__(
        self,
        evaluate_fn,
        *,
        name="Problem",
        eval_timeout=600,
        training_instances=None,
        test_instances=None,
        dependencies=None,
        imports=None,
        task_prompt="",
        example_prompt="",
        logger=None,
    ):
        super().__init__(
            logger=logger,
            training_instances=training_instances,
            test_instances=test_instances,
            name=name,
            eval_timeout=eval_timeout,
        )
        # `Problem.__init__` in current iohblade versions does not accept
        # `dependencies` or `imports` kwargs. Store them on the instance
        # so other code can still access them.
        self.dependencies = dependencies
        self.imports = imports
        if task_prompt:
            self.task_prompt = task_prompt
        if example_prompt:
            self.example_prompt = example_prompt

        self._takes_self = len(inspect.signature(evaluate_fn).parameters) > 1
        self._evaluate_fn_bytes = cloudpickle.dumps(evaluate_fn)
        self._evaluate_fn = None
        self.format_prompt = """
            Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description, describing the main idea. Give the response in the format:
            # Description: <short-description>
            # Code: 
            ```python
            <code>
            ```
            """

    def get_prompt(self):
        return (self.task_prompt or "") + (self.example_prompt or "") + (self.format_prompt or "")
    
    def _get_evaluate_fn(self):
        if self._evaluate_fn is None:
            self._evaluate_fn = cloudpickle.loads(self._evaluate_fn_bytes)
        return self._evaluate_fn

    def evaluate(self, solution: Solution):
        fn = self._get_evaluate_fn()
        return fn(self, solution) if self._takes_self else fn(solution)

    def test(self, solution: Solution):
        return self.evaluate(solution)

    def to_dict(self):
        return {
            "name": self.name,
            "eval_timeout": self.eval_timeout,
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
            "dependencies": self.dependencies,
            "imports": self.imports,
        }
