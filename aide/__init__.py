from dataclasses import dataclass
from .version import __version__ as __version__
from .agent import Agent
from .interpreter import Interpreter
from .journal import Journal
from omegaconf import OmegaConf
from rich.status import Status
from .utils.config import (
    load_task_desc,
    prep_agent_workspace,
    save_run,
    _load_cfg,
    prep_cfg,
)
import traceback
import asyncio
from aide.callbacks.manager import CallbackManager


@dataclass
class Solution:
    code: str
    valid_metric: float


class Experiment:

    def __init__(self, data_dir: str, goal: str, eval: str | None = None):
        """Initialize a new experiment run.

        Args:
            data_dir (str): Path to the directory containing the data files.
            goal (str): Description of the goal of the task.
            eval (str | None, optional): Optional description of the preferred way for the agent to evaluate its solutions.
        """

        _cfg = _load_cfg(use_cli_args=False)
        _cfg.data_dir = data_dir
        _cfg.goal = goal
        _cfg.eval = eval
        self.cfg = prep_cfg(_cfg)

        self.task_desc = load_task_desc(self.cfg)

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(self.cfg)

        self.journal = Journal()
        self.agent = Agent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
        )
        self.interpreter = Interpreter(
            self.cfg.workspace_dir, **OmegaConf.to_container(self.cfg.exec)  # type: ignore
        )

    def run(self, steps: int) -> Solution:
        print("Starting experiment run")
        for _i in range(steps):
            print(f"Step {_i+1}/{steps}: About to call agent.step()")
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # Create a new event loop for each step
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                callback_manager = CallbackManager()
                callback_manager.register_callback("exec", self.interpreter.run)

                loop.run_until_complete(
                    self.agent.step(
                        exec_callback=self.interpreter.run,
                        callback_manager=callback_manager,
                    )
                )
                print(f"Step {_i+1} completed")
            except Exception as e:
                print(f"Error in agent.step: {e}")
                print(traceback.format_exc())
            save_run(self.cfg, self.journal)

            print(f"Journal has {len(self.journal.nodes)} nodes")

        print("Cleanup session")
        self.interpreter.cleanup_session()

        print("Getting best node")
        best_node = self.journal.get_best_node()
        print(f"Best node: {best_node}")

        if best_node is None:
            print("No successful nodes were found")
            return Solution(
                code="# No successful solution found", valid_metric=float("-inf")
            )

        return Solution(code=best_node.code, valid_metric=best_node.metric.value)
