from abc import ABC, abstractmethod

from aide import backend
from aide.agent import Agent
from omegaconf import OmegaConf
from aide.journal import Journal
from aide.interpreter import Interpreter
from aide.callbacks.manager import CallbackManager
from aide.utils.config import load_task_desc, prep_agent_workspace
from aide.utils.serialize import load_code_file, load_json


class Workflow(ABC):
    """Base class for all workflows."""

    def __init__(self, cfg, callback_manager=None):
        """
        Initialize the workflow.
        """
        self.cfg = cfg
        self.callback_manager = callback_manager
        task_desc = load_task_desc(self.cfg)
        task_desc_str = backend.compile_prompt_to_md(task_desc)

        prep_agent_workspace(cfg)

        self.journal = Journal()

        self.agent = Agent(
            task_desc=task_desc_str,
            cfg=cfg,
            journal=self.journal,
        )

        self.interpreter = Interpreter(
            self.cfg.workspace_dir, **OmegaConf.to_container(self.cfg.exec)  # type: ignore
        )

        if self.cfg.initial_solution.exp_name is not None:
            journal_json = (
                self.cfg.log_dir.parent / self.cfg.initial_solution.exp_name / "journal.json"
            ).resolve()
            prev_journal = load_json(journal_json, Journal)
            if cfg.initial_solution.node_id is not None:
                node = prev_journal.get(cfg.initial_solution.node_id)
            else:
                node = prev_journal.get_best_node()
            if node is not None:
                self.agent.journal.append(node)
                
        elif self.cfg.initial_solution.code_file is not None:
            assert (
                self.cfg.initial_solution.node_id is None
                and self.cfg.initial_solution.exp_name is None
            ), "Please specify either code_file or a combination of exp_name and node_id. Specifying both is not allowed."
            node = load_code_file(self.cfg.initial_solution.code_file)
            if node:
                # TODO: Remove this from here once the proper place to set load this file has been identified
                exec_result = self.interpreter.run(code=node.code)
                self.agent.parse_exec_result(node=node, exec_result=exec_result, max_attempts=0)
                self.agent.journal.append(node)
        
        self.callback_manager = (
            callback_manager if callback_manager is not None else CallbackManager()
        )
        try:
            self.callback_manager.callbacks["exec"]
        except KeyError:
            self.callback_manager.register_callback("exec", self.interpreter.run)

    @abstractmethod
    def run(self):
        """Execute the workflow."""
        pass
