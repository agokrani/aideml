from aide.callbacks.manager import CallbackManager
from aide.workflow.base import Workflow
from aide.agent import Agent
from aide.interpreter import Interpreter
from aide.journal import Journal
from aide.utils.config import Config
from aide.utils.config import save_run


class AutoPilot(Workflow):
    def __init__(self, agent: Agent, interpreter: Interpreter, cfg: Config, callback_manager: CallbackManager|None = None):
        self.agent = agent
        self.cfg = cfg
        self.interpreter = interpreter
        self.journal = agent.journal
        self.callback_manager = callback_manager if callback_manager is not None else CallbackManager()

        self.callback_manager.register_callback("exec", self.interpreter.run)
    def run(self):
        global_step = len(self.journal)
        while global_step < self.cfg.agent.steps:
            # Execute one step of the agent
            self.agent.step(callback_manager=self.callback_manager)
            # Save the current state
            save_run(self.cfg, self.journal)
            global_step = len(self.journal)
        # Cleanup the interpreter session
        self.interpreter.cleanup_session()
