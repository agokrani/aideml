from aide.callbacks.manager import CallbackManager
from aide.workflow.base import Workflow
from aide.agent import Agent
from aide.interpreter import Interpreter
from aide.utils.config import Config
from aide.utils.config import save_run


class AutoPilot(Workflow):
    def __init__(
        self,
        cfg: Config,
        callback_manager: CallbackManager | None = None,
    ):
        super().__init__(cfg, callback_manager)
        

    async def run(self):
        global_step = len(self.journal)
        while global_step < self.cfg.agent.steps:
            # Execute one step of the agent
            await self.agent.step(callback_manager=self.callback_manager)
            # Save the current state
            save_run(self.cfg, self.journal)
            global_step = len(self.journal)
            await self.callback_manager.execute_callback("tool_output")
        # Cleanup the interpreter session
        self.interpreter.cleanup_session()
