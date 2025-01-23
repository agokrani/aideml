from aide.agent import Agent
from aide.utils.config import Config
from aide.utils.config import save_run
from aide.workflow.base import Workflow
from aide.interpreter import Interpreter
from aide.runtime.modal import ModalRuntime
from aide.callbacks.manager import CallbackManager


class AutoPilot(Workflow):
    def __init__(
        self,
        agent: Agent,
        interpreter: Interpreter,
        cfg: Config,
        callback_manager: CallbackManager | None = None,
    ):
        self.agent = agent
        self.cfg = cfg
        self.interpreter = interpreter
        self.journal = agent.journal
        self.callback_manager = (
            callback_manager if callback_manager is not None else CallbackManager()
        )

        try:
            self.callback_manager.callbacks["exec"]
        except KeyError:
            self.callback_manager.register_callback("exec", self.interpreter.run)

        if self.cfg.exec.use_modal:
            assert isinstance(self.interpreter, ModalRuntime)
            try:
                self.callback_manager.callbacks["has_submission"]
            except KeyError:
                self.callback_manager.register_callback(
                    "has_submission", self.interpreter.has_submission
                )

            try:
                self.callback_manager.callbacks["remove_submission_directory"]
            except KeyError:
                callback_manager.register_callback(
                    "remove_submission_directory",
                    self.interpreter.remove_previous_submissions_directory,
                )

        # outside of If block on purpose because both local and modal runtimes need to install dependencies
        try:
            self.callback_manager.callbacks["install_dependencies"]
        except KeyError:
            callback_manager.register_callback(
                "install_dependecies", self.interpreter.install_missing_libraries
            )

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
        await self.interpreter.cleanup_session()
