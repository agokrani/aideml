import sys
import signal
import shutil
import asyncio
from omegaconf import Node
from pydantic import BaseModel
from aide.actions.action import Debug, Draft, Finish, Improve
from aide.callbacks.manager import CallbackManager
from aide.utils.metric import WorstMetricValue
from aide.workflow.base import Workflow
from aide.agent import Agent, ActionAgent
from aide.interpreter import Interpreter
from aide.utils.config import Config, save_run
from aide.runtime.runtime import Runtime
from aide.backend import provider_to_tool_response_message_func
from aide.journal import cache_best_node

import logging

logger = logging.getLogger("aide")


class CoPilot(Workflow):
    def __init__(
        self,
        agent: Agent,
        interpreter: Runtime,
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


    #     # Register signal handlers
    #     signal.signal(signal.SIGINT, self._signal_handler) 
    #     signal.signal(signal.SIGTERM, self._signal_handler)

    # async def _signal_handler(self, signum, frame):
    #     """Handle interrupt signals gracefully"""
        
    #     logger.info(f"Received signal {signum}, initiating cleanup...")
    #     try:
    #         # Get or create event loop
    #         try:
    #             loop = asyncio.get_running_loop()
    #         except RuntimeError:
    #             loop = asyncio.new_event_loop()
    #             asyncio.set_event_loop(loop)

    #         # Run cleanup
    #         asyncio.create_task(loop.run_until_complete(self.interpreter.cleanup_session()))
                
    #         logger.info("Cleanup completed successfully")
            
    #     except Exception as e:
    #         logger.error(f"Error during cleanup: {e}")
    #     finally:
    #         sys.exit(1)

    async def run(self):
        action_agent = ActionAgent(self.agent.task_desc, self.cfg)

        messages = []
        current_node = None
        if len(self.journal) == 0:
            await self.callback_manager.execute_callback(
                "stage_start",
                "initial_research",
                message="Performing ",
                supress_errors=True,
            )
            messages.append(
                {"role": "assistant", "content": self.agent.get_initial_research()}
            )

            await self.callback_manager.execute_callback(
                "stage_end", supress_errors=True
            )

            await self.callback_manager.execute_callback(
                "tool_output", self.agent.get_initial_research()
            )

            message = "Would you like to draft a new solution based on this initial research? (y/n)"
            messages.append({"role": "assistant", "content": message})
            await self.callback_manager.execute_callback("tool_output", message)

        else:
            best_node = self.journal.get_best_node()
            if best_node is not None:
                best_solution_message = "Best solution found so far:\n\n"
                best_solution_message += f"{best_node.generate_summary()}"
                messages.append({"role": "assistant", "content": best_solution_message})
                await self.callback_manager.execute_callback(
                    "tool_output", best_solution_message
                )

                message = "Would you like to improve this solution? or draft a new one?"
                messages.append({"role": "assistant", "content": message})
                await self.callback_manager.execute_callback("tool_output", message)
                current_node = best_node
            else:
                assert (
                    len(self.journal) == 1
                ), "Please specify node_id of the initial solution in the config"
                message = "Solution found so far:\n\n"
                message += f"{self.journal[0].generate_summary()}"
                message += "Would you like to debug/improve the current soluton or would you like to draft a new one?"
                messages.append({"role": "assistant", "content": message})
                await self.callback_manager.execute_callback("tool_output", message)
                current_node = self.journal[0]

        while True:
            if self.cfg.agent.data_preview and self.agent.data_preview is None:
                logger.info("Updating data preview...")
                await self.callback_manager.execute_callback(
                    "stage_start",
                    "data_preview",
                    message="Updating ",
                    supress_errors=True,
                )
                self.agent.update_data_preview()
                await self.callback_manager.execute_callback(
                    "stage_end", supress_errors=True
                )

            user_feedback = await self.callback_manager.execute_callback("user_input")

            exit_handler = self.callback_manager.callbacks.get("exit")
            if exit_handler is not None:
                await self.callback_manager.execute_callback("exit", user_feedback, self.interpreter)

            await self.callback_manager.execute_callback(
                "stage_start", "next action", message="Predicting ", supress_errors=True
            )

            messages.append({"role": "user", "content": user_feedback})

            next_action = await action_agent.predict_next_action(user_messages=messages)
            await self.callback_manager.execute_callback(
                "stage_end", supress_errors=True
            )

            if isinstance(next_action, Finish):
                await self.callback_manager.execute_callback(
                    "tool_output",
                    """It appears that you are satisfied with the current solution or wish to conclude the process. 
                    To exit, please use the /exit command. Alternatively, you may continue to refine the solution.""",
                )
                continue

            current_node = await self.step(next_action, parent_node=current_node)
            messages = []  # Clear messages after each step

            exec_result = await self.callback_manager.execute_callback(
                "exec", current_node.code
            )
            
            current_node = await self.agent.parse_exec_result(
                node=current_node,
                exec_result=exec_result,
                callback_manager=self.callback_manager,
            )

            message = "Results from the current solution:\n\n"
            message += f"{current_node.generate_summary()}"

            await self.callback_manager.execute_callback("tool_output", message)

            submission_exists = False
            if not current_node.is_buggy:
                if self.cfg.exec.use_modal:
                    submission_exists = await self.callback_manager.execute_callback("has_submission")
                else:
                    submission_exists = True

                if not submission_exists:
                    current_node.is_buggy = True
                    current_node.metric = WorstMetricValue()
                    logger.info(
                        f"Actually, node {current_node.id} did not produce a submission.csv"
                    )
                    await self.callback_manager.execute_callback(
                        "tool_output",
                        f"Actually, node {current_node.id} did not produce a submission.csv",
                    )

            tool_response_func = provider_to_tool_response_message_func[
                next_action.tool_call_metadata["provider"]
            ]

            messages.append(
                tool_response_func(
                    next_action.tool_call_metadata["tool_call_id"], message
                )
            )

            self.journal.append(current_node)

            best_node = self.journal.get_best_node()
            if best_node is not None:
                if best_node.id == current_node.id:
                    logger.info(f"Node {current_node.id} is the best node so far")
                    await self.callback_manager.execute_callback(
                        "tool_output", f"Node {current_node.id} is the best node so far"
                    )
                    cache_best_node(current_node, self.cfg.workspace_dir, self.cfg.exec.use_modal)
                else:
                    logger.info(f"Node {current_node.id} is not the best node")
                    logger.info(f"Node {best_node.id} is still the best node")

            save_run(self.cfg, self.journal)

    async def step(self, action: BaseModel, parent_node: Node | None = None):

        await self.callback_manager.execute_callback(
            "stage_start", action.__class__.__name__, supress_errors=True
        )

        if isinstance(action, Draft):
            result_node = self.agent._draft([action.user_feedback])
        elif isinstance(action, Improve):
            result_node = self.agent._improve(parent_node, [action.user_feedback])
        elif isinstance(action, Debug):
            result_node = self.agent._debug(parent_node, [action.user_feedback])

        await self.callback_manager.execute_callback("stage_end", supress_errors=True)

        await self.callback_manager.execute_callback(
            "tool_output", f"plan:\n{result_node.plan}\n\n"
        )
        await self.callback_manager.execute_callback(
            "tool_output", f"code:\n{result_node.code}\n\n"
        )

        return result_node
