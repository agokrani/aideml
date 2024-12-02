import shutil
from omegaconf import Node
from pydantic import BaseModel
from aide.actions.action import Debug, Draft, Finish, Improve
from aide.callbacks.manager import CallbackManager
from aide.utils.metric import WorstMetricValue
from aide.workflow.base import Workflow
from aide.agent import Agent, ActionAgent
from aide.interpreter import Interpreter
from aide.utils.config import Config
from aide.backend import provider_to_tool_response_message_func

import logging

logger = logging.getLogger("aide")


class CoPilot(Workflow):
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

    async def run(self):

        action_agent = ActionAgent(self.agent.task_desc, self.cfg)

        messages = []
        current_node = None
        if len(self.journal) == 0:
            messages.append(
                {"role": "assistant", "content": self.agent.get_initial_research()}
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
                best_solution_message = f"Best solution found so far:\n\n"
                best_solution_message += f"{best_node.generate_summary()}"
                messages.append({"role": "assistant", "content": best_solution_message})
                await self.callback_manager.execute_callback(
                    "tool_output", best_solution_message
                )

                message = (
                    f"Would you like to improve this solution? or draft a new one?"
                )
                messages.append({"role": "assistant", "content": message})
                await self.callback_manager.execute_callback("tool_output", message)
                current_node = best_node
            else:
                assert (
                    len(self.journal) == 1
                ), "Please specify node_id of the initial solution in the config"
                message = f"Solution found so far:\n\n"
                message += f"{self.journal[0].generate_summary()}"
                message += "Would you like to debug/improve the current soluton or would you like to draft a new one?"
                messages.append({"role": "assistant", "content": message})
                await self.callback_manager.execute_callback("tool_output", message)
                current_node = self.journal[0]

        while True:
            if self.cfg.agent.data_preview and self.agent.data_preview is None:
                logger.info("Updating data preview...")
                self.agent.update_data_preview()

            user_feedback = await self.callback_manager.execute_callback("user_input")

            exit_handler = self.callback_manager.callbacks.get("exit")
            if exit_handler is not None:
                self.interpreter.cleanup_session()
                await self.callback_manager.execute_callback("exit", user_feedback)

            messages.append({"role": "user", "content": user_feedback})

            next_action = await action_agent.predict_next_action(user_messages=messages)

            if isinstance(next_action, Finish):
                await self.callback_manager.execute_callback(
                    "tool_output",
                    f"""It appears that you are satisfied with the current solution or wish to conclude the process. 
                    To exit, please use the /exit command. Alternatively, you may continue to refine the solution.""",
                )
                continue

            current_node = await self.step(next_action, parent_node=current_node)
            messages = []  # Clear messages after each step

            exec_result = await self.callback_manager.execute_callback(
                "exec", current_node.code, True
            )

            current_node = self.agent.parse_exec_result(
                node=current_node,
                exec_result=exec_result,
            )

            if not current_node.is_buggy:
                if not (
                    self.cfg.workspace_dir / "submission" / "submission.csv"
                ).exists():
                    current_node.is_buggy = True
                    current_node.metric = WorstMetricValue()
                    logger.info(
                        f"Actually, node {current_node.id} did not produce a submission.csv"
                    )
                    await self.callback_manager.execute_callback(
                        "tool_output",
                        f"Actually, node {current_node.id} did not produce a submission.csv",
                    )

            message = f"results of current solution from the {current_node.stage_name} step:\n\n"
            message += f"{current_node.generate_summary()}"

            tool_response_func = provider_to_tool_response_message_func[
                next_action.tool_call_metadata["provider"]
            ]

            messages.append(
                tool_response_func(
                    next_action.tool_call_metadata["tool_call_id"], message
                )
            )

            self.journal.append(current_node)

            # if the current_node is the best node, cache its submission.csv and solution.py
            # to best_solution/ by copying it there
            best_node = self.journal.get_best_node()
            if best_node is not None:
                if best_node.id == current_node.id:

                    logger.info(f"Node {current_node.id} is the best node so far")
                    await self.callback_manager.execute_callback(
                        "tool_output", f"Node {current_node.id} is the best node so far"
                    )

                    best_solution_dir = self.cfg.workspace_dir / "best_solution"
                    best_solution_dir.mkdir(exist_ok=True, parents=True)

                    # copy submission/submission.csv to best_submission/submission.csv
                    best_submission_dir = self.cfg.workspace_dir / "best_submission"
                    best_submission_dir.mkdir(exist_ok=True, parents=True)
                    shutil.copy(
                        self.cfg.workspace_dir / "submission" / "submission.csv",
                        best_submission_dir,
                    )
                    # copy solution.py and relevant node id to best_solution/
                    with open(best_solution_dir / "solution.py", "w") as f:
                        f.write(current_node.code)
                    # take note of the node id of the best node
                    with open(best_solution_dir / "node_id.txt", "w") as f:
                        f.write(str(current_node.id))
                else:
                    logger.info(f"Node {current_node.id} is not the best node")
                    logger.info(f"Node {best_node.id} is still the best node")

    async def step(self, action: BaseModel, parent_node: Node | None = None):
        if isinstance(action, Draft):
            result_node = self.agent._draft([action.user_feedback])
        elif isinstance(action, Improve):
            result_node = self.agent._improve(parent_node, [action.user_feedback])
        elif isinstance(action, Debug):
            self.agent._debug(parent_node, [action.user_feedback])

        await self.callback_manager.execute_callback(
            "tool_output", f"plan:\n{result_node.plan}\n\n"
        )
        await self.callback_manager.execute_callback(
            "tool_output", f"code:\n{result_node.code}\n\n"
        )

        return result_node
