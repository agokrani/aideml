import shutil
import logging
import random
import time
from typing import Any, Callable, List
import humanize
from aide.function import SearchArxiv, SearchPapersWithCode
from aide.actions import Debug, Draft, Improve, Finish, SubmitReview
from .backend import query
from .utils.execution_result import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code
from .utils.util import load_prompt
import traceback
from pathlib import Path

logger = logging.getLogger("aide")


def format_time(time_in_sec: int):
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"


ExecCallbackType = Callable[[str, bool], ExecutionResult]


class ActionAgent:
    def __init__(self, task_desc: str, cfg: Config):
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = self.cfg.agent
        self.message_history = []

    async def predict_next_action(self, user_messages):
        introduction = (
            "You are helpful copilot assisting a machine learning engineer. Your task is to look at the task description "
            "along with the chat history or summary of the previous messages and suggest the next action to take. "
            "The action should be only one of the following: draft, improve, debug, finish"
            "In case the user has not provided enough information, please choose the action that you think is most appropriate "
            "based on the task description and the chat history."
        )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            # "Chat history": self.message_history[:-10],
            # "Recent chat messages": self.message_history[-10:],
        }

        recent_messages = self.message_history[-10:]
        self.message_history.extend(user_messages)

        user_messages[:0] = recent_messages

        query_kwargs = {
            "system_message": prompt,
            "user_messages": user_messages,  # Pass the actual user messages
            "functions": [Debug, Draft, Improve, Finish],  # The action functions
            "model": self.acfg.copilot.model,  # Use copilot model
            "convert_system_to_user": self.acfg.convert_system_to_user,
            "temperature": self.acfg.copilot.temp
        }

        response = query(**query_kwargs)

        self.message_history.append(response.get_tool_message())

        return response


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.start_time = time.time()
        self.current_step = 0
        self._initial_research = None

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)
                logger.info(f"[search policy] debugging node {node_to_debug.id}")
                return node_to_debug

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.info("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.info(f"[search policy] greedy node selected: node {greedy_node.id}")
        return greedy_node

    def plan_and_code_query(
        self, prompt, user_messages=None, retries=3
    ) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        logger(f"Making API call with model: {self.acfg.code.model}")
        completion_text = None
        
        for _ in range(retries):
            query_kwargs = {
                "system_message": prompt,
                "user_messages": user_messages,
                "model": self.acfg.code.model,
                "convert_system_to_user": self.acfg.convert_system_to_user,
                "temperature": self.acfg.code.temp
            }
                        
            # Call the query function with the constructed kwargs
            raw_query_response_tuple = query(**query_kwargs)

            logger.info(f"Raw response tuple from plan/code LLM call: {raw_query_response_tuple}")
            completion_text = raw_query_response_tuple
            logger(f"API response received: {completion_text[:100]}...")

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            logger.info("Plan + code extraction failed, retrying...")
        logger.info("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def plan_query(self, stage: str, user_messages=None, retries=3, **variables) -> str:
        """Generate a natural language plan using external prompt configuration."""
        for attempt in range(retries):
            try:
                # Load prompt from external files
                prompt = load_prompt(stage, "planning", obfuscate=self.acfg.obfuscate, **variables)
                
                # If prompt loading failed (no files), return empty plan
                if not prompt:
                    logger.warning(f"No prompt configuration found for {stage}/planning, returning empty plan")
                    return ""
                
                query_kwargs = {
                    "system_message": prompt,
                    "user_messages": user_messages,
                    "model": self.acfg.code.model,
                    "convert_system_to_user": self.acfg.convert_system_to_user,
                    "temperature": self.acfg.code.temp
                }
                
                plan_text = query(**query_kwargs)
                
                # Validate that we got a reasonable plan
                if plan_text and len(plan_text.strip()) > 10:
                    return plan_text.strip()
                
                logger.info(f"Plan generation failed on attempt {attempt + 1}, retrying...")
                
            except Exception as e:
                logger.warning(f"Plan query failed on attempt {attempt + 1}: {e}")
                
        logger.info("Final plan generation attempt failed, returning empty plan")
        return ""

    def code_query(self, stage: str, plan: str, user_messages=None, retries=3, **variables) -> str:
        """Generate code implementing the provided plan using external prompt configuration."""
        for attempt in range(retries):
            try:
                # Add plan to variables
                variables["plan"] = plan
                
                # Load prompt from external files
                prompt = load_prompt(stage, "coding", obfuscate=self.acfg.obfuscate, **variables)
                
                # If prompt loading failed (no files), fall back to old behavior
                if not prompt:
                    logger.warning(f"No prompt configuration found for {stage}/coding, using fallback")
                    return ""

                query_kwargs = {
                    "system_message": prompt,
                    "user_messages": user_messages,
                    "model": self.acfg.code.model,
                    "convert_system_to_user": self.acfg.convert_system_to_user,
                    "temperature": self.acfg.code.temp
                }
                
                completion_text = query(**query_kwargs)
                
                # Extract code from response
                code = extract_code(completion_text)
                
                if code:
                    return code
                
                logger.info(f"Code extraction failed on attempt {attempt + 1}, retrying...")
                
            except Exception as e:
                logger.warning(f"Code query failed on attempt {attempt + 1}: {e}")
                
        logger.info("Final code generation attempt failed, returning completion text")
        return completion_text if 'completion_text' in locals() else ""

    def _draft(self, user_messages: List | None = None) -> Node:
        # Prepare variables for external prompts
        variables = {
            "task_desc": self.task_desc,
            "memory": self.journal.generate_summary(),
            "data_preview": self.data_preview if self.acfg.data_preview else "",
            "time_remaining": format_time(self.acfg.time_limit - (time.time() - self.start_time)),
            "steps_remaining": str(self.acfg.steps - self.current_step),
            "exec_timeout": humanize.naturaldelta(int(min(self.cfg.exec.timeout, self.acfg.time_limit - (time.time() - self.start_time)))),
        }
        
        # Try new two-phase approach first
        plan = self.plan_query("draft", user_messages, **variables)
        assert plan is not None, "Draft plan should not be None"
        
        code = self.code_query("draft", plan, user_messages, **variables)
        assert code is not None, "Draft code should not be None"
        
        new_node = Node(plan=plan, code=code)
        logger.info(f"Drafted new node {new_node.id} using new prompt system")
        
        return new_node
        

    def _improve(self, parent_node: Node, user_messages: List | None = None) -> Node:
        # Prepare variables for external prompts
        variables = {
            "task_desc": self.task_desc,
            "memory": self.journal.generate_summary(),
            "previous_solution": wrap_code(parent_node.code),
            "time_remaining": format_time(self.acfg.time_limit - (time.time() - self.start_time)),
            "steps_remaining": str(self.acfg.steps - self.current_step),
            "exec_timeout": humanize.naturaldelta(int(min(self.cfg.exec.timeout, self.acfg.time_limit - (time.time() - self.start_time)))),
        }
        
        
        # Try new two-phase approach first
        plan = self.plan_query("improve", user_messages, **variables)
        assert plan is not None, "Improve plan should not be None"
        
        code = self.code_query("improve", plan, user_messages, **variables)
        assert code is not None, "Improve code should not be None"
        
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Improved node {parent_node.id} to create new node {new_node.id} using new prompt system")
        
        return new_node
        

    def _debug(self, parent_node: Node, user_messages: List | None = None) -> Node:
        # Prepare variables for external prompts
        variables = {
            "task_desc": self.task_desc,
            "buggy_code": wrap_code(parent_node.code),
            "execution_output": wrap_code(parent_node.term_out, lang=""),
            "data_preview": self.data_preview if self.acfg.data_preview else "",
            "time_remaining": format_time(self.acfg.time_limit - (time.time() - self.start_time)),
            "steps_remaining": str(self.acfg.steps - self.current_step),
            "exec_timeout": humanize.naturaldelta(int(min(self.cfg.exec.timeout, self.acfg.time_limit - (time.time() - self.start_time)))),
        }
        
        # Try new two-phase approach first
        plan = self.plan_query("debug", user_messages, **variables)
        assert plan is not None, "Debug plan should not be None"
        
        code = self.code_query("debug", plan, user_messages, **variables)
        assert code is not None, "Debug code should not be None"

        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Debugged node {parent_node.id} to create new node {new_node.id} using new prompt system")
        
        return new_node

    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    # For backward compatibility, need to change once the pipeline is verified
    async def step(self, exec_callback: ExecCallbackType = None, callback_manager=None):
        # clear the submission dir from previous steps

        print(f"Starting agent step, current journal size: {len(self.journal.nodes)}")

        if not self.cfg.exec.use_modal:
            shutil.rmtree(self.cfg.workspace_dir / "submission", ignore_errors=True)
            (self.cfg.workspace_dir / "submission").mkdir(exist_ok=True)
        else:
            await callback_manager.execute_callback("remove_submission_directory")

        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        print(f"Search policy selected parent node: {parent_node}")
        logger.info(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            print("Drafting new node")
            await callback_manager.execute_callback(
                "stage_start", "Draft", supress_errors=True
            )
            result_node = self._draft()
            await callback_manager.execute_callback(
                "stage_end", "Draft", supress_errors=True
            )
        elif parent_node.is_buggy:
            print(f"Debugging buggy node: {parent_node.id}")
            await callback_manager.execute_callback(
                "stage_start", "Debug", supress_errors=True
            )
            result_node = self._debug(parent_node)
            await callback_manager.execute_callback(
                "stage_end", "Debug", supress_errors=True
            )
        else:
            print(f"Improving node: {parent_node.id}")
            await callback_manager.execute_callback(
                "stage_start", "Improve", supress_errors=True
            )
            result_node = self._improve(parent_node)
            await callback_manager.execute_callback(
                "stage_end", "Improve", supress_errors=True
            )
        if exec_callback:
            print("Executing node code")
            exec_result = exec_callback(result_node.code)
            print(f"Execution complete, error: {exec_result.exc_type}")
        else:
            exec_result = await callback_manager.execute_callback(
                "exec", result_node.code
            )
        result_node = await self.parse_exec_result(
            node=result_node,
            exec_result=exec_result,
            exec_callback=exec_callback,
            callback_manager=callback_manager,
            use_modal=self.cfg.exec.use_modal,
        )
        # print(f"Node processed, is_buggy: {result_node.is_buggy}, has metric: {result_node.metric is not None}")
        # TODO: Fix this to check submission when using modal. Also verify the cache_best_node function
        # handle final cases where we missed buggy nodes somehow
        if not result_node.is_buggy:
            submission_exists = False
            if self.cfg.exec.use_modal:
                submission_exists = await callback_manager.execute_callback(
                    "has_submission"
                )
            else:
                submission_dir = self.cfg.workspace_dir / "submission"
                logger.info(f"DEBUG (step method): Checking submission directory: {submission_dir.resolve()}")
                if submission_dir.exists():
                    contents = list(submission_dir.iterdir())
                    logger.info(f"DEBUG (step method): Submission directory exists. Contents: {[p.name for p in contents]}")
                    if not contents:
                        logger.info("DEBUG (step method): Submission directory is EMPTY.")
                    if any(submission_dir.iterdir()):
                        submission_exists = True
                else:
                    submission_exists = False
                    logger.info("DEBUG (step method): Submission directory does NOT exist.")
            
            if not submission_exists:
                result_node.is_buggy = True
                result_node.metric = WorstMetricValue()
                logger.info(
                    f"Actually, node {result_node.id} did not produce a submission directory."
                )
        self.journal.append(result_node)
        print(f"Step complete, journal now has {len(self.journal.nodes)} nodes")

        # if the result_node is the best node, cache its submission.csv and solution.py
        # to best_solution/ by copying it there
        best_node = self.journal.get_best_node()
        if best_node is not None:
            if best_node.id == result_node.id:
                logger.info(f"Node {result_node.id} is the best node so far")
                await callback_manager.execute_callback("cache_best_node", result_node)
            else:
                logger.info(f"Node {result_node.id} is not the best node")
                logger.info(f"Node {best_node.id} is still the best node")
        self.current_step += 1

    async def parse_exec_result(
        self,
        node: Node,
        exec_result: ExecutionResult,
        attempts=0,
        max_attempts=3,
        exec_callback: ExecCallbackType = None,
        callback_manager=None,
        use_modal=False,
    ) -> Node:
        logger.info(f"Agent is parsing execution results for node {node.id}")
        node.absorb_exec_result(exec_result)

        logger.info(f"DEBUG (parse_exec_result): exec_result.term_out for node {node.id} (attempt {attempts}):\n{''.join(exec_result.term_out)}")

        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "You have written code to solve this task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            )
        prompt = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        response = query(
            system_message=prompt,
            user_messages=None,
            functions=SubmitReview,
            model=self.acfg.feedback.model,
            temperature=self.acfg.feedback.temp,
            convert_system_to_user=self.acfg.convert_system_to_user,
        )
        if not isinstance(response, SubmitReview):
            logger.error(f"Expected SubmitReview but got {type(response)}")
            return None
        if (
            response.missing_libraries is not None
            and len(response.missing_libraries) > 0
        ):
            if attempts < max_attempts:
                logger.info(
                    f"Agent is missing libraries, attempting to install them: {response.missing_libraries}"
                )
                # install missing libraries
                await callback_manager.execute_callback(
                    "install_dependencies", response.missing_libraries
                )

                # Re-execute the code after installing libraries
                if exec_callback:
                    exec_result = exec_callback(
                        node.code, True if not self.cfg.exec.use_modal else False
                    )
                else:
                    exec_result = await callback_manager.execute_callback(
                        "exec",
                        node.code,
                        True if not self.cfg.exec.use_modal else False,
                    )
                # Recursively parse the new execution result
                return await self.parse_exec_result(
                    node=node,
                    exec_result=exec_result,
                    exec_callback=exec_callback,
                    callback_manager=callback_manager,
                    attempts=attempts + 1,
                    max_attempts=max_attempts,
                    use_modal=use_modal,
                )
            else:
                logger.info(
                    "Maximum attempts reached while trying to install missing libraries"
                )

        # if the metric isn't a float then fill the metric with the worst metric
        if not isinstance(response.metric, float):
            response.metric = None
        # do an extra check, to catch cases where judge fails
        if use_modal:
            has_any_submission = await callback_manager.execute_callback(
                "has_submission"
            )
        else:
            submission_dir = self.cfg.workspace_dir / "submission"
            logger.info(f"DEBUG (parse_exec_result): Checking submission directory: {submission_dir.resolve()}")
            if submission_dir.exists():
                contents = list(submission_dir.iterdir())
                logger.info(f"DEBUG (parse_exec_result): Submission directory exists. Contents: {[p.name for p in contents]}")
                if not contents:
                    logger.info("DEBUG (parse_exec_result): Submission directory is EMPTY.")
            else:
                logger.info("DEBUG (parse_exec_result): Submission directory does NOT exist.")
            has_any_submission = submission_dir.exists() and any(submission_dir.iterdir())

        node.analysis = response.summary
        node.is_buggy = (
            response.is_bug
            or node.exc_type is not None
            or response.metric is None
            or not has_any_submission
        )

        if node.is_buggy:
            logger.info(
                f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv"
            )
            node.metric = WorstMetricValue()
        else:
            logger.info(f"Parsed results: Node {node.id} is not buggy")
            node.metric = MetricValue(
                response.metric, maximize=not response.lower_is_better
            )

        return node
