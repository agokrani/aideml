import sys
import asyncio
import logging
import click
import aide
from omegaconf import OmegaConf
from rich.console import Console
from aide import backend
from aide.agent import Agent
from aide.callbacks.manager import CallbackManager
from aide.interpreter import Interpreter
from aide.journal import Journal
from aide.utils.config import load_cfg, load_task_desc, prep_agent_workspace
from aide.utils.serialize import load_code_file, load_json
from aide.workflow.copilot import CoPilot
from aide.callbacks.stdout import handle_exit, read_input, execute_code

console = Console()


@click.group()
@click.version_option(version=aide.__version__)
def cli():
    """AIDE Agent: The CLI tool for autopilot and copilot ML workflows."""
    pass


@cli.command()
@click.argument(
    "mode", default="autopilot", type=click.Choice(["autopilot", "copilot"])
)
@click.option(
    "--config-path", "-c", type=click.Path(exists=True), help="Path to config file."
)
def start(mode, config_path=None):
    """Start an autopilot or copilot run."""
    if config_path is None:
        console.print("Please provide a valid config path using --config-path option.")
        return

    cfg = load_cfg(config_path)

    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()), format=log_format, handlers=[]
    )

    logger = logging.getLogger("aide")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))

    logger.addHandler(console_handler)

    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    prep_agent_workspace(cfg)

    journal = Journal()
    logger.info(f'Starting run "{cfg.exp_name}"')

    agent = Agent(
        task_desc=task_desc_str,
        cfg=cfg,
        journal=journal,
    )

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec)  # type: ignore
    )

    if cfg.initial_solution.exp_name is not None:
        journal_json = (
            cfg.log_dir.parent / cfg.initial_solution.exp_name / "journal.json"
        ).resolve()
        prev_journal = load_json(journal_json, Journal)
        if cfg.initial_solution.node_id is not None:
            node = prev_journal.get(cfg.initial_solution.node_id)
        else:
            node = prev_journal.get_best_node()
        if node is not None:
            agent.journal.append(node)
    elif cfg.initial_solution.code_file_path is not None:
        assert (
            cfg.initial_solution.node_id is None
            and cfg.initial_solution.exp_name is None
        ), f"Please specify either code_file_path or a combination of exp_name and node_id. Specifying both is not allowed."
        node = load_code_file(cfg.initial_solution.code_file_path)
        if node:
            # TODO: Remove this from here once the proper place to set load this file has been identified
            exec_result = interpreter.run(node=node)
            agent.parse_exec_result(node=node, exec_result=exec_result)
            agent.journal.append(node)

    if mode == "autopilot":
        console.print("Starting autopilot run...\n")
        # aide.run.run()
    elif mode == "copilot":
        console.print("Starting copilot run...\n")

        callback_manager = CallbackManager()

        callback_manager.register_callbacks(
            {
                "tool_output": console.print,
                "user_input": read_input,
                "exec": execute_code(interpreter),
                "exit": handle_exit,
            }
        )

        copilot = CoPilot(agent, interpreter, cfg, callback_manager)
        asyncio.run(copilot.run())


if __name__ == "__main__":
    cli()
