import asyncio
import logging
import click
import aide
from rich.live import Live
from rich.status import Status
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.spinner import Spinner
from rich.text import Text
from rich.padding import Padding
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from aide.callbacks.manager import CallbackManager
from aide.run import VerboseFilter, journal_to_rich_tree
from aide.utils.config import load_cfg
from aide.workflow.autopilot import AutoPilot
from aide.workflow.copilot import CoPilot
from aide.callbacks.stdout import handle_exit, read_input, execute_code

console = Console()
live_display = None


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

    # save logs to files as well, using same format
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # we'll have a normal log file and verbose log file. Only normal to console
    file_handler = logging.FileHandler(cfg.log_dir / "aide.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(VerboseFilter())

    verbose_file_handler = logging.FileHandler(cfg.log_dir / "aide.verbose.log")
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    # This should only be enabled when debugging
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(logging.Formatter(log_format))

    # logger.addHandler(console_handler)

    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)

    logger.info(f'Starting run "{cfg.exp_name}"')

    if mode == "autopilot":
        console.print("Starting autopilot run...\n")
        
        callback_manager = CallbackManager()
        autopilot = AutoPilot(cfg, callback_manager)

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        task_id = progress.add_task("Progress:", total=cfg.agent.steps)
        status = Status("[green]Setting up...")

        def generate_display():
            tree = journal_to_rich_tree(autopilot.journal)
            progress.update(task_id, completed=len(autopilot.journal))

            file_paths = [
                f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
                f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
                f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
            ]

            # Truncate the task description to a fixed number of lines
            task_desc_lines = autopilot.agent.task_desc.strip().split("\n")
            max_lines = 10  # Number of lines to display
            if len(task_desc_lines) > max_lines:
                task_desc_display = "\n".join(task_desc_lines[:max_lines])
                task_desc_display += "..."
            else:
                task_desc_display = autopilot.agent.task_desc.strip()

            left = Group(
                Panel(Text(task_desc_display), title="Task description"),
                progress,
                status,
            )

            right = tree
            wide = Group(*file_paths)

            return Panel(
                Group(
                    Padding(wide, (1, 1, 1, 1)),
                    Columns(
                        [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                        equal=True,
                    ),
                ),
                title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
                subtitle="Press [b]Ctrl+C[/b] to stop the run",
            )

        def exec_callback(*args, **kwargs):
            status.update("[magenta]Executing code...")
            res = autopilot.interpreter.run(*args, **kwargs)
            return res

        def stage_start(stage_name, message=None):
            if message is None:
                status.update(
                    f"[green]Generating plan and code for {stage_name}...[/green]"
                )
            else:
                status.update(f"[green]{message}{stage_name}...[/green]")

        
        autopilot.callback_manager.register_callbacks(
            {"exec": exec_callback, "stage_start": stage_start}
        )

        with Live(generate_display(), refresh_per_second=16, screen=True) as live:

            def update_display(*args, **kwargs):
                live.update(generate_display())

            autopilot.callback_manager.register_callback("tool_output", update_display)
            asyncio.run(autopilot.run())

    elif mode == "copilot":
        console.print("Starting copilot run...\n")
        callback_manager = CallbackManager()

        def stage_start(stage_name, message=None):
            global live_display
            if live_display:
                live_display.stop()
            if message is None:
                final_message = f"Generating plan and code for {stage_name}..."
            else:
                final_message = f"{message}{stage_name}..."
            spinner = Spinner("dots", text=f"[green]{final_message}[/green]")
            live_display = Live(spinner, refresh_per_second=4)
            live_display.start()

        def stage_end():
            global live_display
            live_display.stop()
            live_display = None

        callback_manager.register_callbacks(
            {
                "tool_output": console.print,
                "user_input": read_input,
                "exit": handle_exit,
                "stage_start": stage_start,
                "stage_end": stage_end,
            }
        )
        copilot = CoPilot(cfg, callback_manager)
        
        # HACK: This is a temporary fix to get the copilot interpreter callback
        copilot.callback_manager.register_callback("exec", execute_code(copilot.interpreter))
        
        asyncio.run(copilot.run())


if __name__ == "__main__":
    cli()
