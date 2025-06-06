"""configuration and setup utils"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Hashable, cast

import coolname

import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
import logging

from aide.journal import Journal, filter_journal

from . import tree_export
from . import preproc_data, serialize

shutup.mute_warnings()
logger = logging.getLogger("aide")


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float


@dataclass
class InitialSolutionConfig:
    exp_name: str | None
    node_id: str | None
    code_file: str | None


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int


@dataclass
class AgentConfig:
    steps: int
    time_limit: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool
    convert_system_to_user: bool
    obfuscate: bool

    copilot: StageConfig
    code: StageConfig
    feedback: StageConfig
    advisor: StageConfig

    search: SearchConfig


@dataclass
class DataConfig:
    """Configuration for data sources."""

    provider: str  # "huggingface", "kaggle", "local"
    dataset: str  # dataset name
    path: Path | None = None  # only for local provider
    dataset_kwargs: dict | None = None  # additional kwargs for dataset loading


@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool
    use_modal: bool | None = None
    gpu: str | None = None
    gpu_size: str | None = None
    gpu_count: int | None = None


@dataclass
class Config(Hashable):
    # Required fields first
    log_dir: Path
    log_level: str
    workspace_dir: Path
    debug: bool
    preprocess_data: bool
    copy_data: bool
    exp_name: str
    initial_solution: InitialSolutionConfig
    exec: ExecConfig
    generate_report: bool
    report: StageConfig
    agent: AgentConfig

    # Optional fields with defaults
    data_dir: Path | None = None
    desc_file: Path | None = None
    data: DataConfig | None = None
    goal: str | None = None
    eval: str | None = None
    task_id: str | None = None


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if current_index := int(p.name.split("-")[0]) > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent.parent.parent / "configs" / "config.yaml",
    use_cli_args=False,
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(
    path: Path = Path(__file__).parent.parent.parent / "configs" / "config.yaml",
    use_cli_args=False,
) -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path, use_cli_args=use_cli_args))


def prep_cfg(cfg: Config):
    if "--config-path" in cfg.keys():
        cfg.pop("--config-path")

    # Handle legacy data_dir path resolution
    if "data_dir" in cfg:
        if cfg.data_dir:
            if str(cfg.data_dir).startswith("example_tasks/"):
                cfg.data_dir = Path(__file__).parent.parent.parent / cfg.data_dir
            cfg.data_dir = Path(cfg.data_dir).resolve()

            # Handle legacy config format
            if "data" not in cfg or not cfg.data:
                cfg.data = DataConfig(
                    provider="local", dataset="legacy", path=cfg.data_dir  # placeholder
                )

    # Validation - need either new data config or legacy data_dir
    if not cfg.data and "data_dir" not in cfg:
        raise ValueError("Must specify either 'data_dir' or 'data' configuration.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    # Handle new data config path resolution for local provider
    if cfg.data and cfg.data.provider == "local" and cfg.data.path:
        if str(cfg.data.path).startswith("example_tasks/"):
            cfg.data.path = Path(__file__).parent.parent.parent / cfg.data.path
        cfg.data.path = Path(cfg.data.path).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name and prefix with consecutive index
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and prepare data."""
    from ..data_providers import create_data_provider

    # Setup local workspace directories
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)

    # Get data provider
    provider = create_data_provider(cfg)

    # Prepare data based on execution environment
    if cfg.exec.use_modal:
        logger.info("Modal runtime detected - preparing Modal volume...")
        assert cfg.task_id is not None, "Task ID must be provided for Modal runtime"

        # Modal execution: prepare data in Modal volume
        provider.prepare_modal_data(cfg.task_id, dataset_kwargs=cfg.data.dataset_kwargs)
    else:
        # Local execution only
        provider.prepare_local_data(
            cfg.workspace_dir / "input",
            use_symlinks=not cfg.copy_data,
            dataset_kwargs=cfg.data.dataset_kwargs,
        )

    # Preprocess data
    if cfg.preprocess_data:
        logger.info("Starting data preprocessing...")
        preproc_data(cfg.workspace_dir / "input")
        logger.info("Data preprocessing completed")

    logger.info("Agent workspace preparation completed!")


def save_run(cfg: Config, journal: Journal):
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    filtered_journal = filter_journal(journal)
    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    serialize.dump_json(filtered_journal, cfg.log_dir / "filtered_journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    # create the tree + code visualization
    # only if the journal has nodes
    if len(journal) > 0:
        tree_export.generate(cfg, journal, cfg.log_dir / "tree_plot.html")
    # save the best found solution
    best_node = journal.get_best_node()
    if best_node is not None:
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)
    # concatenate logs
    with open(cfg.log_dir / "full_log.txt", "w") as f:
        f.write(
            concat_logs(
                cfg.log_dir / "aide.log",
                cfg.workspace_dir / "best_solution" / "node_id.txt",
                cfg.log_dir / "filtered_journal.json",
            )
        )


def concat_logs(chrono_log: Path, best_node: Path, journal: Path):
    content = (
        "The following is a concatenation of the log files produced.\n"
        "If a file is missing, it will be indicated.\n\n"
    )

    content += "---First, a chronological, high level log of the AIDE run---\n"
    content += output_file_or_placeholder(chrono_log) + "\n\n"

    content += "---Next, the ID of the best node from the run---\n"
    content += output_file_or_placeholder(best_node) + "\n\n"

    content += "---Finally, the full journal of the run---\n"
    content += output_file_or_placeholder(journal) + "\n\n"

    return content


def output_file_or_placeholder(file: Path):
    if file.exists():
        if file.suffix != ".json":
            return file.read_text()
        else:
            return json.dumps(json.loads(file.read_text()), indent=4)
    else:
        return "File not found."
