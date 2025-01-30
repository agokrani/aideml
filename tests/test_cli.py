import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from click.testing import CliRunner
import aide
from aide.cli import cli, start
from aide.utils.execution_result import ExecutionResult
import asyncio

# Configure pytest-asyncio to use function scope for event loops
pytestmark = pytest.mark.asyncio(scope="function")

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.exp_name = "test_exp"
    config.log_level = "info"
    config.log_dir = Path("/tmp/test_logs")
    config.workspace_dir = Path("/tmp/workspace")
    config.agent = MagicMock()
    config.agent.steps = 5
    config.goal = "maximize"
    config.eval = "accuracy"
    config.initial_solution = MagicMock()
    config.initial_solution.exp_name = None
    config.initial_solution.node_id = None
    config.initial_solution.code_file = None
    config.exec = MagicMock()
    config.exec.use_modal = False
    config.task_id = "test_task"
    config.preprocess_data = False
    return config


async def test_cli_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert aide.__version__ in result.output


async def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert (
        "AIDE Agent: The CLI tool for autopilot and copilot ML workflows."
        in result.output
    )


async def test_start_no_config(runner):
    result = runner.invoke(start, ["autopilot"])
    assert result.exit_code == 0
    assert "Please provide a valid config path" in result.output


async def test_invalid_mode(runner, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy config content")
    result = runner.invoke(start, ["invalid_mode", "-c", str(config_path)])
    assert result.exit_code != 0

    assert "Invalid value for '[[autopilot|copilot]]'" in result.output


@patch("aide.cli.load_cfg")
async def test_logging_setup(mock_load_cfg, runner, mock_config, tmp_path):
    mock_config.log_dir = tmp_path
    mock_load_cfg.return_value = mock_config

    with patch("logging.FileHandler") as mock_handler:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy config content")
        result = runner.invoke(start, ["autopilot", "-c", str(config_path)])
        
        # Verify two file handlers were created
        assert mock_handler.call_count == 2, "Expected two file handlers to be created"
        
        # Verify the calls were made with correct file paths
        calls = mock_handler.call_args_list
        assert str(tmp_path / "aide.log") in str(calls[0]), "First handler should be for aide.log"
        assert str(tmp_path / "aide.verbose.log") in str(calls[1]), "Second handler should be for aide.verbose.log"
        
        # Verify the first handler has the VerboseFilter
        mock_handler.return_value.addFilter.assert_called_once()


@patch("aide.cli.load_cfg")
@patch("aide.cli.load_task_desc")
@patch("aide.cli.prep_agent_workspace")
@patch("aide.cli.Agent")
@patch("aide.cli.get_runtime")
@patch("aide.cli.Journal")
@patch("aide.cli.AutoPilot")
@patch("pathlib.Path.mkdir")
@patch("logging.FileHandler")
async def test_autopilot_mode(
    mock_file_handler,
    mock_mkdir,
    mock_autopilot,
    mock_journal,
    mock_get_runtime,
    mock_agent,
    mock_prep_workspace,
    mock_load_task_desc,
    mock_load_cfg,
    runner,
    mock_config,
    tmp_path,
):
    mock_config.log_dir = tmp_path / "logs"
    mock_config.workspace_dir = tmp_path / "workspace"
    mock_load_cfg.return_value = mock_config
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy config content")
    
    mock_autopilot_instance = AsyncMock()
    mock_autopilot.return_value = mock_autopilot_instance
    mock_autopilot_instance.run = AsyncMock()
    mock_mkdir.return_value = None
    mock_file_handler.return_value = MagicMock()
    
    mock_load_task_desc.return_value = "Test task description"
    
    with patch('asyncio.run') as mock_asyncio_run:
        result = runner.invoke(start, ["autopilot", "-c", str(config_path)])
        print(result.output)
        assert result.exit_code == 0
        mock_load_cfg.assert_called_once_with(str(config_path))
        mock_prep_workspace.assert_called_once()
        mock_agent.assert_called_once()
        mock_get_runtime.assert_called_once()
        mock_autopilot.assert_called_once()
        mock_asyncio_run.assert_called_once()
        mock_mkdir.assert_called()
        assert mock_file_handler.call_count == 2


@patch("aide.cli.load_cfg")
@patch("aide.cli.load_task_desc")
@patch("aide.cli.prep_agent_workspace")
@patch("aide.cli.Agent")
@patch("aide.cli.get_runtime")
@patch("aide.cli.Journal")
@patch("aide.cli.CoPilot")
@patch("pathlib.Path.mkdir")
@patch("logging.FileHandler")
async def test_copilot_mode(
    mock_file_handler,
    mock_mkdir,
    mock_copilot,
    mock_journal,
    mock_get_runtime,
    mock_agent,
    mock_prep_workspace,
    mock_load_task_desc,
    mock_load_cfg,
    runner,
    mock_config,
    tmp_path,
):
    mock_config.log_dir = tmp_path / "logs"
    mock_config.workspace_dir = tmp_path / "workspace"
    mock_load_cfg.return_value = mock_config
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy config content")
    
    # Setup mocks
    mock_copilot_instance = AsyncMock()
    mock_copilot.return_value = mock_copilot_instance
    mock_copilot_instance.run = AsyncMock()
    mock_mkdir.return_value = None
    mock_file_handler.return_value = MagicMock()
    
    # Mock task description
    mock_load_task_desc.return_value = "Test task description"
    
    def mock_run(coro):
        return None
        
    with patch('asyncio.run', side_effect=mock_run):
        result = runner.invoke(start, ["copilot", "-c", str(config_path)])
        
        assert result.exit_code == 0
        mock_load_cfg.assert_called_once_with(str(config_path))
        mock_prep_workspace.assert_called_once()
        mock_agent.assert_called_once()
        mock_get_runtime.assert_called_once()
        mock_copilot.assert_called_once()
        mock_mkdir.assert_called()
        assert mock_file_handler.call_count == 2


@patch("aide.cli.load_cfg")
@patch("aide.cli.load_task_desc")
@patch("aide.cli.prep_agent_workspace")
@patch("pathlib.Path.mkdir")
@patch("logging.FileHandler")
async def test_initial_solution_with_exp_name(
    mock_file_handler,
    mock_mkdir,
    mock_prep_workspace,
    mock_load_task_desc,
    mock_load_cfg,
    runner,
    mock_config,
    tmp_path,
):
    # Setup mock config with initial solution
    mock_config.initial_solution.exp_name = "previous_exp"
    mock_config.initial_solution.node_id = "node123"
    mock_config.initial_solution.code_file = None
    mock_load_cfg.return_value = mock_config
    mock_mkdir.return_value = None
    mock_file_handler.return_value = MagicMock()
    
    # Mock task description
    mock_load_task_desc.return_value = "Test task description"
    
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy config content")
    
    with patch("aide.cli.load_json") as mock_load_json:
        mock_journal = MagicMock()
        mock_node = MagicMock()
        # Setup metric value for comparison
        mock_node.metric = MagicMock()
        mock_node.metric.value = 0.5
        mock_journal.get.return_value = mock_node
        mock_load_json.return_value = mock_journal
        
        with patch('asyncio.run') as mock_asyncio_run:
            result = runner.invoke(start, ["autopilot", "-c", str(config_path)])
            
            assert result.exit_code == 0
            mock_load_json.assert_called_once()
            mock_journal.get.assert_called_once_with("node123")
            mock_mkdir.assert_called()
            assert mock_file_handler.call_count == 2


@patch("aide.cli.load_cfg")
@patch("aide.cli.load_task_desc")
@patch("aide.cli.prep_agent_workspace")
@patch("pathlib.Path.mkdir")
@patch("logging.FileHandler")
@patch("aide.cli.CallbackManager")
@patch("aide.cli.Agent")
async def test_initial_solution_with_code_file(
    mock_agent_class,
    mock_callback_manager,
    mock_file_handler,
    mock_mkdir,
    mock_prep_workspace,
    mock_load_task_desc,
    mock_load_cfg,
    runner,
    mock_config,
    tmp_path,
):
    # Setup mock config with initial solution code file
    mock_config.initial_solution.exp_name = None
    mock_config.initial_solution.node_id = None
    mock_config.initial_solution.code_file = "initial.py"
    mock_config.exec = MagicMock()
    mock_config.exec.use_modal = False
    mock_load_cfg.return_value = mock_config
    mock_mkdir.return_value = None
    mock_file_handler.return_value = MagicMock()
    
    # Mock task description
    mock_load_task_desc.return_value = "Test task description"
    
    # Mock callback manager
    mock_callback_instance = MagicMock()
    mock_callback_manager.return_value = mock_callback_instance
    mock_callback_instance.execute_callback = AsyncMock()
    mock_callback_instance.execute_callback.return_value = {"success": True}
    
    # Mock Agent
    mock_agent_instance = MagicMock()
    mock_agent_class.return_value = mock_agent_instance
    mock_agent_instance.parse_exec_result = AsyncMock()
    mock_agent_instance.parse_exec_result.return_value = MagicMock(is_buggy=False, metric=0.5)
    
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy config content")
    
    with patch("aide.cli.load_code_file") as mock_load_code_file:
        mock_node = MagicMock()
        mock_node.code = "test code"
        mock_node.is_buggy = False
        mock_node.metric = 0.5
        mock_load_code_file.return_value = mock_node
        
        # Mock the interpreter with proper ExecutionResult
        mock_interpreter = AsyncMock()
        exec_result = ExecutionResult(
            term_out="Success output",
            exec_time=1.0,
            exc_type=None,
            exc_info={},
            exc_stack=[]
        )
        mock_interpreter.run.return_value = exec_result
        mock_interpreter.install_missing_libraries = AsyncMock()
        
        with patch("aide.cli.get_runtime", return_value=mock_interpreter):
            # Mock both get_event_loop and run
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_loop.run_until_complete = MagicMock(return_value=exec_result)
                mock_get_loop.return_value = mock_loop
                
                with patch('asyncio.run') as mock_asyncio_run:
                    result = runner.invoke(start, ["autopilot", "-c", str(config_path)])
                    
                    assert result.exit_code == 0
                    mock_load_code_file.assert_called_once_with("initial.py")
                    mock_mkdir.assert_called()
                    assert mock_file_handler.call_count == 2

async def test_missing_config_path(runner):
    result = runner.invoke(start, ["autopilot"])
    assert result.exit_code == 0
    assert "Please provide a valid config path" in result.output


@patch("aide.cli.load_cfg")
@patch("aide.cli.load_task_desc")
@patch("aide.cli.prep_agent_workspace")
@patch("aide.cli.AutoPilot")
@patch("aide.cli.VerboseFilter")
async def test_logging_setup_with_verbose_filter(
    mock_filter,
    mock_autopilot,
    mock_prep_workspace,
    mock_load_task_desc,
    mock_load_cfg,
    runner,
    mock_config,
    tmp_path,
):
    # Setup mock config
    mock_config.log_dir = tmp_path
    mock_config.agent.steps = 5  # Ensure this is an integer, not a MagicMock
    mock_load_cfg.return_value = mock_config
    mock_load_task_desc.return_value = "Test task description"
    
    # Mock AutoPilot
    mock_autopilot_instance = AsyncMock()
    mock_autopilot.return_value = mock_autopilot_instance
    mock_autopilot_instance.run = AsyncMock()
    
    # Mock VerboseFilter
    mock_filter_instance = MagicMock()
    mock_filter.return_value = mock_filter_instance
    
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy config content")
    
    with patch("logging.FileHandler") as mock_handler:
        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance
        
        async def mock_run(coro):
            if asyncio.iscoroutine(coro):
                return await coro
            return coro
            
        with patch('asyncio.run', side_effect=mock_run):
            result = runner.invoke(start, ["autopilot", "-c", str(config_path)])
            
            assert result.exit_code == 0
            # Verify handlers were created with correct paths
            assert mock_handler.call_count == 2
            calls = mock_handler.call_args_list
            assert str(tmp_path / "aide.log") in str(calls[0])
            assert str(tmp_path / "aide.verbose.log") in str(calls[1])
            
            # Verify VerboseFilter was added to the first handler
            mock_handler_instance.addFilter.assert_called_once_with(mock_filter_instance)


async def test_invalid_config_path(runner):
    result = runner.invoke(start, ["autopilot", "-c", "nonexistent.yaml"])
    assert result.exit_code != 0
    assert "Invalid value for '--config-path' / '-c': Path 'nonexistent.yaml' does not exist." in result.output
