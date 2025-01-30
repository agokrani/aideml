import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import logging

from aide.callbacks.manager import CallbackManager
from aide.callbacks.stdout import read_input, handle_exit, execute_code


@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def callback_manager(mock_logger):
    return CallbackManager()


def test_callback_manager_initialization(callback_manager):
    assert callback_manager.callbacks == {}


def test_callback_manager_add_callback(callback_manager):
    callback_manager.register_callback("user_input", read_input)
    assert len(callback_manager.callbacks) == 1
    assert callback_manager.callbacks["user_input"] == read_input    

@pytest.mark.asyncio
async def test_callback_manager_execute_callback(callback_manager):
    mock_callback = MagicMock()
    mock_callback.on_stage_start = AsyncMock()
    
    callback_manager.register_callback("stage_start", mock_callback.on_stage_start)
    await callback_manager.execute_callback("stage_start", "Draft")
    mock_callback.on_stage_start.assert_called_once_with("Draft")


@pytest.mark.asyncio
async def test_callback_manager_execute_callback_with_error(callback_manager, mock_logger):
    mock_callback = MagicMock()
    mock_callback.on_stage_start = AsyncMock(side_effect=Exception("Test error"))
    
    callback_manager.register_callback("stage_start", mock_callback.on_stage_start)
    
    with pytest.raises(Exception) as exc_info:
        await callback_manager.execute_callback("stage_start", "Draft")
    
    assert str(exc_info.value) == "Test error"


@pytest.mark.asyncio
async def test_handle_exit():
    mock_interpreter = AsyncMock()
    # Test normal message
    await handle_exit("normal message", mock_interpreter)
    mock_interpreter.cleanup_session.assert_not_called()
    
    # Test exit message
    with pytest.raises(SystemExit):
        await handle_exit("/exit", mock_interpreter)
    mock_interpreter.cleanup_session.assert_called_once()


@patch('builtins.input')
def test_read_input(mock_input):
    # Test single line input
    mock_input.side_effect = ["line1", ""]
    result = read_input()
    assert result == "line1"
    
    # Test multi-line input
    mock_input.side_effect = ["line1", "line2", "line3", ""]
    result = read_input()
    assert result == "line1\nline2\nline3"


@pytest.mark.asyncio
@patch('aide.callbacks.stdout.Spinner')
@patch('aide.callbacks.stdout.Live')
async def test_execute_code_async(mock_live, mock_spinner):
    # Setup mock spinner
    spinner_instance = MagicMock()
    mock_spinner.return_value = spinner_instance
    
    # Setup mock live context
    mock_live_context = MagicMock()
    mock_live.return_value.__enter__.return_value = mock_live_context
    
    # Setup mock async interpreter
    mock_interpreter = MagicMock()
    mock_interpreter.run = AsyncMock(return_value="execution result")
    
    # Create and test the callback
    callback = execute_code(mock_interpreter)
    result = await callback("test code")
    
    # Verify spinner was created with correct message
    mock_spinner.assert_called_once_with("dots", text="[magenta]Executing code...[/magenta]")
    
    # Verify Live was created with spinner and refresh rate
    mock_live.assert_called_once_with(spinner_instance, refresh_per_second=4)
    
    # Verify the execution
    assert result == "execution result"
    mock_interpreter.run.assert_called_once_with("test code")
    mock_live_context.update.assert_called_once_with("[bold red]Done Executing the code[/bold red]")


@pytest.mark.asyncio
@patch('aide.callbacks.stdout.Spinner')
@patch('aide.callbacks.stdout.Live')
async def test_execute_code_sync(mock_live, mock_spinner):
    # Setup mock spinner
    spinner_instance = MagicMock()
    mock_spinner.return_value = spinner_instance
    
    # Setup mock live context
    mock_live_context = MagicMock()
    mock_live.return_value.__enter__.return_value = mock_live_context
    
    # Setup mock sync interpreter
    mock_interpreter = MagicMock()
    mock_interpreter.run = MagicMock(return_value="execution result")
    
    # Create and test the callback
    callback = execute_code(mock_interpreter)
    result = await callback("test code")
    
    # Verify spinner was created with correct message
    mock_spinner.assert_called_once_with("dots", text="[magenta]Executing code...[/magenta]")
    
    # Verify Live was created with spinner and refresh rate
    mock_live.assert_called_once_with(spinner_instance, refresh_per_second=4)
    
    # Verify the execution
    assert result == "execution result"
    mock_interpreter.run.assert_called_once_with("test code")
    mock_live_context.update.assert_called_once_with("[bold red]Done Executing the code[/bold red]")


