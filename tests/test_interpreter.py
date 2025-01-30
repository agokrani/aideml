import pytest
import tempfile
from pathlib import Path
from aide.utils.execution_result import ExecutionResult
from aide.utils.response import (
    extract_code,
    extract_text_up_to_code,
    wrap_code,
)
from aide.interpreter import Interpreter


@pytest.fixture
def interpreter():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_dir = Path(tmpdir)
        interpreter = Interpreter(working_dir=workspace_dir, timeout=9)
        yield interpreter
        interpreter.cleanup_session()


def test_execution_result_initialization():
    result = ExecutionResult(term_out=["test output"], exec_time=1.0, exc_type=None)
    assert result.term_out == ["test output"]
    assert result.exec_time == 1.0
    assert result.exc_type is None


def test_execution_result_with_exception():
    result = ExecutionResult(
        term_out=["error output"],
        exec_time=1.0,
        exc_type="RuntimeError",
        exc_info={"args": ["Test error occurred"]},
        exc_stack=[("test.py", 1, "test_func", "x = 1/0")],
    )
    assert result.exc_type == "RuntimeError"
    assert result.exc_info["args"] == ["Test error occurred"]
    assert len(result.exc_stack) == 1


def test_extract_code_single_block():
    text = """Here's a simple example:
```python
def hello():
    print("Hello, World!")
```
That's all!"""
    code = extract_code(text)
    assert code.strip() == 'def hello():\n    print("Hello, World!")'


def test_extract_code_multiple_blocks():
    text = """First block:
```python
x = 1
```
Second block:
```python
y = 2
```"""
    code = extract_code(text)
    # extract_code combines all valid code blocks and formats them
    assert code.strip() == "x = 1\n\n\ny = 2"


def test_extract_code_no_blocks():
    text = "This text contains no code blocks"
    code = extract_code(text)
    assert code == ""


def test_extract_code_with_language_specifier():
    text = """Here's some code:
```python
x = 1
```
And some other code:
```javascript
let y = 2;
```"""
    code = extract_code(text)
    # Only Python code blocks are extracted and formatted
    assert code.strip() == "x = 1"


def test_extract_text_up_to_code():
    text = """Here's the plan:
1. Initialize variables
2. Process data

```python
x = 1
y = 2
```"""
    plan = extract_text_up_to_code(text)
    assert "Here's the plan:" in plan
    assert "1. Initialize variables" in plan
    assert "2. Process data" in plan
    assert "```python" not in plan
    assert "x = 1" not in plan


def test_extract_text_up_to_code_no_code():
    text = "This is just a plain text without any code blocks"
    result = extract_text_up_to_code(text)
    # When there's no code block, return empty string
    assert result == ""


def test_wrap_code():
    code = 'print("Hello, World!")'
    wrapped = wrap_code(code)
    assert wrapped.startswith("```")
    assert wrapped.endswith("```")
    assert code in wrapped


def test_wrap_code_with_language():
    code = 'print("Hello, World!")'
    wrapped = wrap_code(code, lang="python")
    assert wrapped.startswith("```python")
    assert wrapped.endswith("```")
    assert code in wrapped


def test_interpreter_run_success(interpreter):
    code = """
print("Hello, World!")
x = 1 + 1
print(f"x = {x}")
"""
    with tempfile.TemporaryDirectory():
        result = interpreter.run(code)

        assert result.exc_type is None
        # Check that both outputs are in the same string
        assert "Hello, World!" in result.term_out[0]
        print(result.term_out)
        assert "x = 2" in result.term_out


def test_interpreter_run_syntax_error(interpreter):
    code = """
print('Hello, World!'
x = 1 + 1  # Missing parenthesis above
"""
    with tempfile.TemporaryDirectory():
        result = interpreter.run(code)

        assert result.exc_type == "SyntaxError"
        assert any("SyntaxError" in line for line in result.term_out)


def test_interpreter_run_runtime_error(interpreter):
    code = """
x = 1 / 0  # Division by zero
"""
    result = interpreter.run(code)

    assert result.exc_type == "ZeroDivisionError"
    assert any("ZeroDivisionError" in line for line in result.term_out)


def test_interpreter_run_with_imports(interpreter):
    code = """
import math
x = math.pi
print(f'pi = {x:.2f}')
"""
    with tempfile.TemporaryDirectory():
        result = interpreter.run(code)

        assert result.exc_type is None
        assert any("pi = 3.14" in line for line in result.term_out)


def test_interpreter_run_with_file_operations(interpreter):
    code = """
# Create a file
with open('test.txt', 'w') as f:
    f.write('Hello, File!')
    
# Read and print the file
with open('test.txt', 'r') as f:
    content = f.read()
print(f'File content: {content}')
"""
    result = interpreter.run(code)

    assert result.exc_type is None
    assert any("File content: Hello, File!" in line for line in result.term_out)
    assert (interpreter.working_dir / "test.txt").exists()


def test_interpreter_run_timeout(interpreter):
    code = """
import time
time.sleep(10)  # Should timeout
"""
    with tempfile.TemporaryDirectory():
        result = interpreter.run(code)

        assert result.exc_type == "TimeoutError"
        assert any("TimeoutError" in line for line in result.term_out)


def test_interpreter_run_process_killed(interpreter):
    code = """
import sys
sys.exit(1)  # More graceful exit than os._exit
"""
    result = interpreter.run(code)

    assert result.exc_type == "SystemExit"
    assert result.exc_info is not None
