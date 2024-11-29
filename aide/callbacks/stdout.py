from aide.interpreter import Interpreter
import asyncio
from rich.live import Live


async def emit_output(output, **kwargs):
    """
    Callback function to emit output to stdout.
    """
    print(output)

# def read_input():
#     """
#     Callback function to read input from stdin.
#     """
#     show = "> "
#     return input(show)

def read_input():
    """
    Callback function to read multi-line input from stdin.
    """
    show = "> "
    lines = []
    while True:
        line = input(show)
        if not line:
            break
        lines.append(line)
    return '\n'.join(lines)

def execute_code(interpreter: Interpreter):
    """
    Callback function to execute code using the interpreter
    with a live "Executing code..." message.
    """
    async def callback(*args, **kwargs):
        """
        Wraps interpreter execution with live updates.
        """
        message = "Executing code..."
        with Live(f"[bold red]{message}[/bold red]", refresh_per_second=4) as live:  # Adjust refresh rate as needed
            result = interpreter.run(*args, **kwargs)
            live.update(f"[bold green]Done Executing the code[/bold green]")
        return result

    return callback