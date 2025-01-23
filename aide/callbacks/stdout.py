import sys
import inspect
from rich.live import Live
from rich.spinner import Spinner
from aide.interpreter import Interpreter


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
    return "\n".join(lines)


async def handle_exit(message, interpreter):
    if message == "/exit":
        await interpreter.cleanup_session()
        sys.exit()


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
        spinner = Spinner("dots", text=f"[magenta]{message}[/magenta]")
        with Live(
            spinner, refresh_per_second=4
        ) as live:  # Adjust refresh rate as needed
            if inspect.iscoroutinefunction(interpreter.run):
                result = await interpreter.run(*args, **kwargs)
            else:
                result = interpreter.run(*args, **kwargs)
            live.update("[bold red]Done Executing the code[/bold red]")
        return result

    return callback
