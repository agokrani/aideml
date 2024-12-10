import sys
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


def handle_exit(message):
    if message == "/exit":
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
            result = interpreter.run(*args, **kwargs)
            live.update("[bold red]Done Executing the code[/bold red]")
        return result

    return callback