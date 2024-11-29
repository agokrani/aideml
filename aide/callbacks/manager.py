import inspect
from typing import Callable, Any, Dict

class CallbackManager:
    def __init__(self):
        self.callbacks: Dict[str, Callable[..., Any]] = {}

    def register_callback(self, name: str, callback: Callable[..., Any]):
        """
        Registers a callback function (sync or async) with a specific name.

        Args:
            name (str): The name of the callback.
            callback (Callable): The callback function.
        """
        self.callbacks[name] = callback
    
    def register_callbacks(self, callbacks: Dict[str, Callable[..., Any]]):
        """
        Registers multiple callback functions with specific names.

        Args:
            callbacks (Dict[str, Callable]): A dictionary of callback functions.
        """
        self.callbacks.update(callbacks)

    async def execute_callback(self, name: str, *args, **kwargs) -> Any:
        """
        Executes a registered callback (sync or async) by its name.

        Args:
            name (str): The name of the callback to execute.
            *args, **kwargs: Arguments to pass to the callback.

        Returns:
            Any: The result of the callback, if it exists.
        """
        if name not in self.callbacks:
            raise ValueError(f"No callback registered with name '{name}'")

        callback = self.callbacks[name]
        if inspect.iscoroutinefunction(callback):
            # If the callback is async, await it
            return await callback(*args, **kwargs)
        else:
            # If the callback is sync, execute it directly
            return callback(*args, **kwargs)