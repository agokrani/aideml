import abc
from pathlib import Path
from aide.journal import Node

class Runtime(abc.ABC):

    @abc.abstractmethod
    async def cleanup_session(self):
        pass

    @abc.abstractmethod
    async def run(self):
        pass
        
    @abc.abstractmethod
    async def cache_best_node(self, node: Node, workspace_dir: Path | str):
        """Cache the best node's submission and solution files.
        
        Args:
            node: The node to cache
            workspace_dir: Directory containing the workspace
        """
        pass
